#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "icechunk==2.0.3",
#   "xarray",
#   "zarr>=3",
#   "numpy",
#   "pandas",
#   "geopandas",
#   "regionmask",
#   "rasterio",
#   "fsspec",
#   "s3fs",
#   "scipy",
# ]
# ///
"""
CDI data prep — derive the JRC-style 14-class Combined Drought Indicator at
ICPAC admin-1 from one or both of two parallel source paths:

  recompute  CHIRPS SPI + GDO SMA + GDO fAPAR icechunk slabs → calculate_cdi()
             per pixel (the original Path B-α evidence channel).
  eadw       Pre-computed ICPAC East African Drought Watch dekadal CDI slabs
             from icpac_cdi_dekadal_icechunk — read directly, no rule eval.

Reads (all anonymous on source.coop, us-west-2):
  - chirps_spi_icechunk            spi1, spi3, spi9 (or spi12)        monthly
  - gdo_sma_icechunk               smang (soil-moisture anomaly)      dekadal
  - gdo_fpar_icechunk              fpanv (vegetation anomaly)         dekadal
  - icpac_cdi_dekadal_icechunk     cdi  (ICPAC EADW operational CDI)  dekadal

CDI class encoding (shared by both paths, per drought_crma/cdi-method.md):
  0           No_drought
  1, 2, 3     Watch              (precipitation deficit only)
  4, 5, 6     Warning            (+ soil-moisture deficit)
  7, 8, 9, 10 Alert              (+ vegetation anomaly)
  11, 12      Partial_recovery
  13, 14      Full_recovery

Output (CSV): one row per admin-1 boundary. With `--cdi-source recompute`
or `eadw` the columns are unprefixed (cdi_class, cdi_level, …, cdi_source);
with `--cdi-source both` the columns are duplicated as `*_recomp` and
`*_eadw`, plus a boolean `cdi_agreement` flag.

Usage:
    # original behaviour — recomputed CDI from raw component stores
    uv run cdi_data_prep.py \\
        --date 2026-04 --adm1 icpac_adm1v3.geojson \\
        --out /tmp/cdi_inputs_2026-04.csv

    # ICPAC EADW pre-computed CDI (cheaper, single-store read)
    uv run cdi_data_prep.py --cdi-source eadw \\
        --date 2026-04 --adm1 icpac_adm1v3.geojson \\
        --out /tmp/cdi_inputs_eadw_2026-04.csv

    # Both — wide CSV with cdi_*_recomp and cdi_*_eadw columns + agreement
    uv run cdi_data_prep.py --cdi-source both \\
        --date 2026-04 --adm1 icpac_adm1v3.geojson \\
        --out /tmp/cdi_inputs_both_2026-04.csv

The output CSV joins on `id` with the CSV from `drought_data_prep.py`.
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import geopandas as gpd
import icechunk as ic
import numpy as np
import pandas as pd
import regionmask
import xarray as xr

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

S3_BUCKET            = "us-west-2.opendata.source.coop"
S3_REGION            = "us-west-2"
CHIRPS_SPI_PREFIX    = "e4drr-project/observations/chirps_spi_icechunk"
GDO_SMA_PREFIX       = "e4drr-project/observations/gdo_sma_icechunk"
GDO_FPAR_PREFIX       = "e4drr-project/observations/gdo_fpar_icechunk"        # GDO MERIS+OLCI operational, 2012-01–now
GDO_FPAR_MODIS_PREFIX = "e4drr-project/observations/gdo_fpar_modis_icechunk"   # GDO fAPAR-MODIS backfill, 2001-01–2015-12
                                                                               #   produced by ibf-thresholds-triggers/thresholds/hf-gdo/gdo_fpar_modis_icechunk.py

# Year boundaries used by the auto-selection logic. The GDO operational
# product (MERIS+OLCI) starts 2012-01; the MODIS backfill covers 2001-01
# through 2015-12 (with 2012-2015 overlap). Both stores expose the same
# fAPAR-anomaly variable on the same EA grid, so calculate_cdi() runs
# unchanged regardless of which one supplies the field.
GDO_FPAR_OPERATIONAL_START = pd.Timestamp("2012-02-01")  # store's first dekad is 2012-01-21
GDO_FPAR_MODIS_START       = pd.Timestamp("2001-01-01")
GDO_FPAR_MODIS_END         = pd.Timestamp("2015-12-31")
EADW_CDI_PREFIX      = "e4drr-project/observations/icpac_cdi_dekadal_icechunk"

# 14-class → 6-level lookup, shared by recompute and EADW paths.
# Index = cdi_class (0..14), value = level idx (1=No_drought .. 6=Alert).
CLASS_TO_LEVEL_IDX = np.array([
    1,           # 0  : No_drought
    4, 4, 4,     # 1, 2, 3   : Watch
    5, 5, 5,     # 4, 5, 6   : Warning
    6, 6, 6, 6,  # 7-10      : Alert
    3, 3,        # 11, 12    : Partial_recovery
    2, 2,        # 13, 14    : Full_recovery
], dtype=np.int16)

ISO_TO_COUNTRY = {
    "BDI": "Burundi", "DJI": "Djibouti", "ERI": "Eritrea", "ETH": "Ethiopia",
    "KEN": "Kenya", "RWA": "Rwanda", "SOM": "Somalia", "SSD": "South Sudan",
    "SDN": "Sudan", "TZA": "Tanzania", "UGA": "Uganda",
}

# CDI level numbering (1=least, 6=most stressed). Order matches the BN's
# CDI_STATES = ["No_drought", "Full_recovery", "Partial_recovery",
#               "Watch", "Warning", "Alert"]
LEVEL_TO_IDX = {
    "No_drought":       1,
    "Full_recovery":    2,
    "Partial_recovery": 3,
    "Watch":            4,
    "Warning":          5,
    "Alert":            6,
}
IDX_TO_LEVEL = {v: k for k, v in LEVEL_TO_IDX.items()}


# ─── store openers ───────────────────────────────────────────────────────────


def open_icechunk_anon(prefix: str) -> xr.Dataset:
    storage = ic.s3_storage(
        bucket=S3_BUCKET, prefix=prefix, region=S3_REGION, anonymous=True,
    )
    repo = ic.Repository.open(storage, config=ic.RepositoryConfig.default())
    return xr.open_zarr(
        repo.readonly_session("main").store, consolidated=False,
    )


def class_grid_to_level_idx(cdi_class: np.ndarray) -> np.ndarray:
    """Map an integer cdi_class array (0..14) to its 6-level idx via lookup.
    Out-of-range values fall back to No_drought (1) so a stray NaN that has
    been cast to e.g. -2147483648 doesn't poison the aggregation."""
    cls = cdi_class.astype(np.int32, copy=False)
    cls = np.where((cls >= 0) & (cls <= 14), cls, 0)
    return CLASS_TO_LEVEL_IDX[cls]


def read_eadw_cdi(D: pd.Timestamp) -> tuple[
    np.ndarray, np.ndarray, xr.DataArray, xr.DataArray, pd.Timestamp,
]:
    """Read the latest dekad ≤ D from icpac_cdi_dekadal_icechunk.

    Returns (cdi_class[lat,lon] int16, cdi_level_idx[lat,lon] int16,
             lat DataArray, lon DataArray, dekad_timestamp).
    """
    print("[cdi-prep] opening ICPAC EADW CDI icechunk ...", flush=True)
    eadw_ds = open_icechunk_anon(EADW_CDI_PREFIX)
    idx = latest_le_idx(eadw_ds.time.values, D)
    t = pd.Timestamp(eadw_ds.time.values[idx])
    print(f"[cdi-prep] EADW dekad: {t.date()}  (store has "
          f"{eadw_ds.sizes['time']} timesteps)", flush=True)
    arr = eadw_ds.cdi.isel(time=idx).load().values
    cdi_class = np.where(np.isfinite(arr), arr, 0).astype(np.int16)
    cdi_class = np.clip(cdi_class, 0, 14)
    cdi_level_idx = class_grid_to_level_idx(cdi_class).astype(np.int16)
    return cdi_class, cdi_level_idx, eadw_ds.lat, eadw_ds.lon, t


# ─── calculate_cdi (port of cdi-method.md::calculate_cdi) ───────────────────


def calculate_cdi(
    spi9_12_lt_m1: bool, spi3_lt_m1: bool, spi1_lt_m2: bool,
    spi3_prev_lt_m1: bool, spi1_prev_lt_m2: bool,
    sma_lt_m1: bool, fapar_lt_m1: bool,
) -> tuple[int, str]:
    """Return (cdi_class 1..14, level_string) per drought_crma/cdi-method.md."""

    # Alert: precipitation shortage + vegetation anomaly
    if fapar_lt_m1 and sma_lt_m1 and spi9_12_lt_m1 and spi3_lt_m1:
        return 10, "Alert"
    if fapar_lt_m1 and sma_lt_m1 and spi3_lt_m1:
        return 9, "Alert"
    if fapar_lt_m1 and spi3_lt_m1:
        return 8, "Alert"
    if fapar_lt_m1 and spi1_lt_m2:
        return 7, "Alert"

    # Warning: precipitation shortage + soil moisture anomaly
    if sma_lt_m1 and spi9_12_lt_m1 and spi3_lt_m1:
        return 6, "Warning"
    if sma_lt_m1 and spi3_lt_m1:
        return 5, "Warning"
    if sma_lt_m1 and spi1_lt_m2:
        return 4, "Warning"

    # Watch: precipitation shortage only
    if spi9_12_lt_m1 and spi3_lt_m1:
        return 3, "Watch"
    if spi3_lt_m1:
        return 2, "Watch"
    if spi1_lt_m2:
        return 1, "Watch"

    # Partial recovery: previous precipitation deficit + vegetation anomaly
    if fapar_lt_m1 and spi3_prev_lt_m1:
        return 12, "Partial_recovery"
    if fapar_lt_m1 and spi1_prev_lt_m2:
        return 11, "Partial_recovery"

    # Full recovery: previous precipitation deficit only
    if spi3_prev_lt_m1:
        return 14, "Full_recovery"
    if spi1_prev_lt_m2:
        return 13, "Full_recovery"

    return 0, "No_drought"


def calculate_cdi_grid(
    spi9_12: np.ndarray, spi3: np.ndarray, spi1: np.ndarray,
    spi3_prev: np.ndarray, spi1_prev: np.ndarray,
    sma: np.ndarray, fapar: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised pixel-wise CDI. All inputs are 2-D arrays on the same grid.

    Returns (cdi_class[H, W] uint8, cdi_level_idx[H, W] uint8). Pixels with any
    NaN among the inputs become class 0 / level 1 (No_drought) — the rule
    table says "no condition met" and we conservatively treat missing data
    as "no information" rather than bias toward severe.
    """
    # Boolean condition grids
    c_spi9_12 = np.isfinite(spi9_12) & (spi9_12 < -1.0)
    c_spi3    = np.isfinite(spi3)    & (spi3    < -1.0)
    c_spi1    = np.isfinite(spi1)    & (spi1    < -2.0)
    c_spi3_p  = np.isfinite(spi3_prev) & (spi3_prev < -1.0)
    c_spi1_p  = np.isfinite(spi1_prev) & (spi1_prev < -2.0)
    c_sma     = np.isfinite(sma)     & (sma     < -1.0)
    c_fapar   = np.isfinite(fapar)   & (fapar   < -1.0)

    h, w = spi3.shape
    cdi_class = np.zeros((h, w), dtype=np.uint8)
    cdi_level_idx = np.full((h, w), LEVEL_TO_IDX["No_drought"], dtype=np.uint8)

    # Apply rules in priority order — same as calculate_cdi() above.
    # Each rule writes into pixels that haven't matched a higher-priority rule.
    matched = np.zeros((h, w), dtype=bool)

    def assign(mask, cls: int, level: str):
        nonlocal matched
        sel = mask & ~matched
        cdi_class[sel] = cls
        cdi_level_idx[sel] = LEVEL_TO_IDX[level]
        matched |= sel

    # Alert
    assign(c_fapar & c_sma & c_spi9_12 & c_spi3, 10, "Alert")
    assign(c_fapar & c_sma & c_spi3,              9, "Alert")
    assign(c_fapar & c_spi3,                      8, "Alert")
    assign(c_fapar & c_spi1,                      7, "Alert")
    # Warning
    assign(c_sma & c_spi9_12 & c_spi3,            6, "Warning")
    assign(c_sma & c_spi3,                        5, "Warning")
    assign(c_sma & c_spi1,                        4, "Warning")
    # Watch
    assign(c_spi9_12 & c_spi3,                    3, "Watch")
    assign(c_spi3,                                2, "Watch")
    assign(c_spi1,                                1, "Watch")
    # Partial recovery
    assign(c_fapar & c_spi3_p,                    12, "Partial_recovery")
    assign(c_fapar & c_spi1_p,                    11, "Partial_recovery")
    # Full recovery
    assign(c_spi3_p,                              14, "Full_recovery")
    assign(c_spi1_p,                              13, "Full_recovery")

    return cdi_class, cdi_level_idx


# ─── slice helpers ───────────────────────────────────────────────────────────


def latest_le_idx(times: np.ndarray, target: pd.Timestamp) -> int:
    """Index of the most recent timestamp ≤ target."""
    ts = pd.to_datetime(times)
    upper_mask = ts <= target
    if not upper_mask.any():
        raise SystemExit(f"No time slice ≤ {target.date()} in dataset")
    return int(np.where(upper_mask)[0].max())


def previous_month_idx(times: np.ndarray, latest_idx: int) -> int:
    """Index roughly 1 month before times[latest_idx]."""
    ts = pd.to_datetime(times)
    target = ts[latest_idx] - pd.DateOffset(months=1)
    upper_mask = ts <= target
    if not upper_mask.any():
        return latest_idx
    return int(np.where(upper_mask)[0].max())


def regrid_to(
    da_src: xr.DataArray, lat_target: xr.DataArray, lon_target: xr.DataArray,
) -> xr.DataArray:
    lat_asc = np.sort(lat_target.values)
    lon_asc = np.sort(lon_target.values)
    interp = da_src.interp(lat=lat_asc, lon=lon_asc, method="nearest")
    return interp.reindex(lat=lat_target.values, lon=lon_target.values)


# ─── adm1 zonal aggregation ──────────────────────────────────────────────────


def build_mask(
    gdf: gpd.GeoDataFrame, lat: xr.DataArray, lon: xr.DataArray,
) -> np.ndarray:
    """Per-pixel region index (int32, -1 = no region).

    Uses rasterio.features.rasterize directly. regionmask's 2D mask path
    internally allocates an `(lon, lat, n_regions)` int64 array which OOMs
    on the EADW grid (3480×2950×227 ≈ 17 GiB)."""
    from rasterio.features import rasterize
    from rasterio.transform import from_origin

    lat_arr = np.asarray(lat.values if hasattr(lat, "values") else lat)
    lon_arr = np.asarray(lon.values if hasattr(lon, "values") else lon)
    n_lat, n_lon = len(lat_arr), len(lon_arr)
    res_lat = abs(float(lat_arr[1] - lat_arr[0]))
    res_lon = abs(float(lon_arr[1] - lon_arr[0]))
    west  = float(lon_arr.min()) - res_lon / 2.0
    north = float(lat_arr.max()) + res_lat / 2.0
    transform = from_origin(west, north, res_lon, res_lat)
    shapes = [(geom, idx) for idx, geom in enumerate(gdf.geometry)]
    # rasterize emits a top-down (max-lat first) array; flip if our data
    # rows are ascending in lat.
    mask = rasterize(
        shapes, out_shape=(n_lat, n_lon),
        transform=transform, fill=-1, dtype=np.int32,
        all_touched=False,
    )
    if lat_arr[0] < lat_arr[-1]:
        mask = mask[::-1, :]
    return mask


def aggregate_per_boundary(
    cdi_class: np.ndarray, cdi_level_idx: np.ndarray,
    mask_arr: np.ndarray, n_regions: int,
    gdf: gpd.GeoDataFrame, lat_vals: np.ndarray, lon_vals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """For each boundary return (max_class, max_level_idx, modal_class, level_fraction)."""
    max_class    = np.zeros(n_regions, dtype=np.int16)
    max_level    = np.full(n_regions, LEVEL_TO_IDX["No_drought"], dtype=np.int16)
    modal_class  = np.zeros(n_regions, dtype=np.int16)
    level_frac   = np.zeros(n_regions, dtype=np.float64)

    for r in range(n_regions):
        sel = mask_arr == r
        if not sel.any():
            # Centroid fallback
            pt = gdf.iloc[r].geometry.centroid
            i = int(np.argmin(np.abs(lat_vals - pt.y)))
            j = int(np.argmin(np.abs(lon_vals - pt.x)))
            max_class[r]   = int(cdi_class[i, j])
            max_level[r]   = int(cdi_level_idx[i, j])
            modal_class[r] = max_class[r]
            level_frac[r]  = 1.0
            continue

        cls_pixels = cdi_class[sel]
        lvl_pixels = cdi_level_idx[sel]

        max_level[r] = int(lvl_pixels.max())
        # Worst class within the worst level
        in_max = lvl_pixels == max_level[r]
        max_class[r] = int(cls_pixels[in_max].max()) if in_max.any() else 0
        # Modal class (most frequent)
        vals, counts = np.unique(cls_pixels, return_counts=True)
        modal_class[r] = int(vals[np.argmax(counts)])
        # Fraction of pixels at the max level
        level_frac[r] = float(in_max.sum()) / float(sel.sum())

    return max_class, max_level, modal_class, level_frac


# ─── main ────────────────────────────────────────────────────────────────────


def _open_fapar_for(target: pd.Timestamp, mode: str
                    ) -> tuple[xr.Dataset | None, str]:
    """Pick the fAPAR icechunk store appropriate for `target` and `mode`.

    Returns (fapar_dataset, source_label). When mode resolves to "none"
    (target predates all fAPAR coverage, or the user forced --fapar-source
    none), returns (None, "none") so the calling code can pass an
    all-NaN/all-False fAPAR mask and let calculate_cdi() degrade
    gracefully through the SMA-only / SPI-only branches of the rule
    cascade.

    Modes:
      "auto"  : MODIS for inits before 2012-01; GDO operational from
                2012-01 onwards. If target < 2001-01 → returns ("none").
      "gdo"   : force GDO operational. Errors out if target < 2012-01.
      "modis" : force GDO-MODIS backfill. Errors out if target outside
                2001-01..2015-12.
      "none"  : skip fAPAR entirely; CDI rule cascade falls through to
                SMA-only / SPI-only branches.
    """
    if mode == "none":
        print("[cdi-prep] fAPAR disabled by --fapar-source=none "
              "(rule cascade will degrade to SMA-only / SPI-only branches)",
              flush=True)
        return (None, "none")

    if mode == "gdo":
        if target < GDO_FPAR_OPERATIONAL_START:
            raise SystemExit(
                f"[cdi-prep] --fapar-source=gdo requires target >= "
                f"{GDO_FPAR_OPERATIONAL_START.date()}, got {target.date()}. "
                f"Use --fapar-source=auto or modis for older targets, or "
                f"--fapar-source=none to skip fAPAR.")
        print("[cdi-prep] opening GDO fAPAR (operational MERIS+OLCI) icechunk ...",
              flush=True)
        return (open_icechunk_anon(GDO_FPAR_PREFIX), "gdo")

    if mode == "modis":
        if not (GDO_FPAR_MODIS_START <= target <= GDO_FPAR_MODIS_END):
            raise SystemExit(
                f"[cdi-prep] --fapar-source=modis requires target in "
                f"[{GDO_FPAR_MODIS_START.date()}, {GDO_FPAR_MODIS_END.date()}], "
                f"got {target.date()}. Use --fapar-source=auto or gdo for "
                f"out-of-range targets, or --fapar-source=none to skip fAPAR.")
        print("[cdi-prep] opening GDO fAPAR-MODIS backfill icechunk ...",
              flush=True)
        return (open_icechunk_anon(GDO_FPAR_MODIS_PREFIX), "modis")

    # mode == "auto"
    if target >= GDO_FPAR_OPERATIONAL_START:
        print(f"[cdi-prep] auto: target >= {GDO_FPAR_OPERATIONAL_START.date()} → using GDO fAPAR (operational)",
              flush=True)
        return (open_icechunk_anon(GDO_FPAR_PREFIX), "gdo")
    if target >= GDO_FPAR_MODIS_START:
        print(f"[cdi-prep] auto: {GDO_FPAR_MODIS_START.date()} <= target < {GDO_FPAR_OPERATIONAL_START.date()} → using GDO fAPAR-MODIS backfill",
              flush=True)
        return (open_icechunk_anon(GDO_FPAR_MODIS_PREFIX), "modis")
    print(f"[cdi-prep] auto: target {target.date()} predates all fAPAR coverage "
          f"(MODIS starts {GDO_FPAR_MODIS_START.date()}); falling back to "
          f"SPI+SMA-only CDI (max class 6, Warning level)", flush=True)
    return (None, "none")


def _aggregate_recompute(
    D: pd.Timestamp, adm1: gpd.GeoDataFrame, spi_long: str,
    fapar_source: str = "auto",
) -> tuple[dict, dict]:
    """Run the original recompute path and return per-boundary stats + provenance.

    `fapar_source` selects the fAPAR icechunk store: see _open_fapar_for().
    """
    print("[cdi-prep] opening CHIRPS SPI icechunk ...", flush=True)
    spi_ds = open_icechunk_anon(CHIRPS_SPI_PREFIX)
    print("[cdi-prep] opening GDO SMA icechunk ...", flush=True)
    sma_ds = open_icechunk_anon(GDO_SMA_PREFIX)
    fp_ds, fapar_label = _open_fapar_for(D, fapar_source)

    spi_idx      = latest_le_idx(spi_ds.time.values, D)
    spi_prev_idx = previous_month_idx(spi_ds.time.values, spi_idx)
    sma_idx      = latest_le_idx(sma_ds.time.values, D)
    spi_t      = pd.Timestamp(spi_ds.time.values[spi_idx])
    sma_t      = pd.Timestamp(sma_ds.time.values[sma_idx])
    fp_t: pd.Timestamp | None = None
    if fp_ds is not None:
        fp_idx = latest_le_idx(fp_ds.time.values, D)
        fp_t   = pd.Timestamp(fp_ds.time.values[fp_idx])
    print(f"[cdi-prep] recompute time slices: SPI={spi_t.date()} "
          f"SMA={sma_t.date()} fAPAR={fp_t.date() if fp_t else 'N/A'} "
          f"(fapar_source={fapar_label})", flush=True)

    spi3      = spi_ds.spi3.isel(time=spi_idx).load()
    spi1      = spi_ds.spi1.isel(time=spi_idx).load()
    spi9_12   = spi_ds[spi_long].isel(time=spi_idx).load()
    spi3_prev = spi_ds.spi3.isel(time=spi_prev_idx).load()
    spi1_prev = spi_ds.spi1.isel(time=spi_prev_idx).load()
    sma       = sma_ds.smang.isel(time=sma_idx).load()
    if fp_ds is not None:
        # GDO operational store names the variable 'fpanv'; the GDO-MODIS
        # backfill store names it 'fapan' (per the upstream Copernicus GDO
        # MODIS NetCDFs). Auto-detect.
        fapar_var = next((v for v in ("fpanv", "fapan") if v in fp_ds.data_vars), None)
        if fapar_var is None:
            raise SystemExit(
                f"[cdi-prep] no fAPAR variable in {fapar_label} store; "
                f"expected 'fpanv' (GDO) or 'fapan' (MODIS), got {list(fp_ds.data_vars)}")
        fapar = fp_ds[fapar_var].isel(time=fp_idx).load()
    else:
        # No fAPAR for this target. Build an all-False mask on the CHIRPS
        # grid so calculate_cdi_grid()'s `fapar_lt_m1` is uniformly False
        # — every Alert / Partial-recovery rule branch (which requires
        # fapar_lt_m1) will fail and the cascade naturally degrades to
        # the Warning / Watch / Full-recovery branches that depend only
        # on SMA + SPI. The output `cdi_class` will therefore be ≤ 6
        # (Warning) or in {13, 14} (Full_recovery) for the affected
        # months — exactly the documented graceful-degradation behaviour.
        fapar = xr.full_like(spi3, fill_value=0.0, dtype="float32")
        fapar = fapar.assign_attrs(units="anomaly_proxy",
                                    note="fAPAR unavailable; calculate_cdi sees "
                                         "fapar_lt_m1 = False everywhere")

    print("[cdi-prep] regridding SMA + fAPAR to CHIRPS grid ...", flush=True)
    sma_rg   = regrid_to(sma,   spi3.lat, spi3.lon)
    fapar_rg = (fapar if fp_ds is None else regrid_to(fapar, spi3.lat, spi3.lon))

    print(f"[cdi-prep] computing CDI grid (recompute, fapar_source={fapar_label}) ...",
          flush=True)
    cdi_class, cdi_level_idx = calculate_cdi_grid(
        spi9_12.values, spi3.values, spi1.values,
        spi3_prev.values, spi1_prev.values,
        sma_rg.values, fapar_rg.values,
    )

    print("[cdi-prep] building admin-1 mask (CHIRPS grid) ...", flush=True)
    mask = build_mask(adm1, spi3.lat, spi3.lon)
    print("[cdi-prep] aggregating per boundary (recompute) ...", flush=True)
    max_class, max_level, modal_class, level_frac = aggregate_per_boundary(
        cdi_class, cdi_level_idx, mask, len(adm1), adm1,
        spi3.lat.values, spi3.lon.values,
    )
    return (
        dict(max_class=max_class, max_level=max_level,
             modal_class=modal_class, level_frac=level_frac),
        dict(spi_time=str(spi_t.date()), sma_time=str(sma_t.date()),
             fapar_time=(str(fp_t.date()) if fp_t else "none"),
             fapar_source=fapar_label),
    )


def _aggregate_eadw(
    D: pd.Timestamp, adm1: gpd.GeoDataFrame,
) -> tuple[dict, dict]:
    """Read ICPAC EADW dekadal CDI directly and aggregate per boundary."""
    cdi_class, cdi_level_idx, lat_da, lon_da, t = read_eadw_cdi(D)

    print("[cdi-prep] building admin-1 mask (EADW grid) ...", flush=True)
    mask = build_mask(adm1, lat_da, lon_da)
    print("[cdi-prep] aggregating per boundary (EADW) ...", flush=True)
    max_class, max_level, modal_class, level_frac = aggregate_per_boundary(
        cdi_class, cdi_level_idx, mask, len(adm1), adm1,
        lat_da.values, lon_da.values,
    )
    return (
        dict(max_class=max_class, max_level=max_level,
             modal_class=modal_class, level_frac=level_frac),
        dict(eadw_dekad=str(t.date())),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="Target month (YYYY-MM or YYYY-MM-DD)")
    ap.add_argument("--adm1", required=True)
    ap.add_argument("--out",  required=True)
    ap.add_argument("--spi-long", default="spi9", choices=["spi9", "spi12"],
                    help="Long-window SPI variable name (default spi9)")
    ap.add_argument("--cdi-source", default="recompute",
                    choices=["recompute", "eadw", "both"],
                    help="Source of CDI: recompute path (default), pre-computed "
                         "ICPAC EADW, or both side-by-side with agreement flag")
    ap.add_argument("--fapar-source", default="auto",
                    choices=["auto", "gdo", "modis", "none"],
                    help="fAPAR icechunk to use within the recompute path: "
                         "'auto' picks GDO operational (>=2012-01) or GDO-MODIS "
                         "backfill (2001-01..2011-12), 'gdo' / 'modis' force a "
                         "single source, 'none' skips fAPAR entirely (cdi rule "
                         "cascade degrades to SPI+SMA, max class = Warning).")
    args = ap.parse_args()

    D = pd.Timestamp(args.date).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    print(f"[cdi-prep] target month: {D.date()}  cdi-source: {args.cdi_source}",
          flush=True)
    if args.cdi_source != "eadw":
        print(f"[cdi-prep] long-SPI: {args.spi_long}", flush=True)

    adm1 = gpd.read_file(args.adm1).reset_index(drop=True)
    n_adm = len(adm1)
    print(f"[cdi-prep] adm1: {n_adm} boundaries", flush=True)

    country = (adm1["GID_1"].str.split(".").str[0]
               .map(ISO_TO_COUNTRY).fillna("Unknown"))

    base = pd.DataFrame({
        "id":          adm1["GID_1"],
        "name":        adm1["NAME_1"],
        "country":     country,
        "target_date": str(D.date()),
    })

    def _columns(stats: dict, suffix: str = "") -> dict:
        s = suffix
        return {
            f"cdi_class{s}":          stats["max_class"],
            f"cdi_level_idx{s}":      stats["max_level"],
            f"cdi_level{s}":          [IDX_TO_LEVEL[int(i)] for i in stats["max_level"]],
            f"cdi_level_fraction{s}": np.round(stats["level_frac"], 4),
            f"cdi_modal_class{s}":    stats["modal_class"],
        }

    if args.cdi_source == "recompute":
        stats, prov = _aggregate_recompute(D, adm1, args.spi_long,
                                            fapar_source=args.fapar_source)
        df = pd.concat([base, pd.DataFrame(_columns(stats))], axis=1)
        df["cdi_source"] = "recomputed"
        for k, v in prov.items():
            df[k] = v
    elif args.cdi_source == "eadw":
        stats, prov = _aggregate_eadw(D, adm1)
        df = pd.concat([base, pd.DataFrame(_columns(stats))], axis=1)
        df["cdi_source"] = "eadw"
        for k, v in prov.items():
            df[k] = v
    else:  # both
        rc_stats, rc_prov = _aggregate_recompute(D, adm1, args.spi_long,
                                                  fapar_source=args.fapar_source)
        ew_stats, ew_prov = _aggregate_eadw(D, adm1)
        df = pd.concat([
            base,
            pd.DataFrame(_columns(rc_stats, "_recomp")),
            pd.DataFrame(_columns(ew_stats, "_eadw")),
        ], axis=1)
        df["cdi_agreement"] = df["cdi_level_idx_recomp"].astype(int) == df["cdi_level_idx_eadw"].astype(int)
        df["cdi_source"] = "recomputed+eadw"
        for k, v in rc_prov.items():
            df[k] = v
        for k, v in ew_prov.items():
            df[k] = v

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[cdi-prep] wrote {out}  rows={len(df)}", flush=True)

    if args.cdi_source == "both":
        n_agree = int(df["cdi_agreement"].sum())
        print(f"[cdi-prep] level agreement: {n_agree}/{len(df)} boundaries "
              f"({n_agree/len(df):.0%})", flush=True)
        for label, col in [("recompute", "cdi_level_recomp"),
                            ("eadw",      "cdi_level_eadw")]:
            print(f"[cdi-prep] per-boundary CDI level ({label}): "
                  f"{df[col].value_counts().to_dict()}", flush=True)
    else:
        print(f"[cdi-prep] per-boundary CDI level: "
              f"{df['cdi_level'].value_counts().to_dict()}", flush=True)


if __name__ == "__main__":
    main()
