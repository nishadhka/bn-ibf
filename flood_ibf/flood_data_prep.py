#!/usr/bin/env -S uv run --with icechunk --with xarray --with "zarr>=3" --with numpy --with pandas --with geopandas --with regionmask --with netcdf4 --with pyarrow --with scipy --with fsspec --with s3fs --with gcsfs --with bottleneck
"""
Flood BN IBF v1 — per-day admin-1 input generator.

Reads:
  - IMERG half-hourly icechunk store (observations)
  - ECMWF TP icechunk store (forecasts)
  - CMORPH return-period NetCDF (pixel-wise thresholds)
  - ICPAC admin-1 GeoJSON

Writes a CSV with one row per admin-1 boundary holding the evidence vector
consumed by flood_bn_ibf_v1.jl:
    id, name, country,
    antecedent_rainfall_mm, antecedent_category,
    rainfall_trend, trend_slope_mm_per_day,
    ecmwf_eprob_heavy, eprob_24h, spatial_coverage,
    forecast_agreement, target_date
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

DURATIONS = ["3hr", "6hr", "12hr", "24hr", "48hr", "72hr", "7day"]
DURATION_HOURS = {"3hr": 3, "6hr": 6, "12hr": 12, "24hr": 24,
                  "48hr": 48, "72hr": 72, "7day": 168}

ISO_TO_COUNTRY = {
    "BDI": "Burundi", "DJI": "Djibouti", "ERI": "Eritrea", "ETH": "Ethiopia",
    "KEN": "Kenya", "RWA": "Rwanda", "SOM": "Somalia", "SSD": "South Sudan",
    "SDN": "Sudan", "TZA": "Tanzania", "UGA": "Uganda",
}


def open_icechunk(prefix: str) -> xr.Dataset:
    storage = ic.s3_storage(
        bucket="e4drr-project",
        prefix=prefix,
        endpoint_url="https://data.source.coop",
        region="us-east-1",
        anonymous=True,
        force_path_style=True,
    )
    repo = ic.Repository.open(storage)
    return xr.open_zarr(
        repo.readonly_session("main").store,
        consolidated=False,
        decode_timedelta=True,
    )


def open_ecmwf_store(pencil: bool) -> xr.Dataset:
    """Select the pencil zarr (per-pixel-member friendly) or the pancake
    icechunk mirror (full-grid friendly). Benchmark 2026-04-16: for the
    current full-init zonal-statistics pipeline, icechunk is ~4× faster;
    the pencil store is the right default once per-pixel soft-evidence
    propagation lands (upgrade #4 deep-path)."""
    if not pencil:
        return open_icechunk("forecasts/ecmwf_ea_tp_icechunk")
    import fsspec
    fs = fsspec.filesystem(
        "s3", anon=True,
        client_kwargs={"endpoint_url": "https://data.source.coop"},
    )
    return xr.open_zarr(
        fs.get_mapper("e4drr-project/forecasts/ecmwf_ea_tp_pencil_zarr"),
        consolidated=False, decode_timedelta=True,
    )


# East-Africa bbox (lat_s, lat_n, lon_w, lon_e) — a touch wider than the
# icpac_adm1v3 extent (21.84-51.42°E, -11.75-23.15°N) so no boundary pixel
# is clipped when we subset the global WeatherBench2 grid.
EA_BBOX = (-12.0, 24.0, 21.0, 52.0)

WB2_IFS_ENS_STORE = "gs://weatherbench2/datasets/ifs_ens/2016-2024-1440x721.zarr"


def open_ifs_ens_wb2(store: str = WB2_IFS_ENS_STORE,
                     bbox: tuple[float, float, float, float] = EA_BBOX,
                     max_lead_h: int = 168) -> xr.Dataset:
    """Open the WeatherBench2 ECMWF IFS-ENS archive (50-member, 0.25°,
    2016-2024, 15-day 6-hourly leads) and normalize it to the same
    interface the icechunk ECMWF store exposes, so the rest of the pipeline
    (ecmwf_window_accums, exceedance/tail aggregation) is unchanged.

    WeatherBench2 schema:  total_precipitation (metres, cumulative-since-init)
      dims = (time=init, number=member, prediction_timedelta=lead, latitude, longitude)
    Target schema:         tp  with dims (init_date, member, lead_time, lat, lon),
      lead_time as timedelta64.  Units stay metres (×1000→mm downstream).
    """
    import gcsfs
    # token="anon" forces anonymous access to the public weatherbench2 bucket;
    # without it gcsfs falls back to ambient ADC, which may lack permission.
    fs = gcsfs.GCSFileSystem(token="anon")
    ds = xr.open_zarr(
        fs.get_mapper(store),
        consolidated=True, decode_timedelta=True,
    )
    lat_s, lat_n, lon_w, lon_e = bbox
    # Orientation-agnostic bbox subset (WB2 lat may be ascending or
    # descending; lon is 0-360 ascending and EA is all positive).
    lat = ds.latitude.values
    lon = ds.longitude.values
    lat_slice = slice(lat_s, lat_n) if lat[0] < lat[-1] else slice(lat_n, lat_s)
    lon_slice = slice(lon_w, lon_e) if lon[0] < lon[-1] else slice(lon_e, lon_w)
    ds = ds.sel(latitude=lat_slice, longitude=lon_slice)
    tp = ds.total_precipitation.rename({
        "time": "init_date", "number": "member",
        "prediction_timedelta": "lead_time",
        "latitude": "lat", "longitude": "lon",
    })
    # Cap the lead axis at the longest duration the pipeline uses (7-day=168 h).
    # This keeps the gap-filling interpolation below (which must load the whole
    # lead axis) from pulling the full 15-day forecast (61 leads → ~29).
    tp = tp.sel(lead_time=slice(np.timedelta64(0, "h"),
                               np.timedelta64(int(max_lead_h), "h")))
    # Ensure lead_time is timedelta64 (decode_timedelta usually handles the
    # 'hours' units; coerce if it arrives as a plain integer hour index).
    if not np.issubdtype(tp.lead_time.dtype, np.timedelta64):
        tp = tp.assign_coords(
            lead_time=tp.lead_time.astype("int64") * np.timedelta64(1, "h"))
    # WB2's raw cumulative total_precipitation has scattered all-NaN lead steps
    # (missing de-accumulation steps, different per init — e.g. 6 h always, plus
    # an init-dependent handful like 24/72/120/150 h). Because the field is
    # cumulative-since-init and therefore monotonic non-decreasing in lead, we
    # fill those interior gaps by linear interpolation over lead_time so every
    # duration accumulation is well-defined (instead of intermittently NaN).
    # interpolate_na fills interior gaps; ffill/bfill handle a NaN at the first
    # or last retained lead (e.g. 168 h itself missing) — carrying the nearest
    # valid cumulative value is a small, conservative fill given monotonicity.
    tp = (tp.chunk({"lead_time": -1})
            .interpolate_na(dim="lead_time", method="linear")
            .ffill(dim="lead_time").bfill(dim="lead_time"))
    return tp.to_dataset(name="tp")


# Soft-evidence binning: mirrors the Julia categorize_* cutoffs in
# flood_bn_ibf_v1.jl so the one-hot limit of these vectors reproduces the
# legacy hard-classification. Sigmas are ~30% of the narrowest bin spacing
# and can be tuned per-node if/when we plug in real physical uncertainty
# (IMERG retrieval noise, ensemble sampling std, Gumbel-fit posterior, …).
_NODE_EDGES = {
    "ant":  [-np.inf, 10.0, 30.0, 60.0, 100.0, np.inf],
    "exc":  [-np.inf, 0.2, 0.4, 0.6, 0.8, np.inf],
    "spa":  [-np.inf, 0.3, 0.6, np.inf],
    "trn":  [-np.inf, -2.0, 2.0, np.inf],
    "tail": [-np.inf, 0.5, 1.0, 2.0, np.inf],
}
_NODE_SIGMA_DEFAULT = {"ant": 10.0, "exc": 0.05, "spa": 0.05, "trn": 1.0, "tail": 0.15}


def soft_bin(x: float, node: str, sigma: float | None = None) -> np.ndarray:
    from scipy import stats as _st
    edges = _NODE_EDGES[node]
    k = len(edges) - 1
    if not np.isfinite(x):
        return np.full(k, 1.0 / k)
    s = _NODE_SIGMA_DEFAULT[node] if sigma is None else sigma
    probs = np.diff(_st.norm.cdf(edges, loc=x, scale=s))
    tot = probs.sum()
    return probs / tot if tot > 0 else np.full(k, 1.0 / k)


def add_soft_columns(df: pd.DataFrame,
                     ant_mm: np.ndarray, exc: np.ndarray,
                     spa: np.ndarray, trn_slope: np.ndarray,
                     tail_ratio: np.ndarray) -> None:
    """In-place: add 5+5+3+3+4=20 soft-evidence columns (ant/exc/spa/trn/tail)."""
    blocks = [("ant", ant_mm, 5), ("exc", exc, 5), ("spa", spa, 3),
              ("trn", trn_slope, 3), ("tail", tail_ratio, 4)]
    for node, vals, k in blocks:
        probs = np.vstack([soft_bin(float(v), node) for v in vals])
        for i in range(k):
            df[f"{node}_p{i+1}"] = np.round(probs[:, i], 4)


def imerg_daily_totals(imerg: xr.Dataset, date_utc: pd.Timestamp) -> xr.DataArray:
    start = pd.Timestamp(date_utc) - pd.Timedelta(days=7)
    end = pd.Timestamp(date_utc) - pd.Timedelta(seconds=1)
    hh = imerg.precipitation.sel(time=slice(start, end))  # mm/hr
    # IMERG encodes missing retrievals as the fill value -9999.9 (not decoded to
    # NaN by the store). Mask any non-physical negative before accumulating, or
    # those fills sum into huge negative antecedent totals (recent operational
    # dates are clean; historical weeks have missing swaths). skipna sums leave
    # fully-missing days at 0.
    hh = hh.where(hh >= 0.0)
    mm = hh * 0.5  # half-hour → mm
    daily = mm.resample(time="1D").sum()
    return daily.astype("float32")


def ecmwf_window_accums(ecmwf: xr.Dataset, init_date: pd.Timestamp) -> dict[str, xr.DataArray]:
    tp = ecmwf.tp.sel(init_date=init_date)  # (member, lead_time, lat, lon) in metres
    lt = tp.lead_time.values
    out: dict[str, xr.DataArray] = {}
    for dur, h in DURATION_HOURS.items():
        td = np.timedelta64(h, "h")
        idx_arr = np.where(lt == td)[0]
        if idx_arr.size == 0:
            idx = int(np.argmin(np.abs(lt - td)))
        else:
            idx = int(idx_arr[0])
        out[dur] = (tp.isel(lead_time=idx) * 1000.0).astype("float32")  # → mm
    return out


def _select_cmorph_rp(ds: xr.Dataset, rp_year: int) -> dict[str, xr.DataArray]:
    rp = ds.return_period_precip.sel(return_period=rp_year)
    if float(rp.lat[0]) > float(rp.lat[-1]):
        rp = rp.isel(lat=slice(None, None, -1))
    if float(rp.lon[0]) > float(rp.lon[-1]):
        rp = rp.isel(lon=slice(None, None, -1))
    return {dur: rp.sel(duration=dur).drop_vars("duration") for dur in DURATIONS}


def load_cmorph_thresholds(path: str, rp_year: int) -> dict[str, xr.DataArray]:
    """Pixel-wise RP thresholds from a local CMORPH return-period NetCDF."""
    return _select_cmorph_rp(xr.open_dataset(path), rp_year)


def load_cmorph_thresholds_icechunk(prefix: str, rp_year: int) -> dict[str, xr.DataArray]:
    """Pixel-wise RP thresholds from the source.coop CMORPH RP icechunk store
    (observations/cmorph_rp_icechunk). Mirrors the drought prep's
    era5_ecmwf_rp_icechunk path; `return_period_precip` has dims
    (duration, return_period, lat, lon)."""
    return _select_cmorph_rp(open_icechunk(prefix), rp_year)


def regrid_to(da_src: xr.DataArray, lat_target: xr.DataArray,
              lon_target: xr.DataArray) -> xr.DataArray:
    lat_asc = np.sort(lat_target.values)
    lon_asc = np.sort(lon_target.values)
    interp = da_src.interp(lat=lat_asc, lon=lon_asc, method="nearest")
    return interp.reindex(lat=lat_target.values, lon=lon_target.values)


def build_mask(gdf: gpd.GeoDataFrame, lat: xr.DataArray, lon: xr.DataArray) -> xr.DataArray:
    regions = regionmask.Regions(
        outlines=list(gdf.geometry),
        numbers=list(range(len(gdf))),
        names=list(gdf["NAME_1"]),
        abbrevs=list(gdf["GID_1"]),
        name="adm1",
    )
    return regions.mask(lon, lat)


def zonal_reduce(da: xr.DataArray, mask: xr.DataArray, lat: xr.DataArray,
                 n_regions: int, thresh: float | None = None) -> np.ndarray:
    """Area-weighted mean (or fraction ≥ thresh) per region. NaN where empty."""
    weights = np.cos(np.deg2rad(lat))
    w2d = weights.broadcast_like(da)
    src = (da >= thresh).astype("float32") if thresh is not None else da
    valid = (~da.isnull()).astype("float32")
    mask_vals = mask.values
    src_vals = src.values
    w_vals = w2d.values
    v_vals = valid.values
    out = np.full(n_regions, np.nan, dtype=np.float64)
    for r in range(n_regions):
        sel = mask_vals == r
        if not sel.any():
            continue
        w = w_vals[sel] * v_vals[sel]
        den = w.sum()
        if den <= 0:
            continue
        num = float((src_vals[sel] * w).sum())
        out[r] = num / float(den)
    return out


def zonal_quantile(da: xr.DataArray, mask: xr.DataArray, n_regions: int,
                   q: float = 0.95) -> np.ndarray:
    """Per-region q-th quantile of pixel values (unweighted). NaN if empty."""
    mask_vals = mask.values
    vals = da.values
    out = np.full(n_regions, np.nan, dtype=np.float64)
    for r in range(n_regions):
        sel = mask_vals == r
        if not sel.any():
            continue
        v = vals[sel]
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        out[r] = float(np.quantile(v, q))
    return out


def zonal_max(da: xr.DataArray, mask: xr.DataArray, n_regions: int) -> np.ndarray:
    """Per-region maximum of pixel values. NaN if empty."""
    mask_vals = mask.values
    vals = da.values
    out = np.full(n_regions, np.nan, dtype=np.float64)
    for r in range(n_regions):
        sel = mask_vals == r
        if not sel.any():
            continue
        v = vals[sel]
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        out[r] = float(np.max(v))
    return out


def fill_small_boundaries(values: np.ndarray, da: xr.DataArray,
                          gdf: gpd.GeoDataFrame, thresh: float | None = None) -> np.ndarray:
    """For boundaries with no pixel hit, sample nearest pixel at centroid."""
    out = values.copy()
    missing = np.where(np.isnan(out))[0]
    if len(missing) == 0:
        return out
    cent = gdf.iloc[missing].geometry.centroid
    src = (da >= thresh).astype("float32") if thresh is not None else da
    for pos, (i, pt) in enumerate(zip(missing, cent)):
        try:
            val = float(src.sel(lat=pt.y, lon=pt.x, method="nearest").values)
        except Exception:
            val = np.nan
        out[i] = val
    return out


def compute_per_member_ratios(
    accums: dict[str, xr.DataArray],
    thresh_ec: dict[str, xr.DataArray],
    mask: xr.DataArray,
    adm1: gpd.GeoDataFrame,
    n_regions: int,
) -> pd.DataFrame:
    """
    For each (boundary, member) pair, compute the max-over-durations of the
    pixel p95 of (accum_mm / threshold_mm). This produces per-member
    storyline material: which specific members project threshold-crossing
    at which boundaries.

    Returns a long-form DataFrame with columns:
        boundary_id, boundary_name, country, member, max_ratio, tail_state
    """
    def _tail(ratio: float) -> str:
        if not np.isfinite(ratio): return "Nil"
        if ratio < 0.5: return "Nil"
        if ratio < 1.0: return "Low"
        if ratio < 2.0: return "Moderate"
        return "High"

    # per-member, per-pixel ratio across durations → single grid per member
    durations = list(accums.keys())
    members = accums[durations[0]].member.values

    # Stack duration-level ratios then max per pixel per member
    n_mem = len(members)
    n_lat = accums[durations[0]].sizes["lat"]
    n_lon = accums[durations[0]].sizes["lon"]
    per_member_ratio = np.zeros((n_mem, n_lat, n_lon), dtype="float32")
    for dur in durations:
        a = accums[dur].values           # (member, lat, lon)
        t = thresh_ec[dur].values        # (lat, lon)
        safe_t = np.where(t > 0, t, np.inf)
        r = a / safe_t[None, :, :]
        per_member_ratio = np.maximum(per_member_ratio, r)

    # Zonal p95 per (boundary, member)
    mask_vals = mask.values
    rows = []
    iso_to_country = ISO_TO_COUNTRY
    for r_idx in range(n_regions):
        sel = mask_vals == r_idx
        if not sel.any():
            # Centroid fallback: pick nearest pixel
            pt = adm1.iloc[r_idx].geometry.centroid
            lat_vals = accums[durations[0]].lat.values
            lon_vals = accums[durations[0]].lon.values
            i = int(np.argmin(np.abs(lat_vals - pt.y)))
            j = int(np.argmin(np.abs(lon_vals - pt.x)))
            member_ratios = per_member_ratio[:, i, j]
        else:
            # Pixel-p95 per member across boundary pixels
            pixels = per_member_ratio[:, sel]  # (member, n_pix)
            member_ratios = np.quantile(pixels, 0.95, axis=1)

        gid = adm1.iloc[r_idx]["GID_1"]
        nm = adm1.iloc[r_idx]["NAME_1"]
        cc = iso_to_country.get(gid.split(".")[0], "Unknown")
        for m_idx, m in enumerate(members):
            rv = float(member_ratios[m_idx])
            rows.append({
                "boundary_id": gid,
                "boundary_name": nm,
                "country": cc,
                "member": str(m),
                "max_ratio": round(rv, 4),
                "tail_state": _tail(rv),
            })
    return pd.DataFrame(rows)


def compute_per_member_evidence(
    accums: dict[str, xr.DataArray],
    thresh_ec: dict[str, xr.DataArray],
    mask: xr.DataArray,
    adm1: gpd.GeoDataFrame,
    n_regions: int,
    antecedent_mm: np.ndarray,
    slopes: np.ndarray,
    trend_band: float,
    target_date: str,
    soft: bool = False,
) -> pd.DataFrame:
    """Full per-member evidence for storyline BN runs.
    For each (boundary, member) emit: antecedent (shared), trend (shared),
    per-member exceedance fraction, spatial coverage, max_ratio, and optional
    soft-evidence columns.
    """
    durations = list(accums.keys())
    members = accums[durations[0]].member.values
    n_mem = len(members)
    n_lat = accums[durations[0]].sizes["lat"]
    n_lon = accums[durations[0]].sizes["lon"]
    mask_vals = mask.values

    # Per-member, per-pixel: max-over-durations of (accum / threshold)
    per_member_ratio = np.zeros((n_mem, n_lat, n_lon), dtype="float32")
    # Per-member, per-pixel: does any duration exceed threshold? (binary)
    per_member_exceed = np.zeros((n_mem, n_lat, n_lon), dtype="float32")
    for dur in durations:
        a = accums[dur].values           # (member, lat, lon)
        t = thresh_ec[dur].values        # (lat, lon)
        safe_t = np.where(t > 0, t, np.inf)
        r = a / safe_t[None, :, :]
        per_member_ratio = np.maximum(per_member_ratio, r)
        per_member_exceed = np.maximum(per_member_exceed,
                                        (a >= t[None, :, :]).astype("float32"))

    rows = []
    for r_idx in range(n_regions):
        sel = mask_vals == r_idx
        gid = adm1.iloc[r_idx]["GID_1"]
        nm = adm1.iloc[r_idx]["NAME_1"]
        cc = ISO_TO_COUNTRY.get(gid.split(".")[0], "Unknown")
        ant_mm_val = float(antecedent_mm[r_idx])
        slope_val = float(slopes[r_idx])
        trend_str = classify_trend(slope_val, trend_band)

        for m_idx, m in enumerate(members):
            if sel.any():
                pix_ratio = per_member_ratio[m_idx, sel]
                pix_exceed = per_member_exceed[m_idx, sel]
                mratio = float(np.quantile(pix_ratio[np.isfinite(pix_ratio)], 0.95)) \
                    if np.isfinite(pix_ratio).any() else 0.0
                mexc = float(np.mean(pix_exceed))
                mspa = float(np.mean(pix_exceed >= 0.5)) if pix_exceed.size > 0 else 0.0
            else:
                # centroid fallback
                pt = adm1.iloc[r_idx].geometry.centroid
                lat_v = accums[durations[0]].lat.values
                lon_v = accums[durations[0]].lon.values
                i = int(np.argmin(np.abs(lat_v - pt.y)))
                j = int(np.argmin(np.abs(lon_v - pt.x)))
                mratio = float(per_member_ratio[m_idx, i, j])
                mexc = float(per_member_exceed[m_idx, i, j])
                mspa = mexc

            row = {
                "boundary_id": gid, "boundary_name": nm, "country": cc,
                "member": str(m), "target_date": target_date,
                "antecedent_rainfall_mm": round(ant_mm_val, 3),
                "rainfall_trend": trend_str,
                "trend_slope_mm_per_day": round(slope_val, 3),
                "member_exc_frac": round(mexc, 4),
                "member_spa_cov": round(mspa, 4),
                "member_max_ratio": round(mratio, 4),
            }
            if soft:
                for node, val, k in [("ant", ant_mm_val, 5), ("exc", mexc, 5),
                                      ("spa", mspa, 3), ("trn", slope_val, 3),
                                      ("tail", mratio, 4)]:
                    probs = soft_bin(val, node)
                    for ki in range(k):
                        row[f"{node}_p{ki+1}"] = round(float(probs[ki]), 4)
            rows.append(row)
    return pd.DataFrame(rows)


def classify_trend(slope: float, band: float) -> str:
    if not np.isfinite(slope):
        return "Stable"
    if slope > band:
        return "Increasing"
    if slope < -band:
        return "Decreasing"
    return "Stable"


def imerg_union_daily_adm(imerg: xr.Dataset, imerg_mask, lat, n_adm: int,
                          start: pd.Timestamp, end: pd.Timestamp) -> dict:
    """Pre-load the IMERG daily admin-1 totals for the whole union window
    [start, end) ONCE (consecutive target days' 7-day antecedent windows
    overlap ~85%, so per-day reloading is wasteful). Returns {date -> (n_adm,)}
    array of zonal-mean daily totals (mm). Mirrors imerg_daily_totals' masking."""
    hh = imerg.precipitation.sel(time=slice(start, end - pd.Timedelta(seconds=1)))
    hh = hh.where(hh >= 0.0)              # mask -9999.9 fill
    daily = (hh * 0.5).resample(time="1D").sum().load()
    dates = pd.to_datetime(daily.time.values)
    out = {}
    for di in range(daily.sizes["time"]):
        adm = zonal_reduce(daily.isel(time=di), imerg_mask, lat, n_adm)
        out[pd.Timestamp(dates[di]).normalize()] = adm
    return out


def process_one_date(D, init_ts, out_path: Path, args, *, adm1, n_adm, country,
                     daily_adm_union, ecmwf, thresh_ec, ec_mask, ref_lat) -> bool:
    """Compute and write the per-day evidence CSV for target date D using the
    pre-opened stores, pre-built masks, and pre-computed IMERG union totals.
    Returns True on success."""
    # ---------------- IMERG antecedent (from pre-loaded union) ----------------
    # 7-day window [D-7, D): calendar days D-7 .. D-1, in chronological order.
    wdays = [(pd.Timestamp(D).normalize() - pd.Timedelta(days=k)) for k in range(7, 0, -1)]
    daily_adm = np.vstack([daily_adm_union.get(d, np.full(n_adm, np.nan)) for d in wdays])
    antecedent_mm = np.nansum(daily_adm, axis=0)
    antecedent_mm[np.isnan(daily_adm).all(axis=0)] = np.nan
    x = np.arange(daily_adm.shape[0], dtype=np.float64)
    slopes = np.full(n_adm, np.nan)
    for i in range(n_adm):
        y = daily_adm[:, i]
        if np.isfinite(y).all():
            slopes[i] = float(np.polyfit(x, y, 1)[0])
    trend_cls = np.array([classify_trend(s, args.trend_band) for s in slopes])

    # ---------------- ECMWF / IFS-ENS exceedance ----------------
    # Load the (gap-filled) forecast for this init ONCE, then slice each
    # duration from memory — calling .load() per-duration would otherwise
    # re-run the whole lead-axis interpolation 7× (the dominant per-day cost).
    tp_init = ecmwf.tp.sel(init_date=init_ts).load()  # (member, lead_time, lat, lon) mm-of-m
    lt = tp_init.lead_time.values
    accums = {}
    for dur, h in DURATION_HOURS.items():
        td = np.timedelta64(h, "h")
        idx_arr = np.where(lt == td)[0]
        idx = int(idx_arr[0]) if idx_arr.size else int(np.argmin(np.abs(lt - td)))
        accums[dur] = (tp_init.isel(lead_time=idx) * 1000.0).astype("float32")  # → mm

    eprob = {}
    ens_max_ratio_per_dur = {}
    for dur in DURATIONS:
        exceeds = (accums[dur] >= thresh_ec[dur]).astype("float32")
        eprob[dur] = exceeds.mean(dim="member")
        ens_max_mm = accums[dur].max(dim="member")
        safe_thresh = thresh_ec[dur].where(thresh_ec[dur] > 0, 1.0)
        ens_max_ratio_per_dur[dur] = ens_max_mm / safe_thresh
    eprob_24 = eprob["24hr"]
    p_heavy = xr.concat([eprob[d] for d in DURATIONS], dim="duration").max("duration")
    max_ratio = xr.concat([ens_max_ratio_per_dur[d] for d in DURATIONS],
                          dim="duration").max("duration")
    ens_mean_24h = accums["24hr"].mean(dim="member")
    ens_max_24h = accums["24hr"].max(dim="member")
    ens_min_24h = accums["24hr"].min(dim="member")

    eprob_heavy_adm = zonal_reduce(p_heavy, ec_mask, ref_lat, n_adm)
    eprob_24h_adm = zonal_reduce(eprob_24, ec_mask, ref_lat, n_adm)
    spatial_cov_adm = zonal_reduce(p_heavy, ec_mask, ref_lat, n_adm, thresh=0.5)
    max_ratio_mean_adm = zonal_reduce(max_ratio, ec_mask, ref_lat, n_adm)
    max_ratio_p95_adm = zonal_quantile(max_ratio, ec_mask, n_adm, q=0.95)
    max_ratio_peak_adm = zonal_max(max_ratio, ec_mask, n_adm)
    hotspot_frac_adm = zonal_reduce(max_ratio, ec_mask, ref_lat, n_adm, thresh=1.0)
    ens_mean_24h_adm = zonal_reduce(ens_mean_24h, ec_mask, ref_lat, n_adm)
    ens_max_24h_adm = zonal_reduce(ens_max_24h, ec_mask, ref_lat, n_adm)
    ens_min_24h_adm = zonal_reduce(ens_min_24h, ec_mask, ref_lat, n_adm)

    eprob_heavy_adm = fill_small_boundaries(eprob_heavy_adm, p_heavy, adm1)
    eprob_24h_adm = fill_small_boundaries(eprob_24h_adm, eprob_24, adm1)
    spatial_cov_adm = fill_small_boundaries(spatial_cov_adm, p_heavy, adm1, thresh=0.5)
    max_ratio_mean_adm = fill_small_boundaries(max_ratio_mean_adm, max_ratio, adm1)
    max_ratio_p95_adm = fill_small_boundaries(max_ratio_p95_adm, max_ratio, adm1)
    max_ratio_peak_adm = fill_small_boundaries(max_ratio_peak_adm, max_ratio, adm1)
    hotspot_frac_adm = fill_small_boundaries(hotspot_frac_adm, max_ratio, adm1, thresh=1.0)
    ens_mean_24h_adm = fill_small_boundaries(ens_mean_24h_adm, ens_mean_24h, adm1)
    ens_max_24h_adm = fill_small_boundaries(ens_max_24h_adm, ens_max_24h, adm1)
    ens_min_24h_adm = fill_small_boundaries(ens_min_24h_adm, ens_min_24h, adm1)

    # ---------------- Assemble output ----------------
    spatial_cov_final = np.fmax(spatial_cov_adm, hotspot_frac_adm)
    df = pd.DataFrame({
        "id": adm1["GID_1"],
        "name": adm1["NAME_1"],
        "country": country,
        "antecedent_rainfall_mm": np.round(antecedent_mm, 3),
        "antecedent_category": "",
        "rainfall_trend": trend_cls,
        "trend_slope_mm_per_day": np.round(slopes, 3),
        "ecmwf_eprob_heavy": np.round(eprob_heavy_adm, 4),
        "eprob_24h": np.round(eprob_24h_adm, 4),
        "spatial_coverage": np.round(spatial_cov_final, 4),
        "spatial_cov_mean_p": np.round(spatial_cov_adm, 4),
        "hotspot_fraction": np.round(hotspot_frac_adm, 4),
        "forecast_agreement": "Medium",
        "ens_max_ratio": np.round(max_ratio_p95_adm, 4),
        "ens_max_ratio_mean": np.round(max_ratio_mean_adm, 4),
        "ens_max_ratio_peak": np.round(max_ratio_peak_adm, 4),
        "ens_mean_24h_mm": np.round(ens_mean_24h_adm, 2),
        "ens_max_24h_mm": np.round(ens_max_24h_adm, 2),
        "ens_min_24h_mm": np.round(ens_min_24h_adm, 2),
        "target_date": str(D.date()),
    })
    if args.soft_evidence:
        add_soft_columns(df, ant_mm=antecedent_mm, exc=eprob_heavy_adm,
                         spa=spatial_cov_final, trn_slope=slopes,
                         tail_ratio=max_ratio_p95_adm)

    # A handful of WB2 IFS-ENS inits are entirely missing (all-NaN tp, e.g.
    # 2019-10-17), which leaves the forecast-derived numeric columns NaN and
    # would reach the Julia BN as `missing` (Float64(::Missing) crash). The
    # soft-evidence columns already encode this as a uniform vector (the correct
    # "no information" handling); fill the remaining hard numeric NaNs with 0 so
    # the CSV has no missing cells. Antecedent (IMERG) is unaffected on such days.
    num_cols = df.select_dtypes(include=[np.number]).columns
    n_nan = int(df[num_cols].isna().to_numpy().sum())
    if n_nan:
        df[num_cols] = df[num_cols].fillna(0.0)
        print(f"[prep] {D.date()}: filled {n_nan} NaN numeric cells with 0 "
              f"(missing forecast/obs — soft evidence stays uniform)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[prep] {D.date()} -> {out_path.name}  rows={len(df)}  "
          f"ant_mean={np.nanmean(antecedent_mm):.1f}mm  "
          f"heavy_mean={np.nanmean(eprob_heavy_adm):.3f}")

    if args.member_evidence_sidecar:
        me_df = compute_per_member_evidence(
            accums, thresh_ec, ec_mask, adm1, n_adm,
            antecedent_mm, slopes, args.trend_band,
            target_date=str(D.date()), soft=args.soft_evidence)
        me_path = Path(args.member_evidence_sidecar)
        me_path.parent.mkdir(parents=True, exist_ok=True)
        me_df.to_csv(me_path, index=False)
        print(f"[prep] wrote member-evidence sidecar {me_path}  rows={len(me_df)}")
    if args.member_sidecar:
        member_df = compute_per_member_ratios(accums, thresh_ec, ec_mask, adm1, n_adm)
        sc = Path(args.member_sidecar)
        sc.parent.mkdir(parents=True, exist_ok=True)
        member_df.to_csv(sc, index=False)
        print(f"[prep] wrote member sidecar {sc}  rows={len(member_df)}")
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="Target date D (YYYY-MM-DD); "
                    "start date when --end-date is given")
    ap.add_argument("--end-date", default=None,
                    help="If set, process the inclusive range --date..--end-date "
                         "in one process (opens stores + builds masks once — much "
                         "faster than 15 separate invocations). Requires --out-dir.")
    ap.add_argument("--out-dir", default=None,
                    help="Output directory for range mode; per-day files are "
                         "written as flood_inputs_<D>[_soft].csv.")
    ap.add_argument("--rp-years", type=int, default=2)
    ap.add_argument("--out", default=None,
                    help="Output CSV for single-date mode (required without --end-date).")
    ap.add_argument("--adm1", default="icpac_adm1v3.geojson")
    ap.add_argument("--cmorph-rp", default="cmorph_ea_return_periods.nc",
                    help="Local CMORPH return-period NetCDF (used only when "
                         "--cmorph-source netcdf).")
    ap.add_argument("--cmorph-source", choices=["icechunk", "netcdf"],
                    default="icechunk",
                    help="Where to read CMORPH RP thresholds from. Default "
                         "'icechunk' reads observations/cmorph_rp_icechunk on "
                         "source.coop (no local .nc needed).")
    ap.add_argument("--cmorph-rp-prefix",
                    default="observations/cmorph_rp_icechunk",
                    help="icechunk store prefix for CMORPH RP thresholds.")
    ap.add_argument("--trend-band", type=float, default=2.0)
    ap.add_argument("--member-sidecar", default=None,
                    help="Optional per-member sidecar CSV path (long format)")
    ap.add_argument("--soft-evidence", action="store_true",
                    help="Emit Gaussian-soft-binned probability columns "
                         "{ant,exc,spa,trn,tail}_p{1..K} alongside the hard class")
    ap.add_argument("--pencil", action="store_true",
                    help="Read ECMWF from the pencil-chunked zarr mirror "
                         "(forecasts/ecmwf_ea_tp_pencil_zarr) instead of the icechunk store")
    ap.add_argument("--forecast-source",
                    choices=["ecmwf_icechunk", "ifs_ens_wb2"],
                    default="ecmwf_icechunk",
                    help="Forecast precip store. Default 'ecmwf_icechunk' is the "
                         "operational source.coop ECMWF ENS. 'ifs_ens_wb2' reads "
                         "the WeatherBench2 archived ECMWF IFS-ENS (50-member, "
                         "0.25°, 2016-2024) for historical-event hindcasts.")
    ap.add_argument("--forecast-init-hour", type=int, default=0,
                    help="Forecast init hour (UTC) selected from the store, e.g. "
                         "0 for the 00Z init (default). WB2 IFS-ENS has 00Z/12Z.")
    ap.add_argument("--member-evidence-sidecar", default=None,
                    help="Enriched per-member sidecar CSV with full 5-parent evidence "
                         "for storyline BN runs (one row per boundary × member)")
    args = ap.parse_args()

    # ---- resolve date list + output paths (single vs range mode) ----
    suffix = "_soft" if args.soft_evidence else ""
    if args.end_date:
        if not args.out_dir:
            ap.error("--end-date requires --out-dir")
        dates = list(pd.date_range(args.date, args.end_date, freq="D"))
        out_dir = Path(args.out_dir)
        out_for = {D: out_dir / f"flood_inputs_{D.date()}{suffix}.csv" for D in dates}
    else:
        if not args.out:
            ap.error("--out is required in single-date mode (no --end-date)")
        dates = [pd.Timestamp(args.date)]
        out_for = {dates[0]: Path(args.out)}
    print(f"[prep] {len(dates)} date(s) {dates[0].date()}..{dates[-1].date()}  "
          f"RP={args.rp_years}yr  band=±{args.trend_band} mm/day  src={args.forecast_source}")

    # ================= one-time setup (date-independent) =================
    adm1 = gpd.read_file(args.adm1).reset_index(drop=True)
    n_adm = len(adm1)
    country = (adm1["GID_1"].str.split(".").str[0]
               .map(ISO_TO_COUNTRY).fillna("Unknown"))
    print(f"[prep] adm1 boundaries: {n_adm}")

    print("[prep] opening IMERG icechunk...")
    imerg = open_icechunk("observations/imerg_hh_icechunk")
    imerg_mask = build_mask(adm1, imerg.lat, imerg.lon)
    # Pre-load the IMERG daily admin-1 totals for the whole antecedent union
    # window [min(dates)-7, max(dates)) ONCE (per-day reloading overlaps ~85%).
    u_start = (dates[0].normalize() - pd.Timedelta(days=7))
    u_end = dates[-1].normalize()
    print(f"[prep] pre-loading IMERG union {u_start.date()}..{u_end.date()} "
          f"({(u_end - u_start).days} days)...")
    daily_adm_union = imerg_union_daily_adm(imerg, imerg_mask, imerg.lat, n_adm,
                                            u_start, u_end)

    if args.forecast_source == "ifs_ens_wb2":
        print("[prep] opening WeatherBench2 ECMWF IFS-ENS (50-member archive)...")
        ecmwf = open_ifs_ens_wb2()
    else:
        print(f"[prep] opening ECMWF {'pencil zarr' if args.pencil else 'icechunk'}...")
        ecmwf = open_ecmwf_store(args.pencil)
    init_dates = pd.to_datetime(ecmwf.init_date.values)

    if args.cmorph_source == "icechunk":
        print(f"[prep] CMORPH RP from icechunk: {args.cmorph_rp_prefix}")
        thresh = load_cmorph_thresholds_icechunk(args.cmorph_rp_prefix, args.rp_years)
    else:
        print(f"[prep] CMORPH RP from NetCDF: {args.cmorph_rp}")
        thresh = load_cmorph_thresholds(args.cmorph_rp, args.rp_years)
    ref_lat, ref_lon = ecmwf.tp.lat, ecmwf.tp.lon
    thresh_ec = {dur: regrid_to(thresh[dur], ref_lat, ref_lon).load() for dur in DURATIONS}
    ec_mask = build_mask(adm1, ref_lat, ref_lon)
    print("[prep] one-time setup done (masks + CMORPH thresholds)")

    # ================= per-date loop =================
    n_ok = 0
    for D in dates:
        init_ts = (D + pd.Timedelta(hours=args.forecast_init_hour)
                   if args.forecast_source == "ifs_ens_wb2" else D)
        if init_ts not in init_dates:
            print(f"[prep] SKIP {D.date()}: init {init_ts} not in forecast store "
                  f"(range {init_dates.min()}..{init_dates.max()})")
            continue
        try:
            n_ok += int(process_one_date(
                D, init_ts, out_for[D], args,
                adm1=adm1, n_adm=n_adm, country=country,
                daily_adm_union=daily_adm_union, ecmwf=ecmwf,
                thresh_ec=thresh_ec, ec_mask=ec_mask, ref_lat=ref_lat))
        except Exception as e:  # one bad day shouldn't sink the whole range
            print(f"[prep] ERROR {D.date()}: {type(e).__name__}: {e}")
    print(f"[prep] done: {n_ok}/{len(dates)} dates written")


if __name__ == "__main__":
    main()
