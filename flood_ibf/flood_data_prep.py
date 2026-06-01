#!/usr/bin/env -S uv run --with icechunk --with xarray --with "zarr>=3" --with numpy --with pandas --with geopandas --with regionmask --with netcdf4 --with pyarrow --with scipy --with fsspec --with s3fs
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="Target date D (YYYY-MM-DD)")
    ap.add_argument("--rp-years", type=int, default=2)
    ap.add_argument("--out", required=True)
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
    ap.add_argument("--member-evidence-sidecar", default=None,
                    help="Enriched per-member sidecar CSV with full 5-parent evidence "
                         "for storyline BN runs (one row per boundary × member)")
    args = ap.parse_args()

    D = pd.Timestamp(args.date)
    print(f"[prep] D={D.date()}  RP={args.rp_years}yr  band=±{args.trend_band} mm/day")

    adm1 = gpd.read_file(args.adm1).reset_index(drop=True)
    n_adm = len(adm1)
    print(f"[prep] adm1 boundaries: {n_adm}")

    # ---------------- IMERG antecedent ----------------
    print("[prep] opening IMERG icechunk...")
    imerg = open_icechunk("observations/imerg_hh_icechunk")
    daily = imerg_daily_totals(imerg, D).load()
    t0 = pd.to_datetime(daily.time.values[0]).date()
    t1 = pd.to_datetime(daily.time.values[-1]).date()
    print(f"[prep] IMERG 7-day totals {t0}..{t1}  shape={daily.shape}")

    imerg_mask = build_mask(adm1, daily.lat, daily.lon)

    daily_adm = np.full((daily.sizes["time"], n_adm), np.nan, dtype=np.float64)
    for di in range(daily.sizes["time"]):
        daily_adm[di] = zonal_reduce(daily.isel(time=di), imerg_mask, daily.lat, n_adm)

    antecedent_mm = np.nansum(daily_adm, axis=0)
    antecedent_mm[np.isnan(daily_adm).all(axis=0)] = np.nan

    x = np.arange(daily.sizes["time"], dtype=np.float64)
    slopes = np.full(n_adm, np.nan)
    for i in range(n_adm):
        y = daily_adm[:, i]
        if np.isfinite(y).all():
            slopes[i] = float(np.polyfit(x, y, 1)[0])
    trend_cls = np.array([classify_trend(s, args.trend_band) for s in slopes])

    # ---------------- ECMWF exceedance ----------------
    print(f"[prep] opening ECMWF {'pencil zarr' if args.pencil else 'icechunk'}...")
    ecmwf = open_ecmwf_store(args.pencil)
    init_dates = pd.to_datetime(ecmwf.init_date.values)
    if D not in init_dates:
        raise SystemExit(f"[prep] init_date {D.date()} not in ECMWF store "
                         f"(range {init_dates.min().date()}..{init_dates.max().date()})")

    accums = ecmwf_window_accums(ecmwf, D)
    for k in list(accums):
        accums[k] = accums[k].load()
    print(f"[prep] ECMWF accums loaded for {list(accums)}")

    if args.cmorph_source == "icechunk":
        print(f"[prep] CMORPH RP from icechunk: {args.cmorph_rp_prefix}")
        thresh = load_cmorph_thresholds_icechunk(args.cmorph_rp_prefix, args.rp_years)
    else:
        print(f"[prep] CMORPH RP from NetCDF: {args.cmorph_rp}")
        thresh = load_cmorph_thresholds(args.cmorph_rp, args.rp_years)
    ref = accums["24hr"].isel(member=0)
    thresh_ec = {dur: regrid_to(thresh[dur], ref.lat, ref.lon) for dur in DURATIONS}

    eprob = {}
    ens_max_ratio_per_dur = {}
    for dur in DURATIONS:
        exceeds = (accums[dur] >= thresh_ec[dur]).astype("float32")
        eprob[dur] = exceeds.mean(dim="member")
        ens_max_mm = accums[dur].max(dim="member")
        ens_min_mm = accums[dur].min(dim="member")
        safe_thresh = thresh_ec[dur].where(thresh_ec[dur] > 0, 1.0)
        ens_max_ratio_per_dur[dur] = ens_max_mm / safe_thresh
    eprob_24 = eprob["24hr"]
    p_heavy = xr.concat([eprob[d] for d in DURATIONS], dim="duration").max("duration")

    # Tail risk: max across durations of (ens_max / threshold) per pixel
    max_ratio = xr.concat([ens_max_ratio_per_dur[d] for d in DURATIONS],
                          dim="duration").max("duration")

    # Ensemble mean and max at 24h for diagnostics
    ens_mean_24h = accums["24hr"].mean(dim="member")
    ens_max_24h = accums["24hr"].max(dim="member")
    ens_min_24h = accums["24hr"].min(dim="member")

    ec_mask = build_mask(adm1, ref.lat, ref.lon)
    eprob_heavy_adm = zonal_reduce(p_heavy, ec_mask, ref.lat, n_adm)
    eprob_24h_adm = zonal_reduce(eprob_24, ec_mask, ref.lat, n_adm)
    spatial_cov_adm = zonal_reduce(p_heavy, ec_mask, ref.lat, n_adm, thresh=0.5)

    # Pixel-level tail aggregation (upgrade from boundary-mean)
    max_ratio_mean_adm = zonal_reduce(max_ratio, ec_mask, ref.lat, n_adm)
    max_ratio_p95_adm = zonal_quantile(max_ratio, ec_mask, n_adm, q=0.95)
    max_ratio_peak_adm = zonal_max(max_ratio, ec_mask, n_adm)
    hotspot_frac_adm = zonal_reduce(max_ratio, ec_mask, ref.lat, n_adm, thresh=1.0)

    ens_mean_24h_adm = zonal_reduce(ens_mean_24h, ec_mask, ref.lat, n_adm)
    ens_max_24h_adm = zonal_reduce(ens_max_24h, ec_mask, ref.lat, n_adm)
    ens_min_24h_adm = zonal_reduce(ens_min_24h, ec_mask, ref.lat, n_adm)

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
    country = (adm1["GID_1"].str.split(".").str[0]
               .map(ISO_TO_COUNTRY).fillna("Unknown"))

    # Spatial coverage now blends the classical P_heavy mask with the
    # pixel-level hotspot fraction (pixels where any member exceeds threshold).
    # Use the max of the two so localized hotspots aren't smoothed away.
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
        "ens_max_ratio": np.round(max_ratio_p95_adm, 4),  # now p95 (pixel-level)
        "ens_max_ratio_mean": np.round(max_ratio_mean_adm, 4),
        "ens_max_ratio_peak": np.round(max_ratio_peak_adm, 4),
        "ens_mean_24h_mm": np.round(ens_mean_24h_adm, 2),
        "ens_max_24h_mm": np.round(ens_max_24h_adm, 2),
        "ens_min_24h_mm": np.round(ens_min_24h_adm, 2),
        "target_date": str(D.date()),
    })

    if args.soft_evidence:
        add_soft_columns(df,
                         ant_mm     = antecedent_mm,
                         exc        = eprob_heavy_adm,
                         spa        = spatial_cov_final,
                         trn_slope  = slopes,
                         tail_ratio = max_ratio_p95_adm)
        print(f"[prep] soft-evidence columns added (20 cols)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[prep] wrote {out}  rows={len(df)}  cols={len(df.columns)}  "
          f"ant_mean={np.nanmean(antecedent_mm):.1f}mm  "
          f"heavy_mean={np.nanmean(eprob_heavy_adm):.3f}")

    if args.member_evidence_sidecar:
        me_df = compute_per_member_evidence(
            accums, thresh_ec, ec_mask, adm1, n_adm,
            antecedent_mm, slopes, args.trend_band,
            target_date=str(D.date()),
            soft=args.soft_evidence,
        )
        me_path = Path(args.member_evidence_sidecar)
        me_path.parent.mkdir(parents=True, exist_ok=True)
        me_df.to_csv(me_path, index=False)
        n_crossing = (me_df["member_max_ratio"] >= 1.0).sum()
        print(f"[prep] wrote member-evidence sidecar {me_path}  rows={len(me_df)}  "
              f"threshold_crossing={n_crossing} ({n_crossing/len(me_df)*100:.1f}%)")

    if args.member_sidecar:
        member_df = compute_per_member_ratios(accums, thresh_ec, ec_mask, adm1, n_adm)
        sidecar_path = Path(args.member_sidecar)
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        member_df.to_csv(sidecar_path, index=False)
        n_crossing = (member_df["max_ratio"] >= 1.0).sum()
        n_rows = len(member_df)
        print(f"[prep] wrote member sidecar {sidecar_path}  rows={n_rows}  "
              f"threshold_crossing_members={n_crossing} ({n_crossing/n_rows*100:.1f}%)")


if __name__ == "__main__":
    main()
