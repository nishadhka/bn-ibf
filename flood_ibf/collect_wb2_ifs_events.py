#!/usr/bin/env -S uv run --with icechunk --with xarray --with "zarr>=3" --with numpy --with pandas --with geopandas --with regionmask --with scipy --with fsspec --with s3fs --with gcsfs --with netcdf4 --with pyyaml --with bottleneck python
"""
Collect the WeatherBench2 archived ECMWF IFS-ENS forecast for the 11 historical
flood-event windows in flood_events.yaml.

For each event window (15 init dates, 00Z), extracts the East-Africa-bbox
total_precipitation (all 50 members) at the pipeline's clean duration leads
(12h/24h/48h/72h/7day — 3h/6h are unavailable at WB2's 6-hourly cadence),
converts to mm, computes the per-pixel exceedance fraction vs the CMORPH 2-yr
return period, and writes:

  wb2_ifs_ens/<key>/forecast_accums.nc   (init_date, duration, member, lat, lon) mm
                                          + exceedance_frac, p_heavy, ens_max_ratio
  wb2_ifs_ens/<key>/summary.csv          per-init-date domain summary
  wb2_ifs_ens/events_summary.csv         all events, per-init-date, one table

This is the forecast-collection / Step-1-precursor stage: everything the BN
needs from the forecast side, on the WB2 grid, minus the admin-1 zonal step
(which needs icpac_adm1v3.geojson).
"""
from __future__ import annotations
import argparse, datetime as dt, warnings
from pathlib import Path

import numpy as np, pandas as pd, xarray as xr, yaml

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import importlib.util
_spec = importlib.util.spec_from_file_location("fdp", str(Path(__file__).with_name("flood_data_prep.py")))
fdp = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(fdp)

# WB2 IFS-ENS is 6-hourly: 3h snaps to lead-0 (=0) and the 6h lead is all-NaN
# (a known de-accumulation gap). Keep only the clean, physical durations.
CLEAN_DURATIONS = ["12hr", "24hr", "48hr", "72hr", "7day"]


def window_dates(peak: dt.date, pre: int, post: int) -> list[pd.Timestamp]:
    start = peak - dt.timedelta(days=pre)
    return [pd.Timestamp(start + dt.timedelta(days=i)) for i in range(pre + post + 1)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default="flood_events.yaml")
    ap.add_argument("--out-dir", default="wb2_ifs_ens")
    ap.add_argument("--rp-years", type=int, default=2)
    ap.add_argument("--only", nargs="*", default=None, help="subset of event keys")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.events))
    d = cfg["defaults"]
    events = cfg["events"]
    if args.only:
        events = [e for e in events if e["key"] in args.only]

    print("[collect] opening WB2 IFS-ENS ...", flush=True)
    ds = fdp.open_ifs_ens_wb2()
    init_index = pd.to_datetime(ds.init_date.values)
    lead_h = (ds.lead_time.values / np.timedelta64(1, "h")).astype(int)

    print(f"[collect] opening CMORPH {args.rp_years}-yr RP (icechunk) ...", flush=True)
    thresh = fdp.load_cmorph_thresholds_icechunk("observations/cmorph_rp_icechunk", args.rp_years)
    ref = ds.tp.isel(init_date=0, member=0, lead_time=0)
    thresh_ec = {dur: fdp.regrid_to(thresh[dur], ref.lat, ref.lon).load() for dur in CLEAN_DURATIONS}

    out_root = Path(args.out_dir); out_root.mkdir(parents=True, exist_ok=True)
    all_rows = []

    for ev in events:
        key = ev["key"]
        peak = ev["peak_date"]
        peak = dt.date.fromisoformat(peak) if isinstance(peak, str) else peak
        pre = ev.get("window_pre", d["window_pre"]); post = ev.get("window_post", d["window_post"])
        dates = window_dates(peak, pre, post)
        present = [t for t in dates if t in init_index]
        missing = [str(t.date()) for t in dates if t not in init_index]
        print(f"\n[{key}] {ev['country']} peak={peak} window={dates[0].date()}..{dates[-1].date()} "
              f"present={len(present)}/{len(dates)}" + (f"  MISSING={missing}" if missing else ""), flush=True)
        if not present:
            print(f"[{key}] no init dates in store — skipped", flush=True); continue

        # Duration accums: cumulative tp at lead == duration (nearest if absent).
        # Loop ONE init at a time and .load() each — the gap-filling interpolation
        # in open_ifs_ens_wb2 must hold the whole (capped) lead axis in memory, so
        # batching all 15 inits at once OOMs a small box; per-init keeps it ~200 MB.
        lead_idx = {}
        for dur in CLEAN_DURATIONS:
            h = fdp.DURATION_HOURS[dur]
            lead_idx[dur] = (int(np.where(lead_h == h)[0][0]) if (lead_h == h).any()
                             else int(np.argmin(np.abs(lead_h - h))))
        print(f"[{key}] loading {len(present)} inits × {len(CLEAN_DURATIONS)} durations × {ds.tp.sizes['member']} members ...", flush=True)
        per_init = []
        for t in present:
            subt = ds.tp.sel(init_date=t)  # (member, lead, lat, lon)
            accs = [(subt.isel(lead_time=lead_idx[dur]) * 1000.0).astype("float32")
                    for dur in CLEAN_DURATIONS]
            ct = xr.concat(accs, dim=pd.Index(CLEAN_DURATIONS, name="duration"))  # (dur, member, lat, lon)
            per_init.append(ct.load())
        cube = xr.concat(per_init, dim=pd.Index(present, name="init_date"))
        cube = cube.transpose("init_date", "duration", "member", "lat", "lon")

        # exceedance fraction (members >= RP) per (init, dur, pixel) and tail ratio
        th = xr.concat([thresh_ec[dur] for dur in CLEAN_DURATIONS],
                       dim=pd.Index(CLEAN_DURATIONS, name="duration"))
        exceed = (cube >= th).mean("member").astype("float32")          # (init, dur, lat, lon)
        p_heavy = exceed.max("duration").astype("float32")              # (init, lat, lon)
        safe_th = th.where(th > 0, np.nan)
        ens_max_ratio = (cube.max("member") / safe_th).max("duration").astype("float32")

        result = xr.Dataset({
            "tp_accum_mm": cube,                 # (init, dur, member, lat, lon)
            "exceedance_frac": exceed,           # (init, dur, lat, lon)
            "p_heavy": p_heavy,                  # (init, lat, lon)
            "ens_max_ratio": ens_max_ratio,      # (init, lat, lon)
        })
        result.attrs.update(event_key=key, country=ev["country"], emdat=ev["emdat"],
                             peak_date=str(peak), rp_years=args.rp_years,
                             source="gs://weatherbench2/datasets/ifs_ens/2016-2024-1440x721.zarr")

        ev_dir = out_root / key; ev_dir.mkdir(parents=True, exist_ok=True)
        enc = {v: {"zlib": True, "complevel": 4} for v in result.data_vars}
        nc = ev_dir / "forecast_accums.nc"
        result.to_netcdf(nc, encoding=enc)
        sz = nc.stat().st_size / 1e6

        # per-init-date domain summary
        rows = []
        for i, t in enumerate(present):
            ph = p_heavy.isel(init_date=i).values
            emr = ens_max_ratio.isel(init_date=i).values
            mm24 = cube.isel(init_date=i).sel(duration="24hr").max("member").values
            mm7d = cube.isel(init_date=i).sel(duration="7day").max("member").values
            rows.append(dict(
                event=key, country=ev["country"], init_date=str(t.date()),
                days_from_peak=(t.date() - peak).days,
                domain_max_p_heavy=round(float(np.nanmax(ph)), 4),
                pixels_p_heavy_gt0=int(np.nansum(ph > 0)),
                # p99 over pixels: robust to single arid pixels where the RP
                # threshold is ~0 and the raw max ratio explodes.
                domain_p99_ens_max_ratio=round(float(np.nanpercentile(emr, 99)), 3),
                pixels_any_member_exceed=int(np.nansum(emr >= 1.0)),
                ens_max_24h_mm=round(float(np.nanmax(mm24)), 1),
                ens_max_7day_mm=round(float(np.nanmax(mm7d)), 1),
            ))
        sdf = pd.DataFrame(rows)
        sdf.to_csv(ev_dir / "summary.csv", index=False)
        all_rows.extend(rows)
        peakrow = sdf.loc[sdf.pixels_any_member_exceed.idxmax()]
        print(f"[{key}] wrote {nc.name} ({sz:.1f} MB) + summary.csv | "
              f"peak-signal init={peakrow.init_date} (d{peakrow.days_from_peak:+d}) "
              f"p99_ens/RP={peakrow.domain_p99_ens_max_ratio} "
              f"px_exceed={peakrow.pixels_any_member_exceed}", flush=True)

    if all_rows:
        comb = pd.DataFrame(all_rows)
        comb.to_csv(out_root / "events_summary.csv", index=False)
        print(f"\n[collect] wrote {out_root/'events_summary.csv'}  rows={len(comb)}", flush=True)


if __name__ == "__main__":
    main()
