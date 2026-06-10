#!/usr/bin/env -S uv run --with pandas --with pyarrow python3
"""
consolidate_flood_events.py
Consolidate the 11 historical flood-event BN artifacts into the canonical
flat layout that upload_bn_artifacts.py / the crma-api consume, MERGING with
the existing operational (Mar-2026) artifacts rather than replacing them.

1. bn-dag JSONs — copy every output/events/<key>/bn-dag/bn-dag-*.json into
   output/bn-dag/ (the uploader's default flood dir). Event dates (2019-2024)
   are distinct from the operational 2026-03 set; intra-event same-date files
   (e.g. bdi/tza/ken overlap in 2024-04) are byte-identical, so a later copy
   just overwrites with the same content.
2. parquet — rebuild the daily + boundary_daily tables from every event's DBN
   CSVs, then UNION with the existing operational parquet (dedup by date /
   date+boundary), and overwrite flood_ibf/output/flood_bn_ibf_*.parquet.

After this, run:
   GOOGLE_APPLICATION_CREDENTIALS=../coiled-data-e4drr_202505.json \
     uv run python3 ../upload_bn_artifacts.py --flood-only [--dry-run]
"""
from __future__ import annotations
import glob, os, shutil, sys
from pathlib import Path
import pandas as pd

import importlib.util
_g = Path(__file__).with_name("generate_bn_parquet.py")
_spec = importlib.util.spec_from_file_location("gbp", str(_g))
gbp = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(gbp)

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
EVENTS = OUT / "events"
BN_DAG = OUT / "bn-dag"
DAILY = OUT / "flood_bn_ibf_daily.parquet"
BOUNDARY = OUT / "flood_bn_ibf_boundary_daily.parquet"


def stage_dag_jsons() -> int:
    BN_DAG.mkdir(parents=True, exist_ok=True)
    n = 0
    for src in sorted(EVENTS.glob("*/bn-dag/bn-dag-*.json")):
        dst = BN_DAG / src.name
        shutil.copy2(src, dst)
        n += 1
    return n


def event_parquets() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build daily + boundary_daily from every event's DBN CSVs (deduped)."""
    csvs = sorted(glob.glob(str(EVENTS / "*" / "dbn" / "flood_bn_v1_*.csv")))
    if not csvs:
        raise SystemExit("no event DBN CSVs found under output/events/*/dbn/")
    frames = []
    import re
    for f in csvs:
        m = re.search(r"(\d{4}-\d{2}-\d{2})", os.path.basename(f))
        if not m:
            continue
        df = pd.read_csv(f); df["file_date"] = m.group(1)
        frames.append(df)
    allrows = pd.concat(frames, ignore_index=True)
    # dedup identical (date, boundary) rows from overlapping-window events
    allrows = allrows.drop_duplicates(subset=["target_date", "boundary_id"], keep="first")
    return gbp.make_daily(allrows), gbp.make_boundary_daily(allrows)


def merge_parquet(ev: pd.DataFrame, existing_path: Path, keys: list[str]) -> pd.DataFrame:
    if existing_path.is_file():
        cur = pd.read_parquet(existing_path)
        # align dtypes on the merge keys to avoid concat surprises
        merged = pd.concat([cur, ev], ignore_index=True)
        merged = merged.drop_duplicates(subset=keys, keep="last")
    else:
        merged = ev
    return merged


def main() -> int:
    n_json = stage_dag_jsons()
    print(f"[consolidate] staged {n_json} event bn-dag JSONs into {BN_DAG}")
    total_json = len(list(BN_DAG.glob('bn-dag-*.json')))
    print(f"[consolidate] {BN_DAG} now holds {total_json} JSONs (operational + events)")

    ev_daily, ev_boundary = event_parquets()
    print(f"[consolidate] event parquet: daily={len(ev_daily)} rows "
          f"({ev_daily.shape}), boundary={len(ev_boundary)} rows")

    daily = merge_parquet(ev_daily, DAILY, keys=["year", "month", "day"])
    daily = daily.sort_values(["year", "month", "day"]).reset_index(drop=True)
    boundary = merge_parquet(ev_boundary, BOUNDARY, keys=["target_date", "boundary_id"])
    boundary = boundary.sort_values(["target_date", "boundary_id"]).reset_index(drop=True)

    daily.to_parquet(DAILY, index=False)
    boundary.to_parquet(BOUNDARY, index=False)
    print(f"[consolidate] wrote {DAILY.name} ({len(daily)} rows, "
          f"{daily[['year','month']].drop_duplicates().shape[0]} months) and "
          f"{BOUNDARY.name} ({len(boundary)} rows, "
          f"{boundary['target_date'].nunique()} dates)")
    print("[consolidate] date span:",
          str(pd.to_datetime(boundary['target_date']).min().date()), "→",
          str(pd.to_datetime(boundary['target_date']).max().date()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
