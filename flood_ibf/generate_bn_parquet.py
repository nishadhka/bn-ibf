#!/usr/bin/env python3
"""
generate_bn_parquet.py
Reads all flood_bn_v1_YYYY-MM-DD.csv files from output/dbn/ and writes two parquet files:
  - flood_bn_ibf_daily.parquet          (1 row per day)
  - flood_bn_ibf_boundary_daily.parquet (1 row per boundary x day)

Usage:
  uv run python3 generate_bn_parquet.py [--input-dir output/dbn] [--out-dir .]
"""
import argparse
import glob
import os
import re

import pandas as pd

RISK_LEVEL_INT = {
    "Minimal": 1,
    "Low": 2,
    "Moderate": 3,
    "High": 4,
    "Extreme": 5,
}


def load_all(input_dir: str) -> pd.DataFrame:
    pattern = os.path.join(input_dir, "flood_bn_v1_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern}")
    frames = []
    for f in files:
        m = re.search(r"(\d{4}-\d{2}-\d{2})", os.path.basename(f))
        if not m:
            continue
        df = pd.read_csv(f)
        df["file_date"] = m.group(1)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def make_daily(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["risk_level_int"] = df["risk_level"].map(RISK_LEVEL_INT).fillna(1).astype(int)
    df["target_date"] = pd.to_datetime(df["target_date"])

    rows = []
    for date, grp in df.groupby("target_date"):
        counts = grp["crma_state"].value_counts()
        max_risk = grp["risk_level_int"].max()
        rows.append(
            {
                "year": date.year,
                "month": date.month,
                "day": date.day,
                "event_key": f"fl-rd-{date.strftime('%Y-%m-%d')}",
                "level": int(max_risk),
                "n_monitor": int(counts.get("Monitor", 0)),
                "n_evaluate": int(counts.get("Evaluate", 0)),
                "n_assess": int(counts.get("Assess", 0)),
                "n_actionable_risk": int(counts.get("Actionable_Risk", 0)),
            }
        )
    return pd.DataFrame(rows).sort_values(["year", "month", "day"]).reset_index(drop=True)


def make_boundary_daily(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["risk_level_int"] = df["risk_level"].map(RISK_LEVEL_INT).fillna(1).astype(int)
    df["target_date"] = pd.to_datetime(df["target_date"])

    keep = [
        "target_date",
        "boundary_id",
        "boundary_name",
        "country",
        "risk_level",
        "risk_level_int",
        "crma_state",
        "traffic_light",
        "p_high_extreme",
        "risk_minimal",
        "risk_low",
        "risk_moderate",
        "risk_high",
        "risk_extreme",
    ]
    out = df[keep].copy()
    out["event_key"] = out["target_date"].dt.strftime("fl-rd-%Y-%m-%d")
    out["shapeID"] = out["boundary_id"]
    out["shapeName"] = out["boundary_name"]
    out["shapeGroup"] = out["country"]
    return out.sort_values(["target_date", "boundary_id"]).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="output/dbn")
    parser.add_argument("--out-dir", default=".")
    args = parser.parse_args()

    print(f"Reading from: {args.input_dir}")
    df = load_all(args.input_dir)
    print(f"  Loaded {len(df)} rows across {df['target_date'].nunique()} dates")

    daily = make_daily(df)
    boundary_daily = make_boundary_daily(df)

    daily_path = os.path.join(args.out_dir, "flood_bn_ibf_daily.parquet")
    boundary_path = os.path.join(args.out_dir, "flood_bn_ibf_boundary_daily.parquet")

    daily.to_parquet(daily_path, index=False)
    boundary_daily.to_parquet(boundary_path, index=False)

    print(f"  Wrote {daily_path} ({len(daily)} rows)")
    print(f"  Wrote {boundary_path} ({len(boundary_daily)} rows)")

    print("\nDaily preview:")
    print(daily.to_string(index=False))


if __name__ == "__main__":
    main()
