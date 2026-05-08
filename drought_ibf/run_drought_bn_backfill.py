#!/usr/bin/env -S uv run --with pandas python3
"""
run_drought_bn_backfill.py

End-to-end driver for the drought-BN backfill described in
`evidence_nodes.md` §8 ("Smallest set of changes to commission the
1981-2024 backfill").

Steps, each independently re-runnable via `--steps`:

  prep    : run drought_data_prep.py for every (init, season) pair in the
            window, writing soft-evidence CSVs into <prep-dir>/.
  bn      : write a manifest of (input_csv, output_csv) pairs, then call
            drought_bn_ibf_v1_warm.jl ONCE in a warm Julia session that
            iterates the entire manifest. Outputs the BN-only posterior
            CSVs into <bn-dir>/.
  cdi     : per init-month, run cdi_data_prep.py + cdi_evidence_update.py
            with the year-gated `cdi_source` / `fapar_source` flags from
            evidence_nodes.md §8. Outputs the post-CDI CSVs into
            <cdi-dir>/.

Why warm Julia: each cold `julia drought_bn_ibf_v1.jl` invocation pays
~95 s of RxInfer JIT compilation. For a 528-init backfill that's ~14
hours of pure overhead. The warm-loop driver pays it once, then runs
every subsequent init in <1 s of actual inference.

Usage:
    # Smoke-test on 4 inits
    uv run python3 run_drought_bn_backfill.py --start 2024-01 --end 2024-04

    # Full 1981-2024 backfill
    uv run python3 run_drought_bn_backfill.py --start 1981-01 --end 2024-12

    # Stage subset (e.g. re-run only CDI after a fix)
    uv run python3 run_drought_bn_backfill.py --steps cdi --start 2010-01 --end 2024-12

    # Dry-run (print plan, do not invoke)
    uv run python3 run_drought_bn_backfill.py --start 1981-01 --end 2024-12 --dry-run
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent

# ── Init-month -> target-season mapping (from drought_data_prep.py
# SEASON_INIT_LEAD). Each calendar month maps to exactly one operational
# target season under the lead-3/4/5 convention.
INIT_TO_SEASON: dict[int, str] = {
    12: "MAM", 1: "MAM", 2: "MAM",
    3:  "JJA", 4: "JJA", 5: "JJA",
    6:  "OND", 7: "OND", 8: "OND",
    9:  "DJF", 10: "DJF", 11: "DJF",
}

# ── CDI year-gating from evidence_nodes.md §8 ──────────────────────────
# Returns (cdi_source, fapar_source) — or None to skip CDI entirely.
def cdi_command_for(target: pd.Timestamp) -> tuple[str, str] | None:
    if target < pd.Timestamp("1995-01-01"):
        return None  # pre-CHIRPS-SMA: no CDI possible
    if target < pd.Timestamp("2001-01-01"):
        return ("recompute", "none")        # SPI+SMA only (Watch+Warning)
    if target < pd.Timestamp("2010-01-01"):
        return ("recompute", "auto")        # auto resolves to MODIS fAPAR
    if target < pd.Timestamp("2012-01-01"):
        return ("both", "auto")             # MODIS fAPAR + EADW
    return ("both", "auto")                 # GDO operational + EADW


# ── Pair generator ────────────────────────────────────────────────────
def init_pairs(start: str, end: str) -> list[tuple[pd.Timestamp, str]]:
    """Yield (init_date, target_season) for every month in [start, end]."""
    months = pd.date_range(pd.Timestamp(start).replace(day=1),
                           pd.Timestamp(end).replace(day=1),
                           freq="MS")
    return [(d, INIT_TO_SEASON[d.month]) for d in months]


# ── Subprocess helpers ────────────────────────────────────────────────
def _run(cmd: list[str], dry_run: bool, label: str = ""):
    pretty = " ".join(shlex.quote(c) for c in cmd)
    print(f"[{label or 'run'}]  $ {pretty}", flush=True)
    if dry_run:
        return 0
    res = subprocess.run(cmd)
    if res.returncode != 0:
        raise SystemExit(f"[{label}] command failed (exit {res.returncode})")
    return res.returncode


# ── Step 1: drought_data_prep.py per (init, season) ───────────────────
def step_prep(pairs, prep_dir: Path, adm1: Path, ensemble_size: int,
              rp_years: int, skip_existing: bool, dry_run: bool):
    prep_dir.mkdir(parents=True, exist_ok=True)
    n = len(pairs)
    t0 = time.time()
    for i, (d, season) in enumerate(pairs, 1):
        out = prep_dir / f"drought_inputs_{d:%Y-%m}_{season}.csv"
        if skip_existing and out.is_file() and out.stat().st_size > 0:
            if i == 1 or i % 25 == 0 or i == n:
                print(f"[prep {i}/{n}] skip existing {out.name}")
            continue
        cmd = [
            "uv", "run",
            "--with", "icechunk", "--with", "xarray", "--with", "zarr>=3",
            "--with", "numpy", "--with", "pandas", "--with", "geopandas",
            "--with", "regionmask", "--with", "scipy", "--with", "pyogrio",
            "--with", "fsspec", "--with", "s3fs", "--with", "aiobotocore",
            "python3", str(REPO_ROOT / "drought_data_prep.py"),
            "--init-month", f"{d:%Y-%m}",
            "--target-season", season,
            "--ensemble-size", str(ensemble_size),
            "--rp-years", str(rp_years),
            "--soft-evidence",
            "--adm1", str(adm1),
            "--out", str(out),
        ]
        _run(cmd, dry_run, label=f"prep {i}/{n}")
    print(f"[prep] done in {time.time()-t0:.1f}s")


# ── Step 2: warm Julia BN inference over a manifest ───────────────────
def step_bn(pairs, prep_dir: Path, bn_dir: Path,
            include_tail_risk: bool, no_agreement: bool,
            cost_loss_ratio: float, dry_run: bool):
    bn_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = bn_dir / "_manifest.csv"
    rows = []
    for d, season in pairs:
        in_csv  = prep_dir / f"drought_inputs_{d:%Y-%m}_{season}.csv"
        out_csv = bn_dir / f"drought_bn_v2_notail_{d:%Y-%m}_{season}.csv"
        rows.append({"input_csv": str(in_csv), "output_csv": str(out_csv)})
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    print(f"[bn] wrote manifest with {len(rows)} entries: {manifest_path}")

    cmd = [
        "julia", f"--project={REPO_ROOT}",
        str(REPO_ROOT / "drought_bn_ibf_v1_warm.jl"),
        "--manifest", str(manifest_path),
        "--cost-loss-ratio", str(cost_loss_ratio),
    ]
    if no_agreement:
        cmd.append("--no-agreement")
    if include_tail_risk:
        cmd.append("--tail-risk")
    _run(cmd, dry_run, label="bn warm-loop")


# ── Step 3: CDI prep + evidence update per init, year-gated ───────────
def step_cdi(pairs, bn_dir: Path, cdi_dir: Path, adm1: Path,
             cdi_inputs_dir: Path, dry_run: bool, skip_existing: bool):
    cdi_dir.mkdir(parents=True, exist_ok=True)
    cdi_inputs_dir.mkdir(parents=True, exist_ok=True)
    n = len(pairs)
    n_skipped_pre1995 = 0
    for i, (d, season) in enumerate(pairs, 1):
        out_csv = cdi_dir / f"drought_bn_v2_notail_cdi_{d:%Y-%m}_{season}.csv"
        if skip_existing and out_csv.is_file() and out_csv.stat().st_size > 0:
            if i == 1 or i % 25 == 0 or i == n:
                print(f"[cdi {i}/{n}] skip existing {out_csv.name}")
            continue

        gating = cdi_command_for(d)
        if gating is None:
            # Pre-1995: no CDI possible; copy the pre-CDI BN result through
            # so downstream consumers see a single canonical output dir.
            n_skipped_pre1995 += 1
            src = bn_dir / f"drought_bn_v2_notail_{d:%Y-%m}_{season}.csv"
            if not dry_run and src.is_file():
                # Cheap pass-through: read + re-write so the schema matches
                # what cdi_evidence_update.py would have produced (plus
                # NaN'd CDI columns). The simplest version: just symlink
                # for now and let downstream tools fall back when the
                # CDI columns are absent.
                if not out_csv.is_symlink() and not out_csv.exists():
                    os.symlink(src.resolve(), out_csv)
            print(f"[cdi {i}/{n}] pre-1995 ({d:%Y-%m}) — no CDI; passthrough")
            continue

        cdi_source, fapar_source = gating
        cdi_inputs_csv = cdi_inputs_dir / f"cdi_inputs_{d:%Y-%m}.csv"

        prep_cmd = [
            "uv", "run",
            "--with", "icechunk", "--with", "xarray", "--with", "zarr>=3",
            "--with", "numpy", "--with", "pandas", "--with", "geopandas",
            "--with", "regionmask", "--with", "scipy", "--with", "pyogrio",
            "--with", "fsspec", "--with", "s3fs", "--with", "aiobotocore",
            "python3", str(REPO_ROOT / "cdi_data_prep.py"),
            "--date", f"{d:%Y-%m}",
            "--adm1", str(adm1),
            "--out", str(cdi_inputs_csv),
            "--cdi-source", cdi_source,
            "--fapar-source", fapar_source,
        ]
        _run(prep_cmd, dry_run, label=f"cdi-prep {i}/{n}  {cdi_source}/{fapar_source}")

        bn_csv = bn_dir / f"drought_bn_v2_notail_{d:%Y-%m}_{season}.csv"
        update_cmd = [
            "uv", "run", "--with", "pandas", "--with", "numpy",
            "python3", str(REPO_ROOT / "cdi_evidence_update.py"),
            "--bn-csv", str(bn_csv),
            "--cdi-csv", str(cdi_inputs_csv),
            "--out", str(out_csv),
        ]
        _run(update_cmd, dry_run, label=f"cdi-update {i}/{n}")
    if n_skipped_pre1995:
        print(f"[cdi] passed {n_skipped_pre1995} pre-1995 inits through "
              "without CDI (no source available)")


# ── Top-level wiring ─────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description=__doc__)
    ap.add_argument("--start", default="1981-01",
                    help="First init month (YYYY-MM). Default 1981-01.")
    ap.add_argument("--end",   default="2024-12",
                    help="Last init month (YYYY-MM). Default 2024-12.")
    ap.add_argument("--steps", default="prep,bn,cdi",
                    help="Comma-separated subset of stages to run "
                         "(prep, bn, cdi). Default 'prep,bn,cdi' (all).")
    ap.add_argument("--prep-dir", default="bn_inputs_v2_backfill",
                    help="Output dir for soft-evidence CSVs (step prep).")
    ap.add_argument("--bn-dir",   default="output_v2_notail_backfill",
                    help="Output dir for BN-only posterior CSVs (step bn).")
    ap.add_argument("--cdi-dir",  default="output_v2_notail_cdi_backfill",
                    help="Output dir for post-CDI posterior CSVs (step cdi).")
    ap.add_argument("--cdi-inputs-dir", default="cdi_inputs_backfill",
                    help="Working dir for cdi_inputs_<init>.csv files (step cdi).")
    ap.add_argument("--adm1",      default="icpac_adm1v3.geojson",
                    help="Path to admin-1 GeoJSON (default icpac_adm1v3.geojson).")
    ap.add_argument("--ensemble-size", type=int, default=25,
                    help="SEAS5 ensemble members for prep (default 25 = full "
                         "1981-now hindcast).")
    ap.add_argument("--rp-years", type=int, default=5,
                    help="ERA5 SPI return-period years for the deficit threshold "
                         "(default 5).")
    ap.add_argument("--cost-loss-ratio", type=float, default=0.20,
                    help="CRMA decision γ (default 0.20).")
    ap.add_argument("--include-tail-risk", action="store_true",
                    help="Pass --tail-risk to the BN (5-parent v1 mode). "
                         "Default off — production v2 is 4-parent.")
    ap.add_argument("--no-agreement", action="store_true", default=True,
                    help="Pass --no-agreement to the BN (default on; "
                         "agreement node is deprecated in v2).")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip a step's output file if it already exists.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Limit to the first N (init, season) pairs (smoke-test).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the plan; do not invoke any subprocess.")
    args = ap.parse_args()

    pairs = init_pairs(args.start, args.end)
    if args.limit:
        pairs = pairs[: args.limit]
    print(f"backfill window: {args.start} .. {args.end}  ({len(pairs)} init-months)")
    if pairs:
        print(f"  first: {pairs[0][0]:%Y-%m}  {pairs[0][1]}")
        print(f"  last:  {pairs[-1][0]:%Y-%m}  {pairs[-1][1]}")

    steps = {s.strip() for s in args.steps.split(",") if s.strip()}
    unknown = steps - {"prep", "bn", "cdi"}
    if unknown:
        ap.error(f"unknown step(s): {sorted(unknown)}; allowed: prep, bn, cdi")

    prep_dir = REPO_ROOT / args.prep_dir
    bn_dir   = REPO_ROOT / args.bn_dir
    cdi_dir  = REPO_ROOT / args.cdi_dir
    cdi_inputs_dir = REPO_ROOT / args.cdi_inputs_dir
    adm1     = (REPO_ROOT / args.adm1).resolve()

    if "prep" in steps:
        step_prep(pairs, prep_dir, adm1, args.ensemble_size, args.rp_years,
                  args.skip_existing, args.dry_run)
    if "bn" in steps:
        step_bn(pairs, prep_dir, bn_dir,
                include_tail_risk=args.include_tail_risk,
                no_agreement=args.no_agreement,
                cost_loss_ratio=args.cost_loss_ratio,
                dry_run=args.dry_run)
    if "cdi" in steps:
        step_cdi(pairs, bn_dir, cdi_dir, adm1, cdi_inputs_dir,
                 args.dry_run, args.skip_existing)

    print("\ndone. Next:")
    print("  generate_drought_bn_parquet.py "
          f"--input-dir {args.cdi_dir} "
          f"--prefix drought_bn_v2_notail_cdi_ "
          f"--out-dir .")
    print("  generate_drought_bn_dag_json.py "
          f"--bn-dir {args.cdi_dir} "
          f"--bn-prefix drought_bn_v2_notail_cdi_ "
          f"--out-dir {args.cdi_dir}/bn-dag")
    print("  upload_bn_artifacts.py --skip-unchanged")


if __name__ == "__main__":
    main()
