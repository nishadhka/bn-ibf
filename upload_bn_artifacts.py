#!/usr/bin/env -S uv run --with google-cloud-storage --with pandas --with python-dotenv python3
"""
upload_bn_artifacts.py
Routine GCS upload of the BN IBF artifacts consumed by the crma-api Cloud
Run service. Run this after every Julia BN inference run, once the four
generators have produced fresh artifacts:

    flood_ibf/   generate_bn_parquet.py + generate_bn_dag_json.py
    drought_ibf/ generate_drought_bn_parquet.py + generate_drought_bn_dag_json.py

Auth:
    Set GOOGLE_APPLICATION_CREDENTIALS to a key file for a service account
    with roles/storage.objectAdmin on the destination bucket. The
    coiled-data-e4drr@e4drr-crafd.iam.gserviceaccount.com SA already has
    the right grant.

Layout uploaded:

    parquet/
        flood_bn_ibf_daily.parquet
        flood_bn_ibf_boundary_daily.parquet
        drought_bn_ibf_monthly.parquet
        drought_bn_ibf_boundary_monthly.parquet
    bn-dag/
        bn-dag-YYYY-MM-DD.json              (one per flood inference day)
        drought-bn-dag-YYYY-MM.json         (one per drought init-month)

Usage:
    uv run python3 upload_bn_artifacts.py
    uv run python3 upload_bn_artifacts.py --dry-run
    uv run python3 upload_bn_artifacts.py --flood-only
    uv run python3 upload_bn_artifacts.py --drought-only
    uv run python3 upload_bn_artifacts.py --parquet-only
    uv run python3 upload_bn_artifacts.py --dag-only
    uv run python3 upload_bn_artifacts.py --bucket my-other-bucket
    uv run python3 upload_bn_artifacts.py --skip-unchanged

The script always uses a content-hash check (md5) when --skip-unchanged is
set, so re-running after a partial deploy only uploads what differs.
Without --skip-unchanged every artifact is overwritten — appropriate for
the post-BN-run case where every file's contents are expected to change.
"""
from __future__ import annotations

import argparse
import base64
import glob
import hashlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path

# google-cloud-storage is the only non-stdlib dep
from google.cloud import storage  # type: ignore
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent
load_dotenv(REPO_ROOT / ".env")

DEFAULT_BUCKET = "crma-mdx-store"

# ---------------------------------------------------------------------------
# Source files — paths relative to REPO_ROOT
# ---------------------------------------------------------------------------

# (local-path, gcs-blob-path, content-type)
PARQUET_FILES: list[tuple[Path, str, str]] = [
    (REPO_ROOT / "flood_ibf"   / "output" / "flood_bn_ibf_daily.parquet",
     "parquet/flood_bn_ibf_daily.parquet",          "application/octet-stream"),
    (REPO_ROOT / "flood_ibf"   / "output" / "flood_bn_ibf_boundary_daily.parquet",
     "parquet/flood_bn_ibf_boundary_daily.parquet", "application/octet-stream"),
    (REPO_ROOT / "drought_ibf" / "drought_bn_ibf_monthly.parquet",
     "parquet/drought_bn_ibf_monthly.parquet",          "application/octet-stream"),
    (REPO_ROOT / "drought_ibf" / "drought_bn_ibf_boundary_monthly.parquet",
     "parquet/drought_bn_ibf_boundary_monthly.parquet", "application/octet-stream"),
]

# Dirs that contain the per-day or per-init DAG JSONs. The script globs
# each dir and uploads every matching file to bn-dag/ with the same name.
DAG_DIRS: list[tuple[Path, str, str]] = [
    (REPO_ROOT / "flood_ibf" / "output" / "bn-dag",
     "bn-dag-*.json",          "bn-dag/"),
    (REPO_ROOT / "drought_ibf" / "output_v2_notail_cdi" / "bn-dag",
     "drought-bn-dag-*.json",  "bn-dag/"),
]


# ---------------------------------------------------------------------------
# Hash helpers
# ---------------------------------------------------------------------------

def md5_b64(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return base64.b64encode(h.digest()).decode("ascii")


# ---------------------------------------------------------------------------
# Upload core
# ---------------------------------------------------------------------------

@dataclass
class Plan:
    src: Path
    dst: str
    content_type: str
    action: str           # "upload" | "skip" | "missing-local"
    reason: str = ""


def plan_upload(bucket: storage.Bucket, src: Path, dst: str, ct: str,
                skip_unchanged: bool) -> Plan:
    if not src.is_file():
        return Plan(src, dst, ct, "missing-local", "local file not found")
    if not skip_unchanged:
        return Plan(src, dst, ct, "upload", "always overwrite")
    blob = bucket.blob(dst)
    if not blob.exists():
        return Plan(src, dst, ct, "upload", "new in GCS")
    blob.reload()
    local_md5 = md5_b64(src)
    if blob.md5_hash == local_md5:
        return Plan(src, dst, ct, "skip", "md5 match")
    return Plan(src, dst, ct, "upload",
                f"md5 differs: local={local_md5[:8]} gcs={blob.md5_hash[:8]}")


def execute(bucket: storage.Bucket, plans: list[Plan], dry_run: bool) -> tuple[int, int, int]:
    n_up = n_skip = n_miss = 0
    for p in plans:
        marker = {"upload": "↑", "skip": "=", "missing-local": "✗"}[p.action]
        line = f"  {marker} {p.dst:50}  ({p.action}: {p.reason})"
        print(line)
        if p.action == "missing-local":
            n_miss += 1
        elif p.action == "skip":
            n_skip += 1
        elif p.action == "upload":
            n_up += 1
            if not dry_run:
                blob = bucket.blob(p.dst)
                blob.upload_from_filename(str(p.src), content_type=p.content_type)
    return n_up, n_skip, n_miss


def collect_dag_pairs(dir_path: Path, glob_pat: str, dst_prefix: str
                       ) -> list[tuple[Path, str, str]]:
    pairs: list[tuple[Path, str, str]] = []
    for p in sorted(dir_path.glob(glob_pat)):
        pairs.append((p, dst_prefix + p.name, "application/json"))
    return pairs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", default=DEFAULT_BUCKET,
                    help=f"GCS bucket name (default: {DEFAULT_BUCKET}).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Plan only; do not actually upload.")
    ap.add_argument("--flood-only",   action="store_true",
                    help="Upload flood parquet + flood DAG JSONs only.")
    ap.add_argument("--drought-only", action="store_true",
                    help="Upload drought parquet + drought DAG JSONs only.")
    ap.add_argument("--parquet-only", action="store_true",
                    help="Upload parquet files only (skip DAG JSONs).")
    ap.add_argument("--dag-only",     action="store_true",
                    help="Upload DAG JSONs only (skip parquet).")
    ap.add_argument("--skip-unchanged", action="store_true",
                    help="Skip files whose md5 already matches the GCS object. "
                         "Without this flag, every file is overwritten.")
    ap.add_argument("--drought-dag-dir", default=None, type=Path,
                    help="Override the drought DAG-JSON source directory. "
                         "Default is drought_ibf/output_v2_notail_cdi/bn-dag. "
                         "Pass output_v2_notail_cdi_backfill/bn-dag to target "
                         "the 1981-2024 backfill output.")
    ap.add_argument("--flood-dag-dir", default=None, type=Path,
                    help="Override the flood DAG-JSON source directory. "
                         "Default is flood_ibf/output/bn-dag.")
    args = ap.parse_args()

    if args.flood_only and args.drought_only:
        ap.error("--flood-only and --drought-only are mutually exclusive")
    if args.parquet_only and args.dag_only:
        ap.error("--parquet-only and --dag-only are mutually exclusive")

    if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        print("WARNING: GOOGLE_APPLICATION_CREDENTIALS is not set. Falling "
              "back to gcloud ADC if available.", file=sys.stderr)

    print(f"bucket: gs://{args.bucket}")
    print(f"mode:   {'DRY-RUN' if args.dry_run else 'WRITE'}"
          f"{' + skip-unchanged' if args.skip_unchanged else ''}")

    client = storage.Client()
    bucket = client.bucket(args.bucket)

    # ── Build the work list ───────────────────────────────────────────────

    parquet_targets: list[tuple[Path, str, str]] = []
    if not args.dag_only:
        for src, dst, ct in PARQUET_FILES:
            if args.flood_only   and "flood_bn_ibf"   not in dst: continue
            if args.drought_only and "drought_bn_ibf" not in dst: continue
            parquet_targets.append((src, dst, ct))

    dag_targets: list[tuple[Path, str, str]] = []
    if not args.parquet_only:
        for src_dir, glob_pat, dst_prefix in DAG_DIRS:
            is_drought = glob_pat.startswith("drought-")
            if args.flood_only   and is_drought:     continue
            if args.drought_only and not is_drought: continue
            override = args.drought_dag_dir if is_drought else args.flood_dag_dir
            effective_dir = (override.resolve() if override else src_dir)
            dag_targets.extend(collect_dag_pairs(effective_dir, glob_pat, dst_prefix))

    # ── Plan ──────────────────────────────────────────────────────────────

    print("\n[parquet]")
    parquet_plans = [plan_upload(bucket, s, d, c, args.skip_unchanged)
                      for s, d, c in parquet_targets]

    print(f"\n[dag-json]  ({len(dag_targets)} files)")
    dag_plans = [plan_upload(bucket, s, d, c, args.skip_unchanged)
                  for s, d, c in dag_targets]

    # ── Execute ───────────────────────────────────────────────────────────

    print("\n[parquet] execute")
    p_up, p_skip, p_miss = execute(bucket, parquet_plans, args.dry_run)
    print("\n[dag-json] execute")
    d_up, d_skip, d_miss = execute(bucket, dag_plans, args.dry_run)

    print("\n──────────────────────────────────────────────────")
    print(f"summary: parquet  uploaded={p_up} skipped={p_skip} missing={p_miss}")
    print(f"         dag-json uploaded={d_up} skipped={d_skip} missing={d_miss}")
    if args.dry_run:
        print("(dry-run — no objects were actually written)")
    return 1 if (p_miss + d_miss) > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
