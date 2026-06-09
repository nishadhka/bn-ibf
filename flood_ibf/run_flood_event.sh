#!/usr/bin/env bash
# run_flood_event.sh — replay the flood BN-IBF 15-day routine for one historical
# event from flood_events.yaml, using the WeatherBench2 archived ECMWF IFS-ENS
# forecast (--forecast-source ifs_ens_wb2) + IMERG antecedent.
#
#   ./run_flood_event.sh ken_2024_04
#
# Window = [peak_date - window_pre, peak_date + window_post] (default 5/9 = 15
# days, matching the operational Mar-2026 layout). Writes everything under
# output/events/<key>/ :
#   bn_inputs/flood_inputs_<D>_soft.csv   per-day soft-evidence (15)
#   dbn/flood_bn_v1_<D>.csv               per-day DBN posteriors (15)
#   flood_bn_v1_dbn_window.csv            combined DBN output
#   flood_bn_ibf_daily.parquet            calendar API parquet
#   flood_bn_ibf_boundary_daily.parquet   choropleth API parquet
#   bn-dag/bn-dag-<D>.json                BN-DAG panel JSON (15)
set -euo pipefail

KEY="${1:?usage: run_flood_event.sh <event_key>   (keys in flood_events.yaml)}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export PATH="$HOME/.juliaup/bin:$PATH"

ADM1="${ADM1:-icpac_adm1v3.geojson}"
if [[ ! -f "$ADM1" ]]; then
  echo "ERROR: admin-1 boundaries '$ADM1' not found in $SCRIPT_DIR." >&2
  echo "       It is a runtime prerequisite (not in git). Place icpac_adm1v3.geojson here" >&2
  echo "       or set ADM1=/path/to/boundaries.geojson." >&2
  exit 1
fi

RP_YEARS="${RP_YEARS:-2}"
COST_LOSS_RATIO="${COST_LOSS_RATIO:-0.2}"

UV_PKGS=(
  --with icechunk --with xarray --with "zarr>=3"
  --with numpy --with pandas --with geopandas --with regionmask
  --with netcdf4 --with pyarrow --with scipy --with fsspec --with s3fs --with gcsfs --with bottleneck
)

# ---- resolve window from flood_events.yaml ----
read -r START END PEAK COUNTRY < <(uv run --with pyyaml python3 - "$KEY" <<'PY'
import sys, yaml, datetime as dt
key = sys.argv[1]
cfg = yaml.safe_load(open("flood_events.yaml"))
d = cfg["defaults"]
ev = next((e for e in cfg["events"] if e["key"] == key), None)
if ev is None:
    sys.exit(f"unknown event key: {key} (see flood_events.yaml)")
peak = ev["peak_date"]
peak = dt.date.fromisoformat(peak) if isinstance(peak, str) else peak
pre  = ev.get("window_pre",  d["window_pre"])
post = ev.get("window_post", d["window_post"])
start = peak - dt.timedelta(days=pre)
end   = peak + dt.timedelta(days=post)
print(start, end, peak, ev["country"].replace(" ", "_"))
PY
)

EVENT_DIR="output/events/$KEY"
IN_DIR="$EVENT_DIR/bn_inputs"
mkdir -p "$IN_DIR" "$EVENT_DIR/bn-dag"

echo "================================================================"
echo "  Flood BN-IBF event hindcast: $KEY ($COUNTRY)"
echo "  peak=$PEAK  window=$START .. $END  RP=${RP_YEARS}yr  forecast=ifs_ens_wb2"
echo "================================================================"

# ---- Step 1: soft-evidence prep for the whole 15-day window in one process ----
# Range mode opens IMERG/IFS-ENS/CMORPH and builds masks ONCE, pre-loads the
# IMERG antecedent union, and loads each forecast init once — ~30 s/day vs
# ~3-6 min/day when each day was a separate invocation.
echo "[prep] $KEY $START..$END -> $IN_DIR"
uv run "${UV_PKGS[@]}" python flood_data_prep.py \
    --date "$START" \
    --end-date "$END" \
    --rp-years "$RP_YEARS" \
    --forecast-source ifs_ens_wb2 \
    --soft-evidence \
    --adm1 "$ADM1" \
    --out-dir "$IN_DIR"

# ---- Step 2: DBN sequence over the 15 days ----
echo "[dbn] $KEY — run_flood_dbn_window.jl"
julia --project=. run_flood_dbn_window.jl \
    --input-dir "$IN_DIR" \
    --out-dir   "$EVENT_DIR" \
    --expect    15

# ---- Step 3: web artifacts (parquet + BN-DAG JSON) ----
echo "[web] parquet + bn-dag JSON -> $EVENT_DIR"
uv run "${UV_PKGS[@]}" python3 generate_bn_parquet.py \
    --input-dir "$EVENT_DIR/dbn" --out-dir "$EVENT_DIR"
uv run "${UV_PKGS[@]}" python3 generate_bn_dag_json.py \
    --input-dir "$IN_DIR" --dbn-dir "$EVENT_DIR/dbn" --out-dir "$EVENT_DIR/bn-dag"

echo "================================================================"
echo "  Done: $EVENT_DIR"
echo "================================================================"
