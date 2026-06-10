# Flood BN-IBF — 11-event run record & disk footprint

What was run to produce the historical 11-event flood hindcast (June 2026),
the artifacts it created, and **why the run occupies ~1.5 GB on disk**.

Companion to `flood_events_run_notes.md` (method, events, results) and
`flood_bn_ibf_run_notes_2026-03.md` (the operational Mar-2026 run this extends).

---

## 1. What the run did

Replayed the validated 15-day flood BN-IBF routine over the **11 GHACOF73
historical flood events** (2019–2024), using the **WeatherBench2 archived ECMWF
IFS-ENS** 50-member forecast (`gs://weatherbench2/datasets/ifs_ens/2016-2024-1440x721.zarr`)
in place of the recent-only operational store, plus IMERG antecedent and CMORPH
2-yr return-period thresholds. Each event = a 15-day window `[peak−5, peak+9]`.

```
 per event (×11):
   flood_data_prep.py  (range mode: stores opened once, init loaded once)
        │  reads  IMERG HH icechunk + WB2 IFS-ENS (50 mbr) + CMORPH RP
        ▼  writes 15 × flood_inputs_<D>_soft.csv   (227 boundaries × ~40 cols)
   run_flood_dbn_window.jl  (RxInfer DBN, α=0.6, lookback=7, γ=0.2)
        ▼  writes 15 × dbn/flood_bn_v1_<D>.csv + combined
   generate_bn_parquet.py + generate_bn_dag_json.py
        ▼  writes daily+boundary parquet + 15 × bn-dag-<D>.json
 then:
   collect_wb2_ifs_events.py   → wb2_ifs_ens/<key>/forecast_accums.nc (+summary)
   consolidate_flood_events.py → stage JSONs flat + merge parquets w/ operational
   upload_bn_artifacts.py      → gs://crma-mdx-store  (154 JSONs + 2 parquet)
```

**Scale:** 11 events × 15 days = **165 boundary-days**, each over 227 admin-1
regions and **50 ensemble members** ⇒ ~1.87 M boundary-member BN evaluations.

**Result:** 9/11 events reached Actionable_Risk in the affected country (see
`flood_events_run_notes.md` §6b). Delivered to the crma-api on GCS.

---

## 2. Disk footprint — where the ~1.5 GB is

| Path | Size | What |
|------|-----:|------|
| `wb2_ifs_ens/*/forecast_accums.nc` | **1.4 GB** | 11 collected forecast cubes (50-member duration accums) |
| `output/bn-dag/*.json` | 19 MB | 154 per-day BN-DAG panel JSONs (139 event + 15 operational) |
| `output/events/*/` | 48 MB | per-event inputs + DBN CSVs + per-event JSONs/parquet |
| `output/*.parquet` | <1 MB | merged calendar + choropleth parquet |
| **total** | **~1.5 GB** | |

**95 % of the footprint is the 11 forecast cubes.** Everything the BN produces
(evidence CSVs, DBN posteriors, DAG JSONs, parquets) is only ~68 MB combined.

### Why each cube is ~130 MB

`collect_wb2_ifs_events.py` saves, per event, the **full 50-member ensemble**
forecast as `tp_accum_mm(init_date, duration, member, lat, lon)`:

```
 15 inits × 5 durations × 50 members × 145 lat × 125 lon × 4 bytes (float32)
   = 271,875,000 values·bytes ≈ 259 MiB uncompressed
   → ~130 MiB on disk (zlib level-4, ~2× — precip is sparse/mostly small)
 × 11 events ≈ 1.4 GB
```

The size is driven almost entirely by the **`member` dimension (50×)**. The cube
keeps every ensemble member so per-member storyline analysis (worst/median/best
plausible world) is reproducible offline without re-fetching from GCS. The
forecast field over East Africa is otherwise small (145×125 ≈ 18 k pixels at
0.25°); it is the 50-member × 15-init × 5-duration product that inflates it.

### Why this is expected, not a leak

- A single deterministic field for one event/day over this box is ~70 KB. The
  ensemble (50 members) × 15 daily inits × 5 accumulation windows is **3 750×**
  that — the cost of keeping a *probabilistic, time-resolved* forecast archive.
- These cubes are the **forecast-collection deliverable** (the `collect` step),
  separate from the BN run itself. The BN only needs them transiently; it
  actually re-reads WB2 directly during prep, so the cubes are an optional cache
  / audit artifact.

---

## 3. Reducing the footprint (if needed)

The 1.4 GB of cubes is **gitignored** (`flood_ibf/wb2_ifs_ens/**/forecast_accums.nc`)
and fully regenerable via `./collect_wb2_ifs_events.py`. Options:

| Action | New size | Trade-off |
|--------|---------:|-----------|
| Delete cubes after upload | ~68 MB | lose offline forecast cache; regenerate on demand |
| Keep ensemble summaries only (drop `member`) | ~30 MB total | lose per-member storylines; keep mean/exceedance/tail |
| Keep members but fewer durations (24h+7d only) | ~560 MB | coarser duration resolution |
| Store as cloud zarr instead of local nc | 0 local | needs network to read |

The committed git artifacts are unaffected by this — only the BN outputs
(`output/`, ~68 MB, of which 19 MB of JSON + the parquets are committed) and the
delivered GCS objects matter for the API. To reclaim the 1.4 GB safely:

```bash
rm -rf flood_ibf/wb2_ifs_ens          # cubes only; regenerate with collect_wb2_ifs_events.py
```

---

## 4. Wall-clock & compute notes

- Per-event ≈ 12–15 min after the range-mode optimization (`5a7a929`): prep
  ~8 min (15 days × ~30 s) + Julia DBN ~5 min + generators ~1 min.
- The pre-optimization per-day path was ~3–6 min **per day** (each day a fresh
  subprocess re-opening every store and re-running the WB2 lead interpolation 7×)
  ⇒ the 10-event batch would have taken ~17 h; it ran in ~2.5 h instead.
- One-time setup: Julia 1.12.6 + RxInfer 4.7.3 precompile (~9 min).
- Memory ceiling 7 GB drove two design choices: per-init WB2 loading in the
  collector and sequential (not parallel) event execution.
