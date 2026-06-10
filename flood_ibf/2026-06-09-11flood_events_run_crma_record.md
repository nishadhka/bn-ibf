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

**Result (worst CRMA state reached in the affected country, within the window):**

| event | country | peak | worst CRMA | day vs peak | #Actionable_Risk | #Assess+ |
|-------|---------|------|------------|:-----------:|:----------------:|:--------:|
| bdi_2024_04 | Burundi | 2024-04-15 | **Actionable_Risk** | −5 | 6 | 6 |
| dji_2019_11 | Djibouti | 2019-11-21 | **Actionable_Risk** | −5 | 5 | 5 |
| eri_2019_08 | Eritrea | 2019-08-15 | **Actionable_Risk** | −5 | 5 | 5 |
| eth_2021_05 | Ethiopia | 2021-05-15 | Assess | −5 | 0 | 2 |
| ken_2024_04 | Kenya | 2024-04-24 | **Actionable_Risk** | +2 | 15 | 33 |
| rwa_2023_05 | Rwanda | 2023-05-02 | **Actionable_Risk** | −5 | 2 | 3 |
| sdn_2019_08 | Sudan | 2019-08-25 | **Actionable_Risk** | −5 | 12 | 16 |
| som_2023_09 | Somalia | 2023-09-25 | **Actionable_Risk** | +5 | 3 | 3 |
| ssd_2019_10 | South Sudan | 2019-10-15 | Assess | −2 | 0 | 1 |
| tza_2024_04 | Tanzania | 2024-04-15 | **Actionable_Risk** | +3 | 11 | 12 |
| uga_2019_05 | Uganda | 2019-05-15 | **Actionable_Risk** | +6 | 8 | 19 |

**9 of 11 reached Actionable_Risk (Red)**; Ethiopia and South Sudan reached
Assess (Orange). Note several events peak at day −5 (window start) — see §2 for
why, and what a more pre-event-weighted window would change. Full per-day
timelines: `flood_events_run_notes.md` §6b. Delivered to the crma-api on GCS
(§3 of `flood_events_run_notes.md` §6c).

---

## 2. Per-event window — what "before the event" means here

Every event was run on a **15-day window anchored to a single best-known peak
date**, `[peak − 5, peak + 9]` — i.e. **5 days before the peak and 9 days
after** (the event sits at ~day 6 of 15, mirroring the operational Mar-2026 run,
where the Nairobi flood of Mar 6–7 sat inside the Mar 1–15 window).

| event | country | event peak (≈) | run window (15 d) | days **before** peak | days after |
|-------|---------|----------------|-------------------|:--------------------:|:----------:|
| bdi_2024_04 | Burundi | 2024-04-15 | 2024-04-10 → 2024-04-24 | 5 | 9 |
| dji_2019_11 | Djibouti | 2019-11-21 | 2019-11-16 → 2019-11-30 | 5 | 9 |
| eri_2019_08 | Eritrea | 2019-08-15 | 2019-08-10 → 2019-08-24 | 5 | 9 |
| eth_2021_05 | Ethiopia | 2021-05-15 | 2021-05-10 → 2021-05-24 | 5 | 9 |
| ken_2024_04 | Kenya | 2024-04-24 | 2024-04-19 → 2024-05-03 | 5 | 9 |
| rwa_2023_05 | Rwanda | 2023-05-02 | 2023-04-27 → 2023-05-11 | 5 | 9 |
| sdn_2019_08 | Sudan | 2019-08-25 | 2019-08-20 → 2019-09-03 | 5 | 9 |
| som_2023_09 | Somalia | 2023-09-25 | 2023-09-20 → 2023-10-04 | 5 | 9 |
| ssd_2019_10 | South Sudan | 2019-10-15 | 2019-10-10 → 2019-10-24 | 5 | 9 |
| tza_2024_04 | Tanzania | 2024-04-15 | 2024-04-10 → 2024-04-24 | 5 | 9 |
| uga_2019_05 | Uganda | 2019-05-15 | 2019-05-10 → 2019-05-24 | 5 | 9 |

### Two clarifications about the current run

1. **It is NOT "15 days before the event."** It is **5 days of lead-in + the
   peak day + 9 days of aftermath**. Only ~5 forecast inits precede the event,
   and ~9 days cover the event itself plus the recession. This was a deliberate
   copy of the Mar-2026 operational layout, not a pre-event-coverage design.
2. **Each event is anchored to one `peak_date`, not its full duration.** The
   source page (`#flood-events`) gives only month-level dates + EM-DAT ids; the
   precise event span (e.g. "12–14 March") is not encoded. `peak_date` in
   `flood_events.yaml` is the single best-known onset/peak, and the 15-day
   window is derived from it. So an event's multi-day duration is *not* modelled
   — it is collapsed to one anchor day.

### Your idea — weight the window toward *before* the event

For an event like **12–14 March 2019**, the current scheme (peak ≈ Mar 13)
gives `Mar 8 → Mar 27` — only Mar 8–12 (5 days) precede the event. If the goal
is to explore the **build-up** (antecedent saturation + how early the forecast
"saw it coming"), a pre-event-weighted window is better. The split is a config
knob, so any of these is a one-line change:

| intent | `window_pre` / `window_post` | window for peak = Mar 13 |
|--------|:----------------------------:|--------------------------|
| current (Mar-2026 layout) | 5 / 9 | Mar 8 → Mar 27 |
| event-centred | 7 / 7 | Mar 6 → Mar 20 |
| **15 days up to the event** | 14 / 0 | **Feb 27 → Mar 13** |
| long lead-in + short tail | 11 / 3 | Mar 2 → Mar 16 |

What actually changes as you add lead-in days:

- **Antecedent** uses a fixed **7-day** IMERG lookback, so beyond ~7–10
  pre-event days it adds no *new* soil-moisture signal — but each extra
  pre-event day adds another **forecast init** (each covers D→D+7), which is
  exactly what extends the "how many days ahead did the BN flag it" lead-time
  evaluation you're after.
- The **DBN** temporal chain resets every **7 days** (lookback = 7), so ~7+
  lead-in days let the risk posterior accumulate before the event.
- Trade-off: more pre-event days ⇒ fewer post-event days ⇒ less coverage of the
  recession and false-alarm decay.

**To switch:** edit `defaults.window_pre` / `window_post` (or add per-event
overrides) in `flood_events.yaml` and re-run `./run_all_flood_events.sh` — no
code change needed; the driver already derives the window and the prep accepts
any date range. WB2 IFS-ENS covers **2016 → 2024**, so even a 14-day pre-event
window for the earliest event (Eritrea, Aug 2019) has ample preceding forecast
data.

---

## 3. Disk footprint — where the ~1.5 GB is

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

## 4. Reducing the footprint (if needed)

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

## 5. Wall-clock & compute notes

- Per-event ≈ 12–15 min after the range-mode optimization (`5a7a929`): prep
  ~8 min (15 days × ~30 s) + Julia DBN ~5 min + generators ~1 min.
- The pre-optimization per-day path was ~3–6 min **per day** (each day a fresh
  subprocess re-opening every store and re-running the WB2 lead interpolation 7×)
  ⇒ the 10-event batch would have taken ~17 h; it ran in ~2.5 h instead.
- One-time setup: Julia 1.12.6 + RxInfer 4.7.3 precompile (~9 min).
- Memory ceiling 7 GB drove two design choices: per-init WB2 loading in the
  collector and sequential (not parallel) event execution.
