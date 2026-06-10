# Flood BN-IBF — 11-event run record & disk footprint

What was run to produce the historical 11-event flood hindcast (June 2026),
the artifacts it created, and **why the run occupies ~1.5 GB on disk**.

Companion to `flood_events_run_notes.md` (method, events, results) and
`flood_bn_ibf_run_notes_2026-03.md` (the operational Mar-2026 run this extends).

> **Version — this records the 16-day `[peak−10, peak+5]` run** (the current one
> on GCS). An earlier pass used a 15-day `[peak−5, peak+9]` window; it was
> replaced after the day−5 window-edge effect showed the build-up was being
> clipped (see §2). All counts below are the 16-day run: 162 dates delivered,
> all 11 events reach Actionable_Risk in-country.

---

## 1. What the run did

Replayed the validated 15-day flood BN-IBF routine over the **11 GHACOF73
historical flood events** (2019–2024), using the **WeatherBench2 archived ECMWF
IFS-ENS** 50-member forecast (`gs://weatherbench2/datasets/ifs_ens/2016-2024-1440x721.zarr`)
in place of the recent-only operational store, plus IMERG antecedent and CMORPH
2-yr return-period thresholds. Each event = a 16-day window `[peak−10, peak+5]`
(see §2 for the window choice; an earlier pass used `[peak−5, peak+9]`).

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

**Scale:** 11 events × 16 days = **176 boundary-days**, each over 227 admin-1
regions and **50 ensemble members** ⇒ ~2.0 M boundary-member BN evaluations.

**Result (16-day window `[peak−10, peak+5]`; worst CRMA reached in the affected
country, the day it occurs vs peak, and the earliest day reaching Assess+):**

| event | country | peak | worst CRMA | worst day | #AR | #Assess+ | earliest Assess+ |
|-------|---------|------|------------|:---------:|:---:|:--------:|:----------------:|
| bdi_2024_04 | Burundi | 2024-04-15 | **Actionable_Risk** | +4 | 6 | 9 | −10 |
| dji_2019_11 | Djibouti | 2019-11-21 | **Actionable_Risk** | −1 | 6 | 6 | −10 |
| eri_2019_08 | Eritrea | 2019-08-15 | **Actionable_Risk** | −10 | 6 | 6 | −10 |
| eth_2021_05 | Ethiopia | 2021-05-15 | **Actionable_Risk** | −10 | 1 | 3 | −10 |
| ken_2024_04 | Kenya | 2024-04-24 | **Actionable_Risk** | −1 | 21 | 28 | −10 |
| rwa_2023_05 | Rwanda | 2023-05-02 | **Actionable_Risk** | 0 | 2 | 5 | −10 |
| sdn_2019_08 | Sudan | 2019-08-25 | **Actionable_Risk** | −10 | 14 | 18 | −10 |
| som_2023_09 | Somalia | 2023-09-25 | **Actionable_Risk** | −2 | 5 | 5 | −3 |
| ssd_2019_10 | South Sudan | 2019-10-15 | **Actionable_Risk** | 0 | 2 | 7 | −10 |
| tza_2024_04 | Tanzania | 2024-04-15 | **Actionable_Risk** | −3 | 13 | 15 | −10 |
| uga_2019_05 | Uganda | 2019-05-15 | **Actionable_Risk** | +5 | 6 | 15 | −10 |

**All 11 events now reach Actionable_Risk (Red)** in the affected country (vs
9/11 under the earlier 5/9 split — Ethiopia and South Sudan are promoted because
the longer build-up window catches more high-risk days). Kenya is strongest
(21 Red / 28 Assess+ boundaries).

**Key finding — the risk window is *prolonged*, not a spike.** The earliest
Assess+ day is **−10 (the window's first day) for 10 of 11 events**. The
window-edge effect seen under 5/9 (events peaking at day −5) did not disappear —
it **moved to day −10**. These are **wet-season saturation floods**: the ground
is already saturated and rain is already forecast 10+ days before the peak, so
the BN flags Assess/Actionable *continuously* through the rainy spell, and the
build-up extends beyond even 10 days. For clean event-vs-background separation a
future step would be an **anomaly framing** (risk above the seasonal norm) rather
than only a longer window. Full per-day timelines: `flood_events_run_notes.md`
§6b; GCS delivery: §6c.

---

## 2. Per-event window — what "before the event" means here

The run uses a **16-day window anchored to a single best-known peak date**,
`[peak − 10, peak + 5]` — **10 days before the peak + the peak + 5 after**. This
replaced an earlier **5/9** split (`[peak−5, peak+9]`, copied from the Mar-2026
operational layout) once we saw the risk signal was pressing against the
5-day-before edge (see §1). The longer pre-window weights coverage toward the
**build-up** and guarantees the full 7-day forecast lead-time to the peak is
captured.

| event | country | event peak (≈) | run window (16 d) | days **before** peak | days after |
|-------|---------|----------------|-------------------|:--------------------:|:----------:|
| bdi_2024_04 | Burundi | 2024-04-15 | 2024-04-05 → 2024-04-20 | 10 | 5 |
| dji_2019_11 | Djibouti | 2019-11-21 | 2019-11-11 → 2019-11-26 | 10 | 5 |
| eri_2019_08 | Eritrea | 2019-08-15 | 2019-08-05 → 2019-08-20 | 10 | 5 |
| eth_2021_05 | Ethiopia | 2021-05-15 | 2021-05-05 → 2021-05-20 | 10 | 5 |
| ken_2024_04 | Kenya | 2024-04-24 | 2024-04-14 → 2024-04-29 | 10 | 5 |
| rwa_2023_05 | Rwanda | 2023-05-02 | 2023-04-22 → 2023-05-07 | 10 | 5 |
| sdn_2019_08 | Sudan | 2019-08-25 | 2019-08-15 → 2019-08-30 | 10 | 5 |
| som_2023_09 | Somalia | 2023-09-25 | 2023-09-15 → 2023-09-30 | 10 | 5 |
| ssd_2019_10 | South Sudan | 2019-10-15 | 2019-10-05 → 2019-10-20 | 10 | 5 |
| tza_2024_04 | Tanzania | 2024-04-15 | 2024-04-05 → 2024-04-20 | 10 | 5 |
| uga_2019_05 | Uganda | 2019-05-15 | 2019-05-05 → 2019-05-20 | 10 | 5 |

Each day in the window is still its **own independent BN analysis** with a 7-day
IMERG antecedent + IFS-ENS forecast out to 7 days; the window only sets which
target days are run. Earliest day whose forecast reaches the peak is **peak−7**
(the 7-day forecast horizon), so days peak−10/−9/−8 add build-up context but
cannot forecast the peak itself. Change the split via `window_pre`/`window_post`
in `flood_events.yaml` (per-event overrides allowed) and re-run.

### Clarification — anchored to one peak date, not the event duration

Each event is anchored to a single `peak_date`, not its full multi-day span.
The source page (`#flood-events`) gives only month-level dates + EM-DAT ids; the
precise event window (e.g. "12–14 March") is not encoded. `peak_date` in
`flood_events.yaml` is the single best-known onset/peak, and the 16-day window is
derived from it — so a multi-day event is collapsed to one anchor day. The wide
10-day pre-window means a peak estimate that is a few days *late* still covers
the true event (it falls inside the pre-window).

### Window history & how to change it

| window | `pre`/`post` | total | rationale |
|--------|:------------:|:-----:|-----------|
| Mar-2026 layout (initial) | 5 / 9 | 15 | copied operational; clipped the build-up |
| **current** | **10 / 5** | **16** | build-up focus; full 7-day lead-time to peak |
| event-centred | 7 / 7 | 15 | symmetric |
| 15 days up to the event | 14 / 0 | 15 | pure pre-event runway |

What each extra lead-in day adds: the **antecedent** uses a fixed 7-day IMERG
lookback (so >7–10 pre-event days add no *new* soil-moisture signal), but each
extra day adds another **forecast init** (covers D→D+7) and lets the **DBN**
(7-day reset) posterior accumulate. Trade-off: more pre ⇒ fewer post ⇒ less
recession coverage. **To change:** edit `defaults.window_pre`/`window_post` (or
per-event overrides) in `flood_events.yaml` and re-run `./run_all_flood_events.sh`
— no code change; the driver derives the window and `--expect` from it.

> The §1 result showed this run's risk signal still saturates the window edge at
> day −10 — i.e. the build-up for these wet-season floods runs even longer. A
> still-longer pre-window would extend the picture, but the cleaner next step is
> an **anomaly framing** (risk above the seasonal climatology) so the event
> separates from the background wet-season risk.

---

## 3. Disk footprint — where the ~1.5 GB is

| Path | Size | What |
|------|-----:|------|
| `wb2_ifs_ens/*/forecast_accums.nc` | **~1.5 GB** | 11 collected forecast cubes (50-member duration accums) — optional, regenerable cache |
| `output/bn-dag/*.json` | 19 MB | 154 per-day BN-DAG panel JSONs (139 event + 15 operational) |
| `output/events/*/` | 48 MB | per-event inputs + DBN CSVs + per-event JSONs/parquet |
| `output/*.parquet` | <1 MB | merged calendar + choropleth parquet |
| **total** | **~1.5 GB** | |

**95 % of the footprint is the 11 forecast cubes.** Everything the BN produces
(evidence CSVs, DBN posteriors, DAG JSONs, parquets) is only ~68 MB combined.

### Why each cube is ~135 MB

`collect_wb2_ifs_events.py` saves, per event, the **full 50-member ensemble**
forecast as `tp_accum_mm(init_date, duration, member, lat, lon)` for the 16-day
window:

```
 16 inits × 5 durations × 50 members × 145 lat × 125 lon × 4 bytes (float32)
   = 290,000,000 values·bytes ≈ 277 MiB uncompressed
   → ~135 MiB on disk (zlib level-4, ~2× — precip is sparse/mostly small)
 × 11 events ≈ 1.5 GB
```

The size is driven almost entirely by the **`member` dimension (50×)**. The cube
keeps every ensemble member so per-member storyline analysis (worst/median/best
plausible world) is reproducible offline without re-fetching from GCS. The
forecast field over East Africa is otherwise small (145×125 ≈ 18 k pixels at
0.25°); it is the 50-member × 16-init × 5-duration product that inflates it.

### Why this is expected, not a leak

- A single deterministic field for one event/day over this box is ~70 KB. The
  ensemble (50 members) × 16 daily inits × 5 accumulation windows is **4 000×**
  that — the cost of keeping a *probabilistic, time-resolved* forecast archive.
- These cubes are the **forecast-collection deliverable** (the `collect` step),
  separate from the BN run itself. The BN only needs them transiently; it
  actually re-reads WB2 directly during prep, so the cubes are an optional cache
  / audit artifact.

---

## 4. Reducing the footprint (if needed)

The ~1.5 GB of cubes is **gitignored** (`flood_ibf/wb2_ifs_ens/**/forecast_accums.nc`)
and fully regenerable via `./collect_wb2_ifs_events.py` (it reads the current
window from `flood_events.yaml`, so the on-disk copy is whatever window it was
last collected for — it has no effect on the BN run or the API). Options:

| Action | New size | Trade-off |
|--------|---------:|-----------|
| Delete cubes after upload | ~68 MB | lose offline forecast cache; regenerate on demand |
| Keep ensemble summaries only (drop `member`) | ~30 MB total | lose per-member storylines; keep mean/exceedance/tail |
| Keep members but fewer durations (24h+7d only) | ~560 MB | coarser duration resolution |
| Store as cloud zarr instead of local nc | 0 local | needs network to read |

The committed git artifacts are unaffected by this — only the BN outputs
(`output/`, ~68 MB, of which 19 MB of JSON + the parquets are committed) and the
delivered GCS objects matter for the API. To reclaim the ~1.5 GB safely:

```bash
rm -rf flood_ibf/wb2_ifs_ens          # cubes only; regenerate with collect_wb2_ifs_events.py
```

---

## 5. Wall-clock & compute notes

- Per-event ≈ 12–16 min after the range-mode optimization (`5a7a929`): prep
  ~8 min (16 days × ~30 s) + Julia DBN ~5 min + generators ~1 min.
- The pre-optimization per-day path was ~3–6 min **per day** (each day a fresh
  subprocess re-opening every store and re-running the WB2 lead interpolation 7×)
  ⇒ the 10-event batch would have taken ~17 h; it ran in ~2.5 h instead.
- One-time setup: Julia 1.12.6 + RxInfer 4.7.3 precompile (~9 min).
- Memory ceiling 7 GB drove two design choices: per-init WB2 loading in the
  collector and sequential (not parallel) event execution.
