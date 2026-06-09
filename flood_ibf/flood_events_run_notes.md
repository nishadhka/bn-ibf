# Flood BN-IBF — 11 historical-event hindcast routine

Replays the operational flood BN-IBF 15-day routine (validated for Nairobi,
Mar 2026 — see `flood_bn_ibf_run_notes_2026-03.md`) over the **11 historical
East-Africa flood events** from the GHACOF73 side event
([icpac-igad/DevOps-hazard-modeling#flood-events](https://github.com/icpac-igad/DevOps-hazard-modeling#flood-events)).

Each event is run as a **15-day window** `[peak − 5, peak + 9]` (same layout as
the Mar-2026 run, where the event sat at ~day 6 of 15). The BN engine, soft-
evidence schema, DBN temporal coupling, CRMA cost-loss output, and web-artifact
generators are **unchanged** — only the forecast data source is swapped.

---

## 1. The forecast-source problem and its resolution

The operational pipeline reads the ECMWF 51-member ensemble from
`forecasts/ecmwf_ea_tp_icechunk` on source.coop, which only holds **recent**
forecast inits. The 11 events span **2019–2024**, so that store has no data for
them.

**Resolution:** WeatherBench2 publishes the **real archived ECMWF IFS-ENS**
forecast at `gs://weatherbench2/datasets/ifs_ens/2016-2024-1440x721.zarr`:

| Property | Value |
|----------|-------|
| Model | ECMWF IFS ENS (the same operational model, archived) |
| Members | **50** (`number`) |
| Resolution | 0.25° (1440×721) — matches the operational store |
| Coverage | **2016-01-01 → 2024-12-31**, 00Z & 12Z inits (covers all 11 events) |
| Leads | 0–360 h (15 days), **6-hourly** (`prediction_timedelta`, 61 steps) |
| Variable | `total_precipitation` (metres, cumulative-since-init) |
| Access | public, anonymous (`gcsfs token="anon"`) |

This is a near drop-in for the operational store and **is** the ensemble, so
no ERA5 / EDA substitution is needed — the historical runs use genuine archived
ECMWF *forecasts*, making this an honest **forecast hindcast** (what the IBF
system would have issued at the time), not a perfect-foresight reanalysis run.

The antecedent half is **unchanged** — IMERG half-hourly
(`observations/imerg_hh_icechunk`, coverage 2000-06 → 2026-04) for the 7-day
antecedent sum and trend. CMORPH 2-yr return-period thresholds come from the
icechunk store as in the operational path.

### Adapter (`flood_data_prep.py`)
- New `--forecast-source {ecmwf_icechunk, ifs_ens_wb2}` (default
  `ecmwf_icechunk` — operational path untouched) and `--forecast-init-hour`
  (default 0 = 00Z).
- `open_ifs_ens_wb2()` opens the WB2 store anonymously, subsets to the East-
  Africa bbox, and **normalizes the schema** to the existing interface
  (`total_precipitation→tp`; `time→init_date`, `number→member`,
  `prediction_timedelta→lead_time` as `timedelta64`, `latitude→lat`,
  `longitude→lon`; units stay metres → existing ×1000 mm conversion). Every
  downstream stage (`ecmwf_window_accums`, exceedance/tail aggregation, soft-
  binning) is reused verbatim.

---

## 2. The 11 events

| key | country | location | EM-DAT | peak (≈) | 15-day window |
|-----|---------|----------|--------|----------|---------------|
| `bdi_2024_04` | Burundi | Bujumbura / Gatumba | 2024-0232-BDI | 2024-04-15 | 04-10 → 04-24 |
| `dji_2019_11` | Djibouti | Djibouti City | 2019-0579-DJI | 2019-11-21 | 11-16 → 11-30 |
| `eri_2019_08` | Eritrea | Highlands | 2019-IBF03-ERI | 2019-08-15 | 08-10 → 08-24 |
| `eth_2021_05` | Ethiopia | Addis / Akaki River | 2021-0343-ETH | 2021-05-15 | 05-10 → 05-24 |
| `ken_2024_04` | Kenya | Nairobi | 2024-0247-KEN | 2024-04-24 | 04-19 → 05-03 |
| `rwa_2023_05` | Rwanda | Western/Northern (nationwide) | 2023-0267-RWA | 2023-05-02 | 04-27 → 05-11 |
| `sdn_2019_08` | Sudan | Khartoum | 2019-0392-SDN | 2019-08-25 | 08-20 → 09-03 |
| `som_2023_09` | Somalia | South (Shabelle/Juba) | 2023-0741-SOM | 2023-09-25 | 09-20 → 10-04 |
| `ssd_2019_10` | South Sudan | Upper Nile | 2019-0486-SSD | 2019-10-15 | 10-10 → 10-24 |
| `tza_2024_04` | Tanzania | Dar es Salaam | 2024-0203-TZA | 2024-04-15 | 04-10 → 04-24 |
| `uga_2019_05` | Uganda | nationwide | 2019-0254-UGA | 2019-05-15 | 05-10 → 05-24 |

Peak dates and the 5/9 window split live in `flood_events.yaml`. They are
best-known (the source page gives month-level dates + EM-DAT numbers); refine
against EM-DAT / DesInventar onset dates as needed — the 9-day post-event tail
absorbs slippage of a few days.

---

## 3. Running it

```bash
cd flood_ibf
# prerequisites: juliaup + instantiated project (see README §Environment),
# `uv`, and icpac_adm1v3.geojson present in this directory.

# one event end-to-end (prep → DBN → parquet + bn-dag JSON)
./run_flood_event.sh ken_2024_04

# all 11 events (or a subset)
./run_all_flood_events.sh
./run_all_flood_events.sh ken_2024_04 rwa_2023_05
```

Single-day adapter check (no geojson needed for the forecast read itself):
```bash
uv run python flood_data_prep.py --date 2019-05-15 \
    --forecast-source ifs_ens_wb2 --soft-evidence \
    --adm1 icpac_adm1v3.geojson --out /tmp/check.csv
```

### Outputs (per event, under `output/events/<key>/`)
| Path | Description |
|------|-------------|
| `bn_inputs/flood_inputs_<D>_soft.csv` | per-day soft-evidence (15 files, 227 rows) |
| `dbn/flood_bn_v1_<D>.csv` | per-day DBN posteriors (15 files) |
| `flood_bn_v1_dbn_window.csv` | combined DBN output (15 × 227 rows) |
| `flood_bn_ibf_daily.parquet` | calendar-API parquet (15 rows) |
| `flood_bn_ibf_boundary_daily.parquet` | choropleth-API parquet (3 405 rows) |
| `bn-dag/bn-dag-<D>.json` | BN-DAG panel JSON (15 files) |

---

## 4. Pipeline / DBN parameters

Identical to the Mar-2026 operational run:

| Parameter | Value |
|-----------|-------|
| Return period | 2 yr (pixel-wise CMORPH) |
| Tail-risk aggregation | pixel p95 of `ens_max / RP` |
| DBN temporal decay α | 0.60 |
| DBN lookback L | 7 days |
| Cost-loss ratio γ | 0.20 |
| Members | 50 (WB2 IFS-ENS) |
| Boundaries | 227 ICPAC admin-1 |

The generalized DBN driver `run_flood_dbn_window.jl` (`--input-dir/--out-dir/
--expect`) replaces the date-hardcoded `run_flood_dbn_15day.jl`.

---

## 5. Caveats specific to the WB2 IFS-ENS hindcast

1. **6-hourly leads + scattered NaN-lead gaps → gap-filled by interpolation.**
   WB2 IFS-ENS leads are 6-hourly, and the raw cumulative `total_precipitation`
   has **all-NaN lead steps scattered across the forecast, different per init**
   (always 6 h, plus an init-dependent handful, e.g. 24/48/72/120/150 h). Because
   the field is cumulative-since-init and therefore monotonic in lead,
   `open_ifs_ens_wb2()` caps the lead axis at 168 h (7 d, the longest duration
   used) and fills those gaps by **linear interpolation over `lead_time`** plus
   `ffill`/`bfill` at the ends (needs `bottleneck`). After filling, every
   duration accumulation is well-defined; verified for the Apr-2024 Kenya window
   (24 h ens-max 132–1001 mm, 7 d ens-max 0.9–2.1 m, exceedance pixels
   3 300–5 600/day, no gaps). The only duration not independently observed is
   **3 h** (snaps to lead 0 h → 0); 6 h is the interpolated value between 0 and
   12 h. The effective finest *independent* accumulation window is therefore
   12 h, so short sub-daily convective bursts seen only at 3–6 h are not resolved.
2. **50 vs 51 members.** WB2 IFS-ENS has 50 members vs the operational store's
   51; exceedance/tail denominators (`mean(dim=member)`) adjust automatically.
3. **Forecast hindcast, not perfect foresight.** These are genuine archived
   forecasts initialized at each window day's 00Z — they carry real forecast
   error, so a missed event is a real miss, not a data artifact.
4. **Approximate peak dates** — see §2; tune in `flood_events.yaml`.
5. **`icpac_adm1v3.geojson` is a runtime prerequisite** (not in git), same as
   the operational run.

---

## 6. Results

### 6a. Forecast collection (`collect_wb2_ifs_events.py`)

`collect_wb2_ifs_events.py` pulls the WB2 IFS-ENS forecast for all 11 windows
and writes, per event, under `wb2_ifs_ens/<key>/`:
`forecast_accums.nc` (the 50-member duration-accumulation cube
`(init_date, duration, member, lat, lon)` in mm + `exceedance_frac`, `p_heavy`,
`ens_max_ratio` grids) and `summary.csv`; plus a combined
`wb2_ifs_ens/events_summary.csv`. This is the full forecast side of the BN
inputs on the WB2 grid — everything except the admin-1 zonal step (which needs
`icpac_adm1v3.geojson`). Run: `./collect_wb2_ifs_events.py [--only <key> ...]`.

**Domain-wide forecast exceedance signal vs the CMORPH 2-yr RP** (East-Africa
box, all 50 members; `p_heavy` = peak-pixel ensemble exceedance fraction,
`px_exceed` = pixels where ≥1 member crosses the 2-yr RP at the event peak day):

| event | country | peak-day p_heavy | peak-day px_exceed | peak-day ens24h_mm | peak-day ens7d_mm | window-max px_exceed (day) |
|-------|---------|-----------------:|-------------------:|-------------------:|------------------:|---------------------------:|
| bdi_2024_04 | Burundi     | 1.00 | 3090 | 282 | 652  | 4419 (d+9) |
| dji_2019_11 | Djibouti    | 1.00 | 6164 | 184 | 573  | 6164 (d+0) |
| eri_2019_08 | Eritrea     | 1.00 | 3959 | 110 | 382  | 4375 (d+1) |
| eth_2021_05 | Ethiopia    | 0.96 |  127 | 123 | 254  |  978 (d−5) |
| ken_2024_04 | Kenya       | 1.00 | 4419 | 177 | 1017 | 5620 (d+5) |
| rwa_2023_05 | Rwanda      | 0.86 |  975 |  77 | 200  | 3444 (d−5) |
| sdn_2019_08 | Sudan       | 1.00 | 2023 | 237 | 322  | 3695 (d−3) |
| som_2023_09 | Somalia     | 1.00 |  533 | 347 | 651  |  764 (d+3) |
| ssd_2019_10 | South Sudan | 1.00 | 5177 | 307 | 389  | 5240 (d−3) |
| tza_2024_04 | Tanzania    | 1.00 | 3090 | 282 | 652  | 4419 (d+9) |
| uga_2019_05 | Uganda      | 0.96 | 2592 | 236 | 446  | 3508 (d+3) |

(Burundi and Tanzania are identical here because both peak 2024-04-15 → the same
15-day window over the same East-Africa forecast field; they separate only at the
admin-1 level.) All 11 events show a strong forecast exceedance signal across
their window — the WB2 IFS-ENS forecast "saw" heavy rain crossing the 2-yr RP at
each event. These are **domain-wide** statistics; the per-event peak day reflects
the heaviest pixel anywhere in East Africa, not necessarily the affected country.

### 6b. Admin-1 CRMA validation (pending the BN run)

The boundary-level CRMA table — peak state in the affected admin-1 around the
event days, mirroring the Nairobi validation in
`flood_bn_ibf_run_notes_2026-03.md` §2 — requires the full BN run via
`./run_flood_event.sh <key>`, which needs `icpac_adm1v3.geojson` and a Julia
environment (neither present on the collection host).

| event | affected adm-1 | peak CRMA (event window) | lead vs peak | notes |
|-------|----------------|--------------------------|--------------|-------|
| ken_2024_04 | Nairobi | _tbd_ | _tbd_ | Apr-2024 Kenya floods |
| … | | | | |
