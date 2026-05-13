# Flood IBF — Bayesian Network for East Africa

Daily admin-1 flood risk assessment for the ICPAC domain (11 East
African countries, 227 admin-1 regions). Combines IMERG satellite
rainfall observations with ECMWF 51-member ensemble forecasts through a
Bayesian Network with **soft evidence**, **dynamic temporal coupling**,
and **per-member storyline selection** to produce a continuously updated
risk posterior and a cost-loss-triggered CRMA output (Monitor /
Evaluate / Assess / Actionable_Risk) per boundary.

Operational run for 2026-03-01 → 2026-03-10 successfully flagged the
Nairobi River flash-flood event of 6–7 March 2026 with a 2-day lead time
(Assess on Mar 4, Actionable_Risk on Mar 7), a signal that an ensemble-
mean threshold system would have missed.

---

## Method at a glance

```
    IMERG HH icechunk          ECMWF TP icechunk            CMORPH return
  (observations, 0.1°)      (51-member ens, 0.25°)        periods (0.073°)
          │                          │                           │
          ▼                          ▼                           ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                      flood_data_prep.py                          │
    │  • 7-day antecedent mm + trend slope  (IMERG → admin-1)          │
    │  • 7 duration accumulations (3h…7d)  (ECMWF → admin-1)           │
    │  • P(any member ≥ 2yr RP) per pixel-duration  (eprob grid)       │
    │  • P_heavy = max over durations                                  │
    │  • ens_max_ratio = p95 of (ens_max / threshold) per boundary     │
    │  • hotspot_fraction = pixels where any member exceeds threshold  │
    │  • Gaussian soft-bin columns: ant_p1..p5, exc_p1..p5, spa_p1..p3,│
    │    trn_p1..p3, tail_p1..p4  (optional; auto-detected downstream) │
    └─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ flood_inputs_YYYY-MM-DD.csv
    ┌─────────────────────────────────────────────────────────────────┐
    │                    flood_bn_ibf_v1.jl                            │
    │                                                                  │
    │  Evidence nodes (5, with tail_risk; agreement disabled):         │
    │    antecedent (Dry…Saturated)    ─────┐                          │
    │    exceedance_prob               ─────┤                          │
    │    spatial_coverage              ─────┼─▶ risk_level             │
    │    rainfall_trend                ─────┤     (5 states)           │
    │    tail_risk (ens-max signal)    ─────┘                          │
    │                                                                  │
    │  Soft evidence via Pearl virtual-evidence channel on each parent │
    │  DBN temporal coupling  : yesterday's posterior blended (α=0.6,  │
    │                           lookback L=7) into today's R_obs       │
    │  Per-member storyline   : BN runs 51× per boundary-day, worst /  │
    │                           median / best picked by P(High∪Extreme)│
    │  Primary inference      : RxInfer.jl message passing             │
    │  Bulk inference         : direct tensor contraction (≈4 500 flops│
    │                           / boundary; matches RxInfer to 1e-9)   │
    │  CRMA output            : cost-loss rule over risk posterior     │
    │                           (Monitor / Evaluate / Assess /         │
    │                            Actionable_Risk; C/L default 0.20)    │
    └─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ flood_bn_v1_YYYY-MM-DD.csv
                         Summary, maps, per-day DAGs, storylines
```

### Key design choices

| Decision | Value | Rationale |
|----------|-------|-----------|
| Target dates | D ∈ {Mar 1…Mar 10 2026} | 10-day hindcast window |
| Observation window | `[D-7, D)` IMERG daily totals | Antecedent moisture (7d) |
| Forecast init | `init_date = D 00Z` | Latest init available for D |
| Durations evaluated | 3h, 6h, 12h, 24h, 48h, 72h, 7d | Matches CMORPH RP durations |
| Return period threshold | **2-year** (pixel-wise from CMORPH) | Sensitive IBF trigger |
| Tail-risk aggregation | **pixel 95th-percentile** of max_ratio | Preserves localized hotspots |
| Spatial coverage | `max(P_heavy≥0.5 mask, hotspot_fraction)` | Captures sub-boundary hot pixels |
| Trend threshold | ±2 mm/day on 7-day regression slope | Decreasing / Stable / Increasing |
| Agreement node | Disabled (`--no-agreement` is the default path) | ECMWF-only; enables 5-parent RxInfer path |
| Soft-binning σ | ant=10 mm, exc=0.05, spa=0.05, trn=1 mm/d, tail=0.15 | ≈25–30% of narrowest bin |
| DBN temporal decay | α = 0.6, lookback L = 7 days | Low-pass filter; resets weekly |
| Cost-loss ratio γ | 0.20 (default; `--cost-loss-ratio` override) | Mid-range FbF trigger (0.1–0.2) |

---

## Pipeline scripts

All Python scripts use the `uv run` shebang — dependencies install
transiently at first execution (no persistent conda env needed).

### Data pipeline

| Script | Role |
|--------|------|
| `flood_data_prep.py` | Per-day data preparation. Reads IMERG + ECMWF icechunk stores and CMORPH RP NetCDF; writes `bn_inputs/flood_inputs_YYYY-MM-DD.csv` (227 rows) including `ens_max_ratio`, `hotspot_fraction`, `ens_min/mean/max_24h_mm`, and optional Gaussian soft-bin columns (`ant_p1..p5`, `exc_p1..p5`, `spa_p1..p3`, `trn_p1..p3`, `tail_p1..p4`) for virtual-evidence inference. Pencil-zarr layout activated via `--pencil` for per-pixel/member queries. |
| `flood_bn_ibf_v1.jl` | Julia BN inference engine. Reads prep CSV, auto-detects soft-evidence columns, runs RxInfer message-passing inference by default (or direct-matmul with `--legacy-inference`), and writes `output/flood_bn_v1_YYYY-MM-DD.csv` with full risk posterior, CRMA state, traffic light, and cost-loss explanation. CLI: `--input-csv IN --output-csv OUT [--no-agreement] [--tail-risk] [--legacy-inference] [--cost-loss-ratio 0.2]`. Also exposes `run_dbn_sequence()` for multi-day DBN chaining and `run_per_member_bn()` + `select_storylines()` for per-member storyline evaluation. |
| `flood_bn_ibf_v1.py` | Python (pgmpy) reference implementation of the same BN. Used for cross-validation; not on the operational path. |
| `run_flood_bn_range.sh` | Driver: loops over `[START, END]`, runs prep → Julia BN each day. Usage: `./run_flood_bn_range.sh 2026-03-01 2026-03-10`. `RP_YEARS` env var overrides the default 2-yr threshold. |
| `summarize_bn.py` | Stacks the 10 daily result CSVs into a per-boundary summary (`max_risk_level`, `day_of_max`, `mean_confidence`, `days_at_alert_or_worse`). |

### Plot routines

| Script | Output | Purpose |
|--------|--------|---------|
| `plot_daily_risk_maps.py` | `output/flood_bn_v1_risk_maps_panel.png` (2×5 panel) and `output/maps/flood_bn_v1_risk_YYYY-MM-DD.png` (10 standalone) | Admin-1 choropleth maps of the daily BN risk level using a 5-step discrete colormap (Minimal → Extreme). |
| `plot_nairobi_diagnostic.py` | `output/nairobi_diagnostic.png` | 4-panel timeline for a single boundary (default Nairobi): antecedent moisture, ECMWF ensemble spread vs CMORPH 2-yr threshold, tail-risk ratio, and BN risk/CRMA per day. The reported flood window (Mar 6–7) is shaded red. |
| `plot_bn_dag_per_day.py` | `output/bn_dags/bn_dag_<boundary>_<date>.png` | Per-day per-boundary DAG visualization showing every evidence node with its raw value + discretized state, the risk_level posterior (5-bar), and the CRMA state (traffic light). Useful for diagnosing why a given risk call was made. |
| `flood_bn_dag.drawio` | — | Editable draw.io schematic of the v2 BN structure: 5 evidence parents, virtual-evidence channels, DBN temporal link, CRMA cost-loss output, per-member storyline sidebar. Open in [app.diagrams.net](https://app.diagrams.net). |

### Documentation files

| File | Contents |
|------|----------|
| `README.md` | This file. |
| `flood_bn_ibf_system_v20260412.md` | Deep technical doc (~850 lines): BN math, CPT construction, expert rules, variable elimination vs message passing vs direct matrix, factor graphs, RxInfer.jl mechanics, end-to-end walkthrough. |
| `flood_bn_ibf_v1_workflow.md` | Original design document with data source definitions, threshold tables, and the `FloodDataLoaderV1` / `FloodBayesianNetworkV1` class structure. |
| `python_julia_bn_comparison.md` | Detailed comparison of the Python (pgmpy) and Julia (RxInfer) implementations — what is identical, what differs, and where Julia enables capabilities Python cannot (streaming, parameter learning, soft evidence, hierarchical models). |
| `probabilistic_logic_v20260413.md` | Conceptual audit of the BN and the path from deductive-frequentist rule system to genuine probabilistic logic (soft evidence, DBN, storylines, cost-loss triggers). |
| `soft_evidence_upgrade_v20260417.md` | Gaussian soft-binning upgrade design notes. |
| `dbn_storyline_analysis_v20260418.md` | Dynamic Bayesian Network temporal coupling + per-member storyline selection design and validation notes. |
| `decision-output-riskassessment.md` | Reasoning for removing the Action node from the BN and replacing it with a deterministic cost-loss CRMA rule. |

---

## Data sources

Only local files listed here; the icechunk / zarr stores are read
directly from `s3://e4drr-project/*` on source.coop via anonymous access.

| Source | Path / URL | Used for |
|--------|-----------|----------|
| IMERG half-hourly | `s3://e4drr-project/observations/imerg_hh_icechunk` | Antecedent 7-day rainfall, trend |
| ECMWF IFS ensemble TP (pancake) | `s3://e4drr-project/forecasts/ecmwf_ea_tp_icechunk` | Full-grid zonal statistics |
| ECMWF IFS ensemble TP (pencil)  | `s3://e4drr-project/forecasts/ecmwf_ea_tp_zarr_pencil` | Per-pixel × all-members (storylines) |
| CMORPH return periods | `cmorph_ea_return_periods.nc` | Pixel-wise 2-yr RP thresholds |
| Admin-1 boundaries | `icpac_adm1v3.geojson` | 227 admin-1 polygons |

---

## Bayesian Network structure

5 discrete evidence parents → 1 hidden risk node, each parent carrying a
paired virtual-evidence observation channel. A deterministic cost-loss
rule then maps the risk posterior to a 4-state CRMA output outside the
BN (no fictitious action-CPT):

| Node | States | Role | Source |
|------|:------:|------|--------|
| `antecedent_rainfall` | Dry, Normal, Wet, Very_Wet, Saturated (5) | Evidence | IMERG 7-day sum |
| `exceedance_prob` | Very_Low, Low, Medium, High, Very_High (5) | Evidence | ECMWF P_heavy (ensemble mean) |
| `spatial_coverage` | Localized, Moderate, Widespread (3) | Evidence | `max(P_heavy≥0.5 mask, hotspot_fraction)` |
| `rainfall_trend` | Decreasing, Stable, Increasing (3) | Evidence | 7-day IMERG slope ±2 mm/day |
| `tail_risk` | None, Low, Moderate, High (4) | Evidence | p95 of (ens_max / 2-yr RP) per pixel |
| `forecast_agreement` | Low, Medium, High (3) | Retired | ECMWF-only; routed through legacy matmul path only |
| `risk_level` | Minimal, Low, Moderate, High, Extreme (5) | Hidden | Inferred (RxInfer MP or tensor contraction) |
| CRMA output | Monitor, Evaluate, Assess, Actionable_Risk (4) | Deterministic rule | Cost-loss trigger on posterior |

**Soft evidence**: each parent `X` receives an identity-CPT observation
channel `X_obs ~ DiscreteTransition(X, I_K)`. When fed a one-hot vector,
this reproduces hard classification; when fed a probability vector from
the Gaussian soft-binning step in `flood_data_prep.py`, it injects
Pearl's virtual evidence (Chan & Darwiche 2005), letting bin-edge
uncertainty propagate through the inference.

**DBN temporal coupling**: yesterday's risk posterior $\hat{r}_{t-1}$
is blended with a uniform prior (`v_t = α · r_{t-1} + (1-α) · u_5`,
`α = 0.6`) and observed on the risk node's own virtual-evidence channel
`R_obs`. The chain resets after `lookback = 7` consecutive days,
preventing indefinite anchoring to stale evidence. Implemented in
`run_dbn_sequence()` in `flood_bn_ibf_v1.jl`.

**Per-member storylines**: for each boundary × date, the BN is
evaluated independently on all 51 ensemble members (using member-
specific exceedance, spatial coverage, and tail risk; antecedent and
trend are shared). `select_storylines()` picks the worst / median /
best members by `P(High) + P(Extreme)` and reports the ensemble-
frequency probability that a world is at least as severe (`1/51 ≈ 2%`
for the worst member).

**Inference**: RxInfer.jl reactive message passing is the default path
(`@model flood_bn_model_5parent`). For bulk per-member runs
(~115 770 inferences over 10 days), a direct tensor contraction
`infer_soft_matmul()` is used, verified against RxInfer to
`|Δ| < 1.5 × 10⁻⁹`. Passing `--legacy-inference` routes back through the
old matmul-with-parent-index-flattening path. RxInfer's **optimised**
`@tullio`-based `DiscreteTransition` rules in `ReactiveMP/.../predefined/
belief_propagation.jl` are predefined for up to **4 conditioning parents**
(a 5-D CPT tensor); beyond that, RxInfer falls through to the generic
structured-message rule in `categoricals.jl`, which supports arbitrary
parent counts at a per-inference performance cost. `include_agreement =
true` (6 conditioning parents) therefore works but on the generic path,
which is why `infer_soft_matmul()` / `--legacy-inference` is the
preferred bulk-run alternative.

**CRMA cost-loss trigger**: the risk posterior is mapped to a 4-state
output (Monitor / Evaluate / Assess / Actionable_Risk) by thresholds
derived from the cost-loss ratio γ = C/L (default 0.20):

```
Actionable_Risk : P(High) + P(Extreme)              ≥ γ
Assess          : P(Mod) + P(High) + P(Extreme)     ≥ max(2γ, 0.40)
Evaluate        : P(Low) + P(Mod) + P(High) + P(Ex) ≥ max(3γ, 0.30)
Monitor         : otherwise
```

WMO-aligned traffic light: Monitor (Green), Evaluate (Yellow), Assess
(Orange), Actionable_Risk (Red).

**Why tail_risk matters**: the mean exceedance probability `P_heavy`
under-represents cases where only 1–2 of 51 ensemble members produce
extreme rainfall — this rounds to ~2–4% and discretizes to "Very_Low",
missing the tail signal entirely. The `tail_risk` node captures the
ratio of the ensemble maximum to the threshold; when any single
member-pixel combination hits ≥ the 2-yr threshold, the node fires
Moderate or High and directly elevates the risk posterior.

**Why pixel p95 matters**: the boundary-mean of `ens_max_ratio` smooths
over localized hotspots. For a small admin region (e.g. Nairobi,
~5 ECMWF pixels), a single hot pixel gets diluted 5×. Using the 95th
percentile of pixel-level ratios preserves the worst-case signal while
being robust to single outlier pixels.

---

## Running the pipeline

### One-shot for the Mar 1–10 window
```bash
./run_flood_bn_range.sh 2026-03-01 2026-03-10
./summarize_bn.py
./plot_daily_risk_maps.py
./plot_nairobi_diagnostic.py
./plot_bn_dag_per_day.py --boundary Nairobi --start 2026-03-01 --end 2026-03-10
```

### Single day, single boundary diagnostic
```bash
./flood_data_prep.py --date 2026-03-06 --out bn_inputs/flood_inputs_2026-03-06.csv
julia --project=. flood_bn_ibf_v1.jl \
    --input-csv bn_inputs/flood_inputs_2026-03-06.csv \
    --output-csv output/flood_bn_v1_2026-03-06.csv \
    --no-agreement --tail-risk --cost-loss-ratio 0.20
./plot_bn_dag_per_day.py --date 2026-03-06 --boundary Nairobi
```

### DBN chain + per-member storylines (from Julia REPL)
```julia
include("flood_bn_ibf_v1.jl")

# 10-day DBN sequence
csvs = ["bn_inputs/flood_inputs_$(d).csv" for d in
        "2026-03-01":"2026-03-10"]
dbn  = run_dbn_sequence(csvs; temporal_decay=0.6, lookback=7,
                        cost_loss_ratio=0.20)
CSV.write("output/flood_bn_dbn_seq.csv", dbn)

# Per-member storylines for one day
pm       = run_per_member_bn("bn_inputs/flood_inputs_members_2026-03-06.csv")
stories  = select_storylines(pm)
CSV.write("output/storylines_2026-03-06.csv", stories)
```

### Environment

#### Python
Scripts use the `uv run --with …` shebang — dependencies install transiently at
first execution. No persistent conda/venv is required; `uv` is the only
prerequisite (`curl -LsSf https://astral.sh/uv/install.sh | sh`).

#### Julia
Required: Julia ≥ 1.10. Install via `juliaup` and instantiate the project
environment (pulls `CSV`, `DataFrames`, and `RxInfer` pinned by
`Manifest.toml`):

```bash
# 1. Install juliaup + latest stable Julia (one-off)
curl -fsSL https://install.julialang.org | sh -s -- --yes \
     --default-channel release --background-selfupdate 0
export PATH="$HOME/.juliaup/bin:$PATH"          # add to ~/.bashrc for persistence
julia --version                                  # expect 1.10+ (tested on 1.12.6)

# 2. Install project deps (CSV, DataFrames, RxInfer) into this repo's env
cd flood_ibf
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# 3. Smoke-test — first call compiles RxInfer (~30 s); subsequent calls are warm
julia --project=. -e 'using RxInfer; println("RxInfer ", pkgversion(RxInfer))'
```

`RxInfer.jl` is **required** (not optional) on the primary inference
path: the BN is compiled as a proper `@model` with multi-parent
`DiscreteTransition` tensor CPTs, and inference goes through RxInfer's
reactive message passing so the same code path handles both hard
(one-hot) and soft (probability-vector) evidence on each parent, plus
the DBN temporal coupling via virtual evidence on `R_obs`. See
`probabilistic_logic_v20260413.md` §8 and
`soft_evidence_upgrade_v20260417.md` for background.

---

## Outputs directory layout

```
flood_ibf/
├── bn_inputs/                                  # daily evidence CSVs (per-run)
│   └── flood_inputs_2026-03-DD.csv             # 227 rows × ≥17 cols (+ soft-bin cols if enabled)
├── output/
│   ├── flood_bn_v1_2026-03-DD.csv              # 227 rows × risk posterior + CRMA + traffic light
│   ├── flood_bn_v1_2026-03-01_to_10_summary.csv  # per-boundary peak summary
│   ├── flood_bn_dbn_seq.csv                    # DBN chain output (boundary × day)
│   ├── storylines_2026-03-DD.csv               # per-day worst/median/best per boundary
│   ├── flood_bn_v1_risk_maps_panel.png         # 2×5 combined map
│   ├── nairobi_diagnostic.png                  # 4-panel Nairobi timeline
│   ├── maps/
│   │   └── flood_bn_v1_risk_2026-03-DD.png     # per-day standalone map
│   └── bn_dags/
│       └── bn_dag_<boundary>_2026-03-DD.png    # per-day BN DAG
├── flood_data_prep.py                          # data pipeline
├── flood_bn_ibf_v1.jl                          # Julia BN (operational)
├── flood_bn_ibf_v1.py                          # Python BN (reference)
├── run_flood_bn_range.sh                       # orchestrator
├── summarize_bn.py                             # 10-day aggregator
├── plot_daily_risk_maps.py                     # choropleth maps
├── plot_nairobi_diagnostic.py                  # boundary-specific timeline
├── plot_bn_dag_per_day.py                      # per-day DAG with posteriors
├── flood_bn_dag.drawio                         # editable BN schematic (v2)
├── Project.toml / Manifest.toml                # Julia environment
├── icpac_adm1v3.geojson                        # 227 admin-1 polygons
├── cmorph_ea_return_periods.nc                 # pixel-wise RP thresholds
└── *.md                                        # documentation files
```

### Result CSV schema (`flood_bn_v1_YYYY-MM-DD.csv`)

| Column | Description |
|--------|-------------|
| `boundary_id`, `boundary_name`, `country` | Admin-1 identifiers |
| `antecedent_category`, `rainfall_trend` | Hard-classified states for QC |
| `risk_level` | argmax of risk posterior (Minimal..Extreme) |
| `crma_state` | 4-state CRMA output (Monitor / Evaluate / Assess / Actionable_Risk) |
| `traffic_light` | Green / Yellow / Orange / Red |
| `crma_explanation` | Which cost-loss threshold fired, with numeric values |
| `recommended_action` | Legacy argmax action (Monitor..Act); retained for back-compat, not the primary signal |
| `confidence` | Peak action-probability mass (legacy) |
| `risk_minimal..risk_extreme` | Full 5-state risk posterior |
| `action_monitor..action_act` | Full 4-state legacy-action posterior |

---

## Results for 2026-03-01 to 2026-03-10

**Mar 1–10, 2026 (pixel-p95 + tail_risk + soft evidence + DBN,
γ = 0.20, α = 0.6)** over 2 270 boundary-days:

| CRMA state | Static BN | DBN (α=0.6) | Δ |
|------------|----------:|------------:|---:|
| Monitor          | 1 455 | 1 461 | +6 |
| Evaluate         |   223 |   259 | +36 |
| Assess           |   260 |   392 | +132 |
| Actionable_Risk  |   332 |   158 | −174 (−52 %) |

Soft evidence changed the CRMA state on 21 of 2 270 boundary-days
(14 promotions, 5 demotions, 1 escalation, 1 lateral shift); all
occurred within ~2σ of a discretisation bin edge.

**Nairobi case validation** — Nairobi River flash flood overnight
6–7 March 2026 (≥25 initial deaths, ≥108 across Kenya by month-end,
per [Wikipedia: 2026 Kenya floods](https://en.wikipedia.org/wiki/2026_Kenya_floods)):

| Date | Risk | CRMA state | Tail ratio | Notes |
|------|:----:|:----------:|:----------:|-------|
| Mar 1 | Moderate | Evaluate | 0.84 | Saturated ground (135 mm antecedent) |
| Mar 4 | **Moderate** | **Assess** | **1.11** | **2-day lead; tail crosses threshold** |
| Mar 6 | High | Actionable_Risk | 0.54 | Day of flood onset; DBN carries Mar 4 signal forward |
| Mar 7 | Moderate | Assess | 0.44 | Post-event dry-out; DBN smooths back down |

The worst-case ensemble-member storyline for Mar 6 projected 131 mm at
pixel scale vs. an ensemble mean of ~18 mm — a tail the mean-based
system would have reported as benign.

---

## Known limitations

1. **Short forecast lead**: init=D 00Z means each day's forecast only covers D→D+7. Multi-init fusion (pooling forecasts from D-3, D-2, D-1, D) could extend effective lead time.
2. **Scale mismatch**: ECMWF 0.25° (~28 km) averages over entire urban areas. CMORPH thresholds at 0.073° are point-scale. Convective rainfall is unpredictable at this scale — the ensemble spread captures it but the mean undersells; per-member storylines partially mitigate but do not eliminate the scale gap.
3. **Single-model forecasts**: only ECMWF available. Adding NOAA GEFS would activate the `forecast_agreement` node (currently retired) and provide multi-model robustness; re-enabling agreement takes the BN to a 6-parent CPT, which is beyond ReactiveMP's predefined-rule range (1-4 conditioning parents). It still runs via the generic structured-message rule with a per-inference slowdown, so the choice is performance vs. expressiveness, not feasibility — `infer_soft_matmul()` is the bulk-run workaround.
4. **Fixed expert CPTs**: the risk CPT encodes domain knowledge but has no empirical calibration. A future Bayesian-learning upgrade (Dirichlet priors on CPT columns, fed by historical flood events from EMDAT / FloodList / ICPAC incident DB) would refine these.
5. **Hazard likelihood only**: the `risk_level` node is a hazard-likelihood indicator, not a full IPCC AR6 WGII risk (= hazard × exposure × vulnerability). Joining population (WorldPop), vulnerability (INFORM), and critical infrastructure layers is a separate roadmap item.
6. **Independent boundaries**: no spatial smoothing between neighbors — a hotspot in Bungoma does not elevate the prior for adjacent Busia. Hierarchical/MRF extensions are possible in RxInfer.
7. **Admin-1 resolution**: operates at 227 admin-1 polygons; the EGU abstract and several partner workflows require admin-2 (~1 000 polygons), which needs re-benchmarking of the zonal-statistics stage.
8. **Heuristic soft-binning σ**: bandwidths are rule-of-thumb (25–30 % of narrowest bin); future work should derive them from physical measurement uncertainty.

See `flood_bn_ibf_system_v20260412.md` and
`probabilistic_logic_v20260413.md` for discussion of each upgrade path.

---

## Citation / provenance

- **Data**: ICPAC IBF team — IMERG from NASA GES DISC, ECMWF IFS ensemble, CMORPH v1.0 (NOAA CDR), admin boundaries from GADM/ICPAC.
- **Stores**: public anonymous read from [source.coop/e4drr-project](https://source.coop/e4drr-project).
- **Method**: Expert BN originally by ICPAC IBF team (Python/pgmpy); Julia/RxInfer port, pixel-level upgrades, soft-evidence / DBN / per-member storyline upgrades, and CRMA cost-loss output in this repo.
