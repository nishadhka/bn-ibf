# Current architecture (v1) — CDI as post-hoc likelihood update

This document tags the **currently-shipping** drought BN pipeline (the
one that just produced the 1981-2024 backfill of 528 CDI-updated CSVs)
so it's unambiguous what is implemented today vs what is proposed in
[`cdi_as_parent_v2.md`](./cdi_as_parent_v2.md). The v2 doc describes
the planned upgrade; **this doc describes what is on disk and on the
remote right now.**

Companion to [`cdi_bn_integration.md`](./cdi_bn_integration.md) — that
older doc proposed three placement options for CDI (Path A parallel /
Path B parent / Path C component-parents); the current implementation
is the **Path B-α/B-γ post-hoc variant**, not the joint-BN variant
originally favoured. This doc explains the variant that actually shipped
and why.

---

## 1. Pipeline at a glance

```
┌────────────────────┐    ┌─────────────────┐    ┌─────────────────────────┐
│ prep (Python)      │ -> │ bn (Julia warm) │ -> │ cdi (Python post-hoc)   │
│ SEAS5+ERA5 SPI3 →  │    │ 4-parent BN @   │    │ likelihood update with  │
│ 4 soft-evidence    │    │ model →         │    │ L[cdi, risk] matrix     │
│ vectors per init   │    │ risk_posterior  │    │ → revised posterior     │
└────────────────────┘    └─────────────────┘    └─────────────────────────┘
```

Three orchestrator stages in `run_drought_bn_backfill.py`:

| Stage | Driver | Output |
|---|---|---|
| `prep` | `drought_data_prep.py` (per init) | `bn_inputs_v2_backfill/drought_inputs_<YYYY-MM>_<season>.csv` |
| `bn`   | `drought_bn_ibf_v1_warm.jl` (one warm Julia loop over manifest) | `output_v2_notail_backfill/drought_bn_v2_notail_<YYYY-MM>_<season>.csv` |
| `cdi`  | `cdi_data_prep.py` + **`cdi_evidence_update.py`** (per init) | `output_v2_notail_cdi_backfill/drought_bn_v2_notail_cdi_<YYYY-MM>_<season>.csv` |

---

## 2. The 4 BN parents (CDI is NOT among them)

The Julia `@model` in `drought_bn_ibf_v1.jl` has exactly four parents of
`risk_level` in the v2 production build (the deprecated
`forecast_agreement` and `tail_risk` nodes can be enabled by flag but
are off by default for `drought_bn_v2_notail`):

| Parent | States | Driven by | Source data |
|---|---|---|---|
| `current_spi3` | 5 (Above_Normal … Severe_Drought) | observed CHIRPS/ERA5 SPI3 at init month | ERA5 |
| `forecast_deficit` | 5 (Very_Low … Very_High) | SEAS5 SPI3 ensemble vs ERA5 RP=5yr thresholds | SEAS5 + ERA5 |
| `spatial_coverage` | 3 (Localized / Moderate / Widespread) | fraction of polygon area with SPI3 < threshold | SEAS5 |
| `spi3_trend` | 3 (Improving / Stable / Deteriorating) | OLS slope of observed SPI3 over preceding 6 months | ERA5 |

Each parent receives a **soft-evidence vector** (`cur_p_*`, `def_p_*`,
`spa_p_*`, `trn_p_*`) computed in `drought_data_prep.py` and consumed
by the warm Julia loop. The CPT for `risk_level` is generated
programmatically by `compute_risk_probs()` in
`drought_bn_ibf_v1.jl:215-311` — see [`cdi_as_parent_v2.md` §1](./cdi_as_parent_v2.md)
for the parametric scoring rule.

**CDI does not appear anywhere in the Julia inference.** Its influence
enters in a separate Python step *after* the BN posterior is written.

---

## 3. How CDI enters: post-hoc likelihood update

`cdi_evidence_update.py` reads two CSVs per init —
the BN posterior and the CDI input — and applies a multiplicative
likelihood update on the risk probability vector:

```
posterior_post_cdi(risk)  ∝  posterior_pre_cdi(risk)  ·  L[cdi_level, risk]
```

where `L` is the 6×5 column-stochastic matrix `P(cdi | risk_level)`
hard-coded in `cdi_evidence_update.py:83-93`:

| | risk_Min | risk_Low | risk_Mod | risk_High | risk_Ext |
|---|---:|---:|---:|---:|---:|
| **No_drought** | 0.50 | 0.30 | 0.10 | 0.05 | 0.02 |
| **Full_recovery** | 0.25 | 0.25 | 0.15 | 0.05 | 0.03 |
| **Partial_recovery** | 0.10 | 0.20 | 0.20 | 0.10 | 0.05 |
| **Watch** | 0.10 | 0.15 | 0.25 | 0.20 | 0.15 |
| **Warning** | 0.04 | 0.08 | 0.20 | 0.30 | 0.25 |
| **Alert** | 0.01 | 0.02 | 0.10 | 0.30 | 0.50 |

(columns normalised to sum to 1)

The diagonal is the dominant trend — high risk should typically
co-occur with CDI Alert; low risk with No_drought — but the off-diagonal
mass encodes realistic measurement noise (forecast-leading boundaries
where BN says High but CDI obs is still Watch because drought hasn't
manifested yet, and the reverse case where CDI shows Alert but the
forecast saw recovery).

### 3a. Single-source vs two-source update (Path B-α vs B-γ)

The orchestrator's `--cdi-source both` flag triggers a wider CSV with
both an EADW operational reading and a recompute reading. When both
are present, `cdi_evidence_update.py` applies the likelihood twice:

```
single-source (Path B-α):
    posterior  ∝  prior  ·  L[cdi_level, :]

two-source (Path B-γ):
    posterior  ∝  prior  ·  L[cdi_level_recomp, :]  ·  L[cdi_level_eadw, :]
```

When the two channels agree, the product is more peaked (sharper
update); when they disagree, the product spreads mass across more
risk levels (softer update). No tuning knob between the regimes — the
L matrix's noise structure carries the weight.

---

## 4. Mathematical equivalence to "CDI as child observation"

This is the subtle point: the post-hoc multiplicative update is
**mathematically identical** to introducing CDI as a child observation
node of `risk_level` in the Julia BN with the same likelihood matrix
`L = P(cdi | risk_level)`:

```
parents (4)  ──>  risk_level  ──>  cdi_obs  (observed)
                                  ↑
                          P(cdi | risk) = L
```

The current implementation just performs the inference in **two passes**
(Julia BN for the prior over `risk_level`, then Python post-hoc for the
likelihood update) instead of one joint message-passing pass in
RxInfer. Bayes' rule doesn't care which language or which pass.

Why two passes instead of one:

- **Operational separation**: the Julia BN ships independent of the CDI
  pipeline. CDI source can change (MODIS vs operational vs recompute)
  without touching Julia code.
- **Diagnostic separability**: pre-CDI and post-CDI posteriors live in
  the same output CSV side-by-side; partners can see exactly what CDI
  contributed per boundary-init.
- **Late-binding data availability**: pre-1995 inits skip the post-hoc
  step entirely (no CDI exists); 2012-01 fAPAR-boundary inits gated by
  era-aware source selection. None of this branching needs to live in
  the Julia model.

What this costs (and v2 fixes):

- **Two-pass implementation invites bugs** that joint inference can't
  have — the 2012-01 `GDO_FPAR_OPERATIONAL_START` cutoff fix (commit
  7425587) is exactly this class.
- **Data-availability gating is in code** (`cdi_command_for()` in
  `run_drought_bn_backfill.py`, the pre-1995 symlink passthrough
  branch), not in the model. Each era boundary is a maintenance burden.
- **Source heterogeneity is a categorical choice** (MODIS/operational/
  recompute), not a soft uncertainty width. A noisier era contributes
  the same L matrix as a cleaner era.

These are the trade-offs `cdi_as_parent_v2.md` proposes to resolve by
folding CDI into the Julia model as a soft-evidence vector on a 5th
parent.

---

## 5. Output schema (v1 CSV)

The CDI-updated CSV (`drought_bn_v2_notail_cdi_<YYYY-MM>_<season>.csv`)
preserves **both** pre-CDI and post-CDI columns so the delta is
auditable:

| Column group | Pre-CDI (from BN) | Post-CDI (after update) |
|---|---|---|
| Risk posterior | `risk_minimal_pre_cdi …` | `risk_minimal …` |
| CRMA decision | `crma_state_pre_cdi` | `crma_state` |
| Traffic light | `traffic_light_pre_cdi` | `traffic_light` |
| CDI inputs | — | `cdi_level`, `cdi_class` (or `_recomp`/`_eadw` pair) |

This pre/post layout is the **defining diagnostic of v1**. Stakeholders
use it to explain to partners which boundary-months had their CRMA
state changed by CDI evidence and by how much.

Operational example from the 2024-12 init (last in backfill):

```
CRMA before CDI: Monitor 122 · Evaluate 3 · Assess 66 · Actionable 36
CRMA after  CDI: Monitor 145 · Evaluate 45 · Assess 20 · Actionable 17
96 CRMA flips · 55 % of boundaries had two CDI sources disagree
```

These delta diagnostics disappear under v2 unless explicitly preserved
via a parallel "CDI-off" shadow run.

---

## 6. Era-aware CDI sourcing (the data-availability ladder)

CDI inputs come from different sources depending on the init year. The
year-gating is in `run_drought_bn_backfill.py::cdi_command_for()`:

| Init year | `cdi_source` | `fapar_source` | Status |
|---|---|---|---|
| < 1995 | — | — | **Skip**: no CDI store. BN posterior copied through unchanged (symlink) |
| 1995-2000 | `recompute` | `none` | SPI + SMA only (no vegetation anomaly) |
| 2001-2011 | `recompute` | `auto` (→ MODIS) | + GDO-MODIS fAPAR backfill |
| 2012-01 | `both` | `auto` (→ MODIS) | MODIS-only (operational store starts 2012-01-21; see commit 7425587) |
| 2012-02 + | `both` | `auto` (→ operational) | GDO MERIS+OLCI operational fAPAR + EADW |

Each era boundary corresponds to a measurable change in the input data
quality. The v1 architecture handles this via **conditional branching
in code** (year comparisons in `cdi_command_for`, era-aware
`_open_fapar_for` in `cdi_data_prep.py`). v2 proposes folding the same
information into a soft-evidence σ width on the BN's CDI parent —
no code branches.

---

## 7. Backfill artifacts produced

The 1981-2024 backfill (528 inits, ~13 h wall) shipped these
artifacts on disk under `drought_ibf/`:

```
bn_inputs_v2_backfill/         528 CSVs · per-init prep soft-evidence
output_v2_notail_backfill/     528 CSVs · per-init BN-only posteriors
                               + _manifest.csv (Julia warm-loop input)
output_v2_notail_cdi_backfill/ 528 CSVs · per-init CDI-updated posteriors
cdi_inputs_backfill/           360 CSVs · per-init CDI input vectors
                               (168 pre-1995 inits skip CDI entirely)
drought_bn_ibf_monthly.parquet               regional roll-up
drought_bn_ibf_boundary_monthly.parquet      per-boundary roll-up
output_v2_notail_cdi_backfill/bn-dag/        per-init DAG JSONs for the web frontend
```

All produced by the v1 architecture described above.

---

## 8. When to use v1 (this) vs v2 (proposed in `cdi_as_parent_v2.md`)

- **v1** (this) — production, partner-facing CRMA outputs, anything
  where the pre-/post-CDI diagnostic separation matters. Operationally
  stable. Bug-class: era-boundary cutoffs need maintenance.
- **v2** (proposed) — research/upgrade path. Single coherent posterior,
  no era-boundary code branches, source heterogeneity modelled as σ
  widths rather than categorical store selection. Requires a
  ~10-line Julia patch + recalibration of one weight in
  `compute_risk_probs` + a 4-step validation pass. See
  `cdi_as_parent_v2.md` for the implementation footprint.

Neither version replaces the other for free: v1 has a year of partner
validation behind its CRMA outputs and a clean diagnostic separation;
v2 has architectural cleanliness and bug-class elimination. The
recommended migration is to **pilot v2 on a parallel branch using the
1995-2010 historic period** where v1 CRMA is most trusted, treat v1 as
the ground truth, and only retire v1 once joint inference reproduces
the v1 CRMA states within tolerance on a held-out year.

---

## 9. File map

| File | Role | v1 | v2 (proposed) |
|---|---|:---:|:---:|
| `drought_data_prep.py` | SEAS5+ERA5 prep CSV | ✓ | ✓ (extended) |
| `cdi_data_prep.py` | CDI input CSV per init | ✓ | refactored to emit soft-vector |
| `drought_bn_ibf_v1.jl` | 4-parent BN @model + CPT rule | ✓ | extended to 5-parent |
| `drought_bn_ibf_v1_warm.jl` | warm-Julia driver | ✓ | reads extra CDI column |
| `cdi_evidence_update.py` | post-hoc likelihood update | ✓ | **deleted** (or kept as "CDI-off shadow") |
| `run_drought_bn_backfill.py` | 3-stage orchestrator | ✓ | collapsed to 2 stages |
| `generate_drought_bn_parquet.py` | parquet roll-up | ✓ | unchanged |
| `generate_drought_bn_dag_json.py` | per-init DAG JSON | ✓ | unchanged |

The post-step parquet and DAG JSON generators are version-agnostic —
they consume whatever CSV the upstream stages produced and don't care
whether CDI entered via post-hoc update or BN evidence.
