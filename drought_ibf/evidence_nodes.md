# Drought BN — Evidence Nodes, Data Sources, Adaptive Behaviour

This document is the canonical reference for **what each evidence node in the
operational drought BN is, where its values come from, and how the pipeline
behaves when one or more sources is unavailable for a given month**. It is
the spec the rest of `drought_ibf/` (and the `crma-api` consumers) follow.

It complements:
- [`bn_interpretation.md`](bn_interpretation.md) — how the BN
  *computes* a posterior given a complete evidence vector for one boundary.
- [`cdi_bn_integration.md`](cdi_bn_integration.md) — design rationale for
  applying CDI as a virtual-evidence layer (Path B-γ) on top of the BN.

This document focuses on **data availability** and the **adaptive evidence
strategy**: which months can use which evidence channel, what happens when
something is missing, and what the planned 2005-2026 historical run looks
like under the current and proposed source set.

---

## 1. The pipeline at a glance

```
                                             ┌────────────────────────────┐
ERA5 SPI (zarr, 1940–now) ──┐                │ drought_bn_ibf_v1.jl       │
                            ├──► drought_data_prep.py ──► drought_inputs_<init>.csv ──►│  RxInfer.jl 4-parent BN    │──► drought_bn_v2_notail_<init>.csv
SEAS5 SPI3 (icechunk,       │                                                │ (cur, def, spa, trn → risk │
    1981–now, 25-mem        │                                                │  → action)                 │
    hindcast block)         ┘                                                └─────────────┬──────────────┘
                                                                                           │
                                                                                           ▼
CHIRPS SPI / GDO SMA / GDO fAPAR (icechunk, ≥1991/1995/2012) ┐                  ┌─────────────────────────────┐
ICPAC EADW CDI (icechunk, 2010–now)                          ├─► cdi_data_prep.py ──► cdi_inputs_<init>.csv ──►│ cdi_evidence_update.py      │──► drought_bn_v2_notail_cdi_<init>.csv
GDO fAPAR-MODIS (icechunk, 2001–2015, see §6)                ┘                  │ Path B-γ multiplicative      │
                                                                                │ joint likelihood (virtual    │
                                                                                │ evidence)                    │
                                                                                └──────────────────────────────┘
```

The BN side (top row) and the CDI side (bottom row) are decoupled. **Either
one can run alone, both can run, or only the BN can run**. Which combination
is feasible for a given month is decided by data availability — see §7 and
§8 for the matrix.

---

## 2. Evidence-type taxonomy

Probabilistic evidence on a Bayesian network can take three forms. The
drought BN uses all three:

| Term | Meaning | Where in the drought BN |
|---|---|---|
| **Hard evidence** | One state is observed with certainty: $P(X=k) = 1$ for one $k$ | Not used in v2 (v1 used hard evidence on `current_spi3_category`; v2 superseded by soft) |
| **Soft evidence (likelihood)** | A probability distribution over states: $P(X=k) = p_k$ for $k=1..K$ with $\sum p_k = 1$ | All four parent nodes in v2: `cur`, `def`, `spa`, `trn` (and `tail` in 5-parent variant). Computed by Gaussian-CDF binning around the observed continuous value (`drought_data_prep.py:soft_bin`) |
| **Virtual evidence** (Pearl's noisy-channel evidence) | An auxiliary variable $L$ with a known likelihood-conditional CPT $P(L \| X)$, applied through the Bayes update $P(X \| L) \propto P(X) \cdot P(L \| X)$ | The CDI evidence channel (Path B-γ): the observed CDI class is treated as a noisy observation of the *latent* `risk_level`, with a 14×5 likelihood matrix $L$ that encodes which CDI classes are evidence for which risk levels and how noisy that channel is |

The distinction matters operationally: **hard and soft evidence are inputs
to the BN's forward inference**; **virtual evidence is applied *after* the
BN has computed its posterior**, by multiplying the posterior elementwise
with the relevant column of $L$ and renormalising.

---

## 3. Per-node specification

### Parent nodes — soft evidence consumed by the BN

| Node | What it is | Source | Soft prob columns | Time coverage | Used in current run |
|---|---|---|---|---|---|
| `cur` (current SPI-3) | Observed SPI-3 ending at the BN init month | ERA5 SPI zarr (`era5_ecmwf_pencil`) | `cur_p1..cur_p5` (5 states: Severe / Moderate / Mild Drought / Normal / Above_Normal — column order REVERSED relative to STATES order, see `_REVERSE_NODES` in `drought_data_prep.py`) | **1940-01 → 2026-01** | ✓ |
| `def` (forecast deficit prob) | **Empirical probability across the 25 SEAS5 ensemble members of crossing the per-pixel ERA5-fitted SPI-3 return-period threshold** at the row's target-season lead, then zonally averaged over the admin-1 polygon. v2 uses the RP threshold; v1 used a scalar `--deficit-spi=-1.0` instead | SEAS5 SPI-3 icechunk (`seas51_spi3_10km_icechunk_v2`, first 25 members for full 1981-now hindcast coverage) **AND** ERA5 SPI return-period icechunk (`era5_ecmwf_rp_icechunk`, 5-yr fitted threshold by default; selectable via `--rp-years`) | `def_p1..def_p5` (5 states: Very_Low / Low / Medium / High / Very_High); raw value in `forecast_deficit_prob`; threshold lineage recorded in `deficit_threshold_source` (e.g. `era5_ecmwf_rp_icechunk:5yr fitted`) | **1981-01 → 2026-04** | ✓ |
| `spa` (spatial coverage) | Fraction of the admin-1 polygon's pixels where the RP exceedance is widespread. Combines two sub-metrics over the same SEAS5-vs-RP boolean field as `def`: `spatial_coverage` = fraction of pixels where the majority of members crosses RP (`p_def_l1 > 0.5`); `hotspot_fraction` = fraction of pixels where **any** member crosses RP (`crosses_rp_any`) | SEAS5 SPI-3 + ERA5 RP icechunk (zonal reductions of the same exceedance field used by `def`) | `spa_p1..spa_p3` (3 states: Localized / Moderate / Widespread); raw values in `spatial_coverage` and `hotspot_fraction` | 1981-01 → 2026-04 | ✓ |
| `trn` (SPI-3 trend) | Slope of the monthly observed SPI-3 series over the last `--trend-months` (default 6) months ending at init | ERA5 SPI zarr | `trn_p1..trn_p3` (3 states: Deteriorating / Stable / Improving — REVERSED column order) | 1940-01 → 2026-01 | ✓ |
| `tail` (worst-case ens-min SPI) | The 5th-percentile SPI-3 across SEAS5 members at the target lead — the "bad ensemble member" tail of the same exceedance distribution that `def` and `spa` summarise | SEAS5 SPI-3 icechunk; binned with reference to the same ERA5 RP threshold scale as `def`/`spa` | `tail_p1..tail_p4` (4 states: High / Moderate / Low / Nil — REVERSED column order); raw value in `ens_min_spi_peak` | 1981-01 → 2026-04 | **dropped in v2_notail** (was in v1 5-parent variant; removed because it was driving 84% of admin-months to Actionable_Risk) |

All five parent columns are written to the per-month soft CSV
(`drought_inputs_<init>_<season>.csv`) regardless of which BN flavour the
Julia inference will consume — `tail_*` columns are simply ignored by
`drought_bn_ibf_v1.jl` when it runs in 4-parent mode.

### Empirical RP-exceedance is the backbone of `def` + `spa` + `tail`

A common question: "Is there a return-period threshold exceedance step
between SEAS5 and `era5_ecmwf_rp_icechunk` that surfaces as an evidence
node?" — **yes, and it drives three of the five parent nodes**, not one.

The shared lineage:

```
era5_ecmwf_rp_icechunk[spi_period=SPI3, return_period=5yr, fitted]
       │   per-pixel SPI threshold on the ERA5 climatology grid
       ▼
SEAS5 SPI-3 forecast (member, lat, lon) at target-season lead
       │   per-pixel boolean: is this member's SPI ≤ that pixel's RP threshold?
       ▼
deficit_lead = (fc_target ≤ rp_thresh)                      # (member, lat, lon)
       │
       ├──► p_def_l1 = deficit_lead.mean(dim="member")       # empirical exceedance prob, per pixel
       │       │
       │       ├──► zonal_mean(p_def_l1, admin1 mask)        ──► forecast_deficit_prob ──► def_p1..def_p5
       │       └──► zonal_mean(p_def_l1 > 0.5, admin1 mask)  ──► spatial_coverage      ──► spa_p1..spa_p3 (driver A)
       │
       ├──► crosses_rp_any = deficit_lead.any(dim="member")  # any-member exceedance, per pixel
       │       └──► zonal_mean(crosses_rp_any > 0.5)         ──► hotspot_fraction      ──► spa_p1..spa_p3 (driver B)
       │
       └──► ens_min_anylead = fc_target.min(dim="member")    # worst-member SPI tail, per pixel
               └──► zonal_quantile(q=0.05, admin1 mask)      ──► ens_min_spi_peak      ──► tail_p1..tail_p4 (5-parent only)
```

Three different *summaries* of the same per-pixel exceedance field:

| Summary | Captures | Surfaces in |
|---|---|---|
| Mean over members, mean over pixels | "How likely is a typical member to cross RP somewhere typical in the polygon" | `def` |
| Fraction of pixels with majority-members crossing | "How widespread is the exceedance" | `spa` (`spatial_coverage`) |
| Fraction of pixels with any-member crossing | "How big is the hotspot envelope" | `spa` (`hotspot_fraction`) |
| 5th-percentile worst member | "How bad is the bad member" | `tail` (5-parent only) |

The threshold-source string in every output CSV is
`deficit_threshold_source = "era5_ecmwf_rp_icechunk:Nyr fitted"` (where N
is `--rp-years`, default 5). v1 used a scalar `--deficit-spi=-1.0` instead
(McKee moderate-drought) — that path is preserved for backwards
compatibility but the **production v2 path is the per-pixel RP threshold**.

So there is no separate "exceedance" node in the BN graph because the
exceedance computation is already the *substrate* of three of the five
parents — bringing it back as a fourth parent would double-count the same
SEAS5-vs-RP signal three times. What the `def` / `spa` / `tail` split
buys you is **independent summary statistics of that exceedance field**:
location-mean, spatial-coverage, and worst-member tail, each entering the
risk CPT through its own conditional dependency.

### Latent / output nodes

| Node | States | Computed by | Notes |
|---|---|---|---|
| `risk_level` | Minimal / Low / Moderate / High / Extreme (5 states) | RxInfer.jl marginal over the explicit CPT (`drought_bn_ibf_v1.jl::compute_risk_probs`) | Posterior over 5 risk states given the parent soft evidence |
| `action` | Monitor / Alert / Prepare / Act (4 states) | Cost-loss decision rule (γ = 0.20) over `risk_level` posterior | Translated to CRMA traffic-light states (Monitor / Evaluate / Assess / Actionable_Risk) |
| `cdi_class` (virtual evidence) | 14-class JRC CDI: No_drought (0) plus Watch (1–3) / Warning (4–6) / Alert (7–10) / Partial_recovery (11–12) / Full_recovery (13–14) | `cdi_data_prep.py` (recompute) or read directly from `icpac_cdi_dekadal_icechunk` (eadw) | NOT a BN parent — applied as virtual evidence in `cdi_evidence_update.py` |

---

## 4. The CDI virtual-evidence channel (Path B-γ)

The CDI is *not* a parent node in the BN. Instead, the BN's 5-state
`risk_level` posterior is updated by multiplying with a likelihood vector
derived from the observed CDI class:

```
P_post(risk = r | CDI = c) ∝ P_BN(risk = r) · L[c, r]
```

where `L` is a 14×5 likelihood matrix (`cdi_evidence_update.py::LIKELIHOOD`)
that encodes "if CDI = Alert (class 7-10), the latent risk is more likely to
be High or Extreme; if CDI = No_drought (0), the latent risk is more likely
to be Minimal". The matrix is calibrated so that the channel is *noisy*:
even an Alert CDI does not put 100% mass on Extreme.

### Two CDI source paths

The drought CDI itself is computed from two independent sources:

| Path | Components | Implementation | Years |
|---|---|---|---|
| **`recompute`** | CHIRPS SPI-1, SPI-3, SPI-9/12 + GDO SMA + GDO fAPAR → JRC 14-class rule cascade (`calculate_cdi_grid`) | `cdi_data_prep.py --cdi-source recompute` | **2012–now** (limited by GDO fAPAR start; CHIRPS goes back to 1991, SMA to 1995) |
| **`eadw`** | ICPAC EADW pre-computed CDI dekadal (operational) | `cdi_data_prep.py --cdi-source eadw` | **2010–now** |

When both paths are available (`--cdi-source both`), the per-boundary
likelihoods are combined multiplicatively:

```
L_joint[r] = L_recompute[c_r, r] · L_eadw[c_e, r]
```

Disagreement between sources naturally inflates uncertainty: if recompute
says Alert (high mass on High/Extreme) and EADW says Watch (high mass on
Low/Moderate), the joint multiplication spreads probability across more
risk levels rather than collapsing to one. The `cdi_agreement` boolean in
the post-CDI CSVs flags the agreement state for tooltip rendering.

### Graceful degradation in the recompute rule cascade

`calculate_cdi()` is structured so that **missing components don't break the
rule** — they just lower the maximum severity attainable that month:

| Available components | Highest reachable CDI class | Rule branch |
|---|---|---|
| SPI + SMA + fAPAR | Class 7–10 (Alert) | full cascade |
| SPI + SMA, no fAPAR | Class 4–6 (Warning) | falls through Alert branches |
| SPI only | Class 1–3 (Watch) | falls through Warning + Alert |
| Previous SPI only | Class 11–14 (Recovery) | recovery branches |
| None | Class 0 (No_drought) | conservatively unclassified |

For example, in 2008-06 (CHIRPS ✓, SMA ✓, fAPAR ✗) the same boundary that
would have been classified as `Alert(7)` with all three components instead
gets `Warning(4)` from the SPI+SMA branch. The associated likelihood vector
`L[Warning(4), :]` is broader and gentler than `L[Alert(7), :]`, so the BN
posterior is updated less aggressively — which is **the right behaviour**:
without fAPAR we genuinely have less evidence to push the posterior with.

---

## 5. Adaptive evidence strategy under data availability

The full pipeline state machine for any single (init-month, target-season)
pair:

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    Is ERA5 SPI obs available?                            │
│                              ↓ no                                        │
│             [skip month entirely — cur / trn cannot be computed]         │
│                              ↓ yes                                       │
│                    Is SEAS5 init available?                              │
│                              ↓ no                                        │
│             [skip month entirely — def / spa / tail cannot be computed]  │
│                              ↓ yes                                       │
│                  ▶ Run drought_data_prep.py (4-parent soft evidence)     │
│                  ▶ Run drought_bn_ibf_v1.jl  (BN posterior)              │
│                              ↓                                           │
│                    Is any CDI source available?                          │
│                              ↓ no                                        │
│             [emit pre-CDI posterior as the final result]                 │
│                              ↓ yes                                       │
│                    Are both EADW and recompute available?                │
│                          ↓ yes        ↓ EADW only      ↓ recompute only  │
│                  cdi-source=both    cdi-source=eadw   cdi-source=recompute│
│                              ↓                                           │
│                  ▶ Run cdi_data_prep.py                                  │
│                  ▶ Run cdi_evidence_update.py (virtual evidence)         │
│                              ↓                                           │
│                  Within recompute, fAPAR / SMA may also be missing       │
│                  → graceful degradation in calculate_cdi() rule cascade  │
└──────────────────────────────────────────────────────────────────────────┘
```

The BN itself is **always 4-parent (or 5-parent), always soft**. The
adaptiveness lives entirely in:

1. Whether the CDI step runs at all (yes if any source exists, no if neither).
2. Within the CDI step, which CDI source path is used (`recompute`, `eadw`,
   or `both`).
3. Within the recompute path, which JRC rule branch fires (controlled by
   which of SPI / SMA / fAPAR are populated for the month).

No code path tries to "fake" missing evidence with a uniform distribution
on a parent node — this would silently bias the BN posterior toward the
prior. Instead, missing observation/forecast → skip the month; missing
CDI → fall back to BN-only.

---

## 6. Extending CDI back to 2001 with MODIS fAPAR

The current operational `recompute` floor is **2012-01**, set by the GDO
fAPAR (MERIS+OLCI) start date. CHIRPS (1991) and SMA (1995) both reach
much further back, so for the 1995-2011 window CDI-recompute would only
have its precipitation and soil-moisture branches — meaning it can reach
at most CDI class 6 (Warning), not class 10 (Alert).

The script
[`ibf-thresholds-triggers/thresholds/hf-gdo/gdo_fpar_modis_icechunk.py`](../../ibf-thresholds-triggers/thresholds/hf-gdo/gdo_fpar_modis_icechunk.py)
ingests the **GDO fAPAR-MODIS** product (MOD15A2H back-derived anomalies)
into a parallel icechunk store covering **2001-01 → 2015-12**. Combining
this with the operational GDO fAPAR (2012-01 → now) gives **continuous
fAPAR-anomaly coverage from 2001-01 onwards**, with a 2012–2015 overlap
window for cross-validation.

### What changed in `cdi_data_prep.py` (commit referenced below)

The script now ships a year-aware fAPAR opener (`_open_fapar_for`) and a
new CLI flag `--fapar-source {auto,gdo,modis,none}` (default `auto`).
Routing logic:

| target | `auto` mode | resulting `fapar_source` label in CSV |
|---|---|---|
| < 2001-01 | None — graceful degradation to SPI+SMA-only CDI | `none` |
| 2001-01 — 2011-12 | GDO fAPAR-MODIS backfill (`gdo_fpar_modis_icechunk`) | `modis` |
| 2012-01 — now | GDO fAPAR operational (`gdo_fpar_icechunk`) | `gdo` |

```python
# drought_ibf/cdi_data_prep.py — added near the existing prefix constants
GDO_FPAR_PREFIX            = "e4drr-project/observations/gdo_fpar_icechunk"
GDO_FPAR_MODIS_PREFIX      = "e4drr-project/observations/gdo_fpar_modis_icechunk"
GDO_FPAR_OPERATIONAL_START = pd.Timestamp("2012-01-01")
GDO_FPAR_MODIS_START       = pd.Timestamp("2001-01-01")
GDO_FPAR_MODIS_END         = pd.Timestamp("2015-12-31")

def _open_fapar_for(target, mode):
    """Pick the fAPAR icechunk store appropriate for `target` and `mode`.
    Returns (fapar_dataset_or_None, source_label)."""
    # full implementation in cdi_data_prep.py
```

When the helper returns `(None, "none")` (target predates 2001-01 or the
user passed `--fapar-source none`), `_aggregate_recompute` builds an
all-False fAPAR mask on the CHIRPS grid. That makes
`calculate_cdi_grid()`'s `fapar_lt_m1` boolean uniformly False, so every
Alert-class branch (which requires `fapar_lt_m1`) fails and the rule
cascade naturally falls through to Warning / Watch / Recovery branches —
exactly the graceful-degradation behaviour described in §4.

Both stores (operational GDO + MODIS backfill) expose the same `fpanv`
fAPAR-anomaly variable on the same EA grid, so the regridder onto the
CHIRPS grid + the `lt_m1` thresholding is unchanged.

The chosen source is recorded per-row in the output CSV's `fapar_source`
column so the downstream BN run can audit which CDI lineage produced
each post-CDI posterior.

---

## 7. Coverage matrix 1981-01 → 2026-04

| Period | Months | ERA5 obs | SEAS5 fcst | CHIRPS SPI | GDO SMA | GDO fAPAR (operational) | GDO fAPAR (MODIS, planned) | EADW CDI | What can run today | What can run with MODIS extension |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1981-01 — 1990-12 | 120 | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | BN-only | BN-only |
| 1991-01 — 1994-12 | 48 | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | BN-only | BN-only |
| 1995-01 — 2000-12 | 72 | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | BN-only (would be Watch-class CDI in `recompute`, currently disabled until §6 is wired) | BN + CDI (Watch+Warning, no Alert) |
| 2001-01 — 2009-12 | 108 | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✗ | BN-only | **BN + CDI (recompute)** ← new with §6 |
| 2010-01 — 2011-12 | 24 | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | BN + CDI (eadw only) | **BN + CDI (both)** ← new with §6 |
| 2012-01 — 2015-12 | 48 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ (overlap) | ✓ | BN + CDI (both) | BN + CDI (both, MODIS as cross-check) |
| 2016-01 — 2026-04 | 124 | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | BN + CDI (both) | BN + CDI (both) |

**Take-aways**

- The BN itself runs unchanged across the full 1981–now window — every
  parent node has continuous coverage from SEAS5 (1981) and ERA5 (1940).
- Without any extension, only **2010-01 onwards** gets a CDI update, and
  only **2012-01 onwards** gets the *full* JRC 14-class CDI.
- Wiring the MODIS fAPAR source into `cdi_data_prep.py` (§6) extends
  full-CDI coverage back to **2001-01**, and gives a usable Watch+Warning
  CDI back to **1995-01** if you also accept the SMA-only path for
  pre-2001 months.

---

## 8. Reflection on the planned 2005 → 2026 run

Targeting 2005-01 → 2026-04 = **256 init-months** (one (init, season) pair
per init-month per the `SEASON_INIT_LEAD` table). Under the planned
MODIS-fAPAR extension:

| Sub-period | Months | Mode | CDI source | Notes |
|---|---:|---|---|---|
| 2005-01 — 2009-12 | 60 | BN + CDI | recompute (with MODIS fAPAR) | Full 14-class CDI; no EADW available |
| 2010-01 — 2011-12 | 24 | BN + CDI | both | Both paths available; multiplicative joint update |
| 2012-01 — 2015-12 | 48 | BN + CDI | both | Operational fAPAR + MODIS overlap (cross-check window) |
| 2016-01 — 2026-04 | 124 | BN + CDI | both | Operational fAPAR only; same as production today |
| **Total** | **256** | | | |

**Without** the MODIS-fAPAR extension, the same 256 months would split as:

| Sub-period | Months | Mode | CDI source |
|---|---:|---|---|
| 2005-01 — 2009-12 | 60 | BN-only | (no CDI for these months) |
| 2010-01 — 2011-12 | 24 | BN + CDI | eadw only |
| 2012-01 — 2026-04 | 172 | BN + CDI | both |

The MODIS extension is therefore worth ≈ 20% of the run window in terms of
months that get a CDI update (60 months out of 256), and is the **only**
path to bring the post-CDI posterior coverage continuously back to 2005
without changing the BN itself.

### Compute estimate for the 256-month run

From the 2026-05-01 rerun notes (`bn_ibf_rerun_2026-05-01.md`):

| Stage | Per-init wall | × 256 inits | With warm-Julia loop |
|---|---:|---:|---:|
| `drought_data_prep.py` | ~70 s | ~5 h | (network-bound, no warm-up gain) |
| `drought_bn_ibf_v1.jl` (cold per-init) | ~95 s | ~6.8 h | (~1 s/call if Julia kept warm) |
| `cdi_data_prep.py` (recompute) | ~50 s | ~3.6 h | |
| `cdi_evidence_update.py` | ~5 s | ~20 min | |
| `generate_drought_bn_parquet.py` + `…_dag_json.py` | ~5 s | once at the end | |

Sequential cold ≈ **~16 h**; warm-Julia loop ≈ **~9 h**; Coiled / Lithops
parallelised ≈ **~30-60 min** end-to-end.

### Smallest set of changes to commission the 1981 → 2024 backfill

The full 1981-01 → 2024-12 sweep is 528 (init, season) pairs. The smallest
set of changes — building on the patches landed in this revision — is:

1. **Loop driver** — `run_drought_bn_backfill.sh` (or `.py`) iterating the
   528 (init, season) pairs:
   - calls `drought_data_prep.py --init-month <YYYY-MM> --target-season <S>`
     for each pair, writing into `bn_inputs_v2_backfill/`
   - feeds CSVs to a **single warm Julia process** running
     `drought_bn_ibf_v1.jl` (cuts the per-init cost from ~95 s of JIT
     startup down to <1 s of inference per call)
   - writes BN posteriors into `output_v2_notail_backfill/` for the BN-only
     posterior and `output_v2_notail_cdi_backfill/` for the post-CDI posterior

2. **CDI gating by year** — drive `cdi_data_prep.py` + `cdi_evidence_update.py`
   only for inits where the relevant CDI source is available. With the
   `--fapar-source=auto` patch landed in this revision, the gating
   becomes:

   | init range | CDI command |
   |---|---|
   | 1981-01 — 1994-12 | (none — pre-CDI posterior is the final output) |
   | 1995-01 — 2000-12 | `cdi_data_prep.py --cdi-source recompute --fapar-source none` (Watch+Warning only, no fAPAR available) |
   | 2001-01 — 2009-12 | `cdi_data_prep.py --cdi-source recompute --fapar-source auto` (auto resolves to MODIS) |
   | 2010-01 — 2011-12 | `cdi_data_prep.py --cdi-source both --fapar-source auto` (MODIS recompute + EADW operational) |
   | 2012-01 — 2024-12 | `cdi_data_prep.py --cdi-source both --fapar-source auto` (GDO recompute + EADW operational) |

   The pre-2001 SPI+SMA-only branch is what was previously called the
   "alternative — no script change" fallback in §6; with the
   `--fapar-source none` flag it's now a first-class operating mode rather
   than an accidental side effect of running the recompute path on a
   month with no fAPAR.

3. **Single-pass artifact generation** — after the 528-month sweep
   completes, run `generate_drought_bn_parquet.py` and
   `generate_drought_bn_dag_json.py` once over the full backfill output
   dir to produce a single 528-row monthly parquet and 528 DAG JSONs.

4. **Upload** via `upload_bn_artifacts.py` (the routine end-of-run helper
   in the repo root). With `--skip-unchanged` it will only upload the new
   480-odd objects (since the existing 16-month `output_v2_notail_cdi/`
   subset is already in GCS unchanged).

### Output deliverables

The same upload pipeline used today (`upload_bn_artifacts.py`) handles the
backfill output as long as it lands in the conventional dirs:

```
drought_ibf/
    output_v2_notail_cdi/                       (post-CDI, primary deliverable)
        drought_bn_v2_notail_cdi_<init>_<season>.csv          × 256
        bn-dag/
            drought-bn-dag-<init>.json                        × 256
        drought_bn_ibf_monthly.parquet                        (regenerated)
        drought_bn_ibf_boundary_monthly.parquet               (regenerated)
    output_v2_notail/                           (pre-CDI, debug)
        drought_bn_v2_notail_<init>_<season>.csv              × 256
```

After the run completes:

```
uv run python3 generate_drought_bn_parquet.py
uv run python3 generate_drought_bn_dag_json.py
GOOGLE_APPLICATION_CREDENTIALS=… uv run python3 upload_bn_artifacts.py --skip-unchanged
```

— and the existing `crma-api` Cloud Run instance picks up the new 256-row
calendar parquet + 256 DAG JSONs at the next request, no rebuild required.

---

## 9. Summary

- The BN has **5 nodes consuming evidence**: 4 parents in production
  (`cur`, `def`, `spa`, `trn`) plus an optional `tail` parent in the
  5-parent v1. All consume **soft evidence** (probability vectors over
  discrete states), computed by Gaussian-CDF binning around the underlying
  continuous observation/forecast.
- The **CDI** is not a BN parent — it is a **virtual-evidence** layer
  applied multiplicatively to the BN's 5-state risk posterior, with the
  observed CDI class indexing into a 14×5 noisy-channel likelihood matrix.
- The CDI itself is computed from **two independent source paths**
  (recompute and eadw), and within the recompute path the JRC 14-class
  rule cascade **gracefully degrades** when components are missing.
- The BN can run **unchanged for 1981–now** because all four (or five)
  parent nodes have continuous coverage from ERA5 + SEAS5.
- The CDI update currently only covers **2010-01 → now** (eadw) or
  **2012-01 → now** (recompute). Wiring the MODIS-fAPAR icechunk store
  produced by `gdo_fpar_modis_icechunk.py` extends full-CDI coverage back
  to **2001-01**.
- The planned 2005-2026 backfill (256 init-months) gets full BN + CDI
  coverage with the MODIS extension; without it, 60 months (2005-2009) are
  BN-only.
