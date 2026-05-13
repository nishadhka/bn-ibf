# CDI as a 5th BN parent — implementation note (v2)

Companion to `cdi_bn_integration.md` (which laid out the original design)
and `evidence_nodes.md` (which inventories the current 4-parent BN). This
doc clears up one specific question that came up while running the
1981-2024 backfill and the 2012-01 fAPAR-cutoff bug:

> Is CDI-as-parent expensive because we'd need a 5-parent CPT? Do we have
> to choose between full elicitation, noisy-MAX/OR decomposition, and
> empirical learning?

**Short answer: no.** The existing drought (and flood) BN does not store
a hand-elicited tabular CPT. It generates the CPT *programmatically* from
a compact parametric scoring rule. Adding a 5th parent is roughly a
ten-line code change, not a workshop exercise.

---

## 1. What the current CPT actually is

`drought_bn_ibf_v1.jl::compute_risk_probs(current, deficit, spatial, trend, agreement, tail)`
is a deterministic function from parent state tuple → 5-state risk
probability vector. The function has three parts:

1. **Additive base score** — weighted sum of parent state indices

   ```julia
   base_risk = c * 0.30 + d * 0.55
   base_risk += {0.0, 0.25, 0.5}[spatial]
   base_risk += {-0.30, 0.0, 0.35}[trend]
   base_risk += {0.0, 0.10, 0.35, 0.60}[tail]
   ```

2. **Expert-override rules** — eight named scenario rules (Rule 1-5, T1-T3)
   that short-circuit the binning for archetypes that would otherwise be
   mis-scored by the linear base (e.g. low-mean-deficit but high-tail is
   a real failure mode of mean-based scoring).

3. **Binning + agreement blend** — `base_risk` falls into one of five
   bins (`< 1`, `< 2`, `< 3`, `< 4`, else), each producing a fixed
   probability vector; low forecast agreement blends with a uniform.

`build_risk_cpt` then enumerates all `5 × 5 × 3 × 3 × 3 × 4 = 2 700`
parent state tuples, calls `compute_risk_probs` on each, and stacks the
results into the tensor RxInfer's `DiscreteTransition` consumes. The CPT
is *materialised* at build time but *parameterised* by the scoring rule.

This is closer to **noisy-additive with rule overrides** than to either
noisy-MAX or a hand-elicited table. The structural decomposition the
literature suggests for 5-parent CPTs is already implicit in the code.

---

## 2. Implication for CDI

If CDI joins as a 5th parent (or 6th, if `forecast_agreement` is kept),
the new state space is

```
current(5) × deficit(5) × spatial(3) × trend(3) × tail(4) × cdi(5)  =  4 500 cells
```

In a hand-elicited regime this is intimidating. With the parametric
rule, the only new pieces are:

- **One new weight** in the additive base — `cdi_weight * (cdi_state - 1)`
- **One or two interaction rules** for edges where CDI carries
  information the other parents miss (sketch below)
- **One extra loop dimension** in `build_risk_cpt`

The 4 500 cells get generated, not authored.

### 2a. CDI state space

Five drought-state bins matching the antecedent semantics already used
elsewhere in the model:

| CDI level (from `cdi_evidence_update`) | BN `cdi_state` index | Maps to drought-state bin |
|---|---:|---|
| No drought, Full recovery | 1 | Normal |
| Partial recovery, Watch | 2 | Mild deficit |
| Warning (SPI + SMA) | 3 | Moderate deficit |
| Alert (any) | 4 | Severe deficit |
| Alert class 10 (SPI + SMA + fAPAR all firing) | 5 | Extreme deficit |

This mirrors the `current_spi3` 5-state space (Above_Normal …
Severe_Drought), which makes the rule additions cheap because CDI
and current_spi3 share semantics.

### 2b. Code-level patch

```julia
# 1. add parameter
function compute_risk_probs(
    current::Int, deficit::Int, spatial::Int, trend::Int,
    agreement::Int=3, tail::Int=1,
    cdi::Int=1,                              # NEW — default = "no info"
)::Vector{Float64}
    ...
    cdi_idx = cdi - 1                        # 0..4 (drier ↑)

    # 2. additive contribution
    base_risk += 0.40 * cdi_idx              # weight TBD by validation

    # 3. one new interaction rule (illustrative)
    if cdi_idx >= 3 && d <= 1
        # CDI says Severe/Extreme drought already observed BUT
        # forecast deficit is low → trust observation over forecast
        return [0.0, 0.05, 0.25, 0.50, 0.20]
    end
    # ... existing rules unchanged ...
end

# 4. extra loop in CPT builder
for cdi in 1:5, tl in 1:4, ag in 1:3, tr in 1:3, sp in 1:3, df in 1:5, cu in 1:5
    idx += 1; cpt[:, idx] = compute_risk_probs(cu, df, sp, tr, ag, tl, cdi)
end
```

That is the entire structural change.

---

## 3. Era-aware soft evidence on the CDI parent

This is the part that **eliminates a whole class of bugs** the
post-hoc-update pipeline keeps hitting (the 2012-01 fAPAR cutoff bug,
the pre-1995 symlink passthrough, the source-availability gating in
`run_drought_bn_backfill.py::cdi_command_for`).

In a BN, "observation missing" or "observation noisy" is just a
soft-evidence vector on the parent. The same Bayesian semantics handle
every data-availability regime:

| Era | CDI signal quality | Soft-evidence width σ | Behaviour at risk_level |
|---|---|---:|---|
| < 1995 | not available | uniform `[0.2, 0.2, 0.2, 0.2, 0.2]` | CDI parent factored out; SPI3-antecedent runs the BN |
| 1995-2000 | SPI + SMA, no fAPAR | σ ≈ 0.45 | weak contribution; SPI3 still primary |
| 2001-2011 | + MODIS fAPAR | σ ≈ 0.30 | meaningful contribution |
| 2012-01 only | fAPAR boundary dekad | σ widened to ≈ 0.40 for this init | no special-case code; just a wider bin |
| 2012+ | + EADW operational | σ ≈ 0.20 | sharply peaked; dominant antecedent signal |

No `if target < pd.Timestamp("2012-01-01")` anywhere. No `GDO_FPAR_OPERATIONAL_START`
constant to maintain. No symlink passthrough for pre-1995 inits. The
Bayesian semantics carry the data-availability story.

### Pre-1995 SPI3-only fallback (the user's specific question)

> CDI is not available for the pre-2000 year, so it acts as a
> replacement for the antecedent condition before CDI availability.
> Does P2 work in that case?

Yes — and the switching is implicit:

- For `target < 1995-01-01`, `cdi_data_prep` emits `[0.2, 0.2, 0.2, 0.2, 0.2]`.
- The BN multiplies this uniform vector into the joint posterior;
  the `cdi_state` parent contributes equal mass to every risk outcome.
- The posterior on risk_level is then *entirely* shaped by the other
  parents — current SPI3, deficit, spatial, trend, tail.
- De facto: SPI3 antecedent runs the BN, with CDI factored out.

No code branch picks SPI3-or-CDI. The model handles it.

---

## 4. Where calibration *is* needed

Three numbers — not three thousand:

| Parameter | What it controls | How to set |
|---|---|---|
| `cdi_weight` (suggested 0.40) | How strongly CDI moves base_risk per state step | Sweep on 2018-2024 (where post-hoc CRMA is most trusted); pick the value where joint posterior matches post-hoc CRMA on > 80% of boundary-init pairs |
| `cdi_interaction_rule_thresholds` | When CDI overrides forecast | Pull from existing `cdi_evidence_update.py` rule logic — it already encodes which CDI levels override which forecast scenarios; port the conditions |
| `cdi_soft_bin_sigma_by_era` | Source-quality penalty | Anchor σ at 0.45 / 0.30 / 0.20 from §3; refine if validation shows the 2001-2011 MODIS era is systematically over- or under-weighted vs 2012+ EADW |

That's it. No 4 500-cell elicitation.

---

## 5. Validation plan

A four-step validation, each cheap enough to run in an afternoon:

1. **Reproducibility check (2018-2024)** — re-run the existing 84
   inits (where post-hoc CDI update is the operational ground truth);
   compare joint-inference CRMA vs post-hoc CRMA per boundary-init.
   Target: > 85% agreement on Actionable_Risk and Monitor at minimum.

2. **Pre-CDI era sanity (1981-1994)** — verify joint inference with
   uniform CDI soft-evidence produces identical posteriors to the
   current SPI3-only path. Should be bit-identical modulo numerical
   noise (uniform vector × CPT row = CPT row averaged).

3. **Boundary-year stress (1995, 2001, 2012)** — three inits where the
   data-availability regime changes. Spot-check that the σ-widening
   produces gradual, sensible shifts in posterior, not discontinuities.

4. **Event corroboration** — pick five named drought events
   (e.g. 2010-2011 Horn of Africa, 2016-17 Eastern, 2022 Western)
   and verify the joint posterior fires Actionable_Risk on
   the right boundary-month timing.

---

## 6. What this replaces / retires

Once validated:

- **Delete** `cdi_evidence_update.py` (post-hoc step)
- **Delete** `cdi_command_for()` and `GDO_FPAR_OPERATIONAL_START` gating
- **Collapse** `run_drought_bn_backfill.py` from `prep,bn,cdi` to `prep,bn`
- **Keep** `cdi_data_prep.py` but refactor its output: emit a 5-vector
  per boundary (`cdi_p_normal,…,cdi_p_extreme`) instead of a single
  categorical level, with σ chosen by era
- **Keep** the `cdi_inputs_*.csv` files as a diagnostic intermediate;
  the new soft-evidence vector goes into the prep CSV alongside the
  existing SPI3 evidence columns

Net change to the pipeline:

```
old:   prep (Python) → bn (Julia warm) → cdi (Python post-hoc update)
new:   prep+cdi-fold (Python) → bn (Julia warm)
```

---

## 7. RxInfer parent-count: not a hard ceiling, just a performance cliff

Source-verified against ReactiveMP 5.6.6 (the version pinned in
`Manifest.toml`). The widely-cited "5-parent ceiling" is imprecise:

- **Predefined `@tullio` fast rules** in
  `ReactiveMP/src/rules/discrete_transition/predefined/belief_propagation.jl`
  exist for **1, 2, 3, and 4 conditioning parents** (CPT tensor up to
  5-D). These are the optimised inference path.
- **Generic structured-message rule** in
  `ReactiveMP/src/rules/discrete_transition/categoricals.jl:169-203`
  handles **any number of parents** via dynamic `sum_out_dimensions` /
  `multiply_dimensions!` calls. Slower per inference but no parent-count
  limit.

So adding CDI (4 → 5 parents) leaves the fast path and lands on the
generic rule: estimated ~2-4× per-init slowdown, to be benchmarked.
Adding CDI + METAR (4 → 6 parents) stays on the same generic path,
with a further ~5× growth in the CPT-build enumeration cost (one-shot,
not per-init). Neither is a feasibility blocker for the warm Julia
loop's ~1 s/init baseline — both are still well under 10 s/init worst
case for the 528-init backfill.

Practical implications:

- **`forecast_agreement` can stay deprecated** for v2 without performance
  pressure — there's no slot to free up, both 5- and 6-parent paths use
  the same generic rule anyway.
- **Factorisation is no longer load-bearing** — the proposal to absorb
  `current_spi3` and `cdi` into an intermediate `current_drought_state`
  node (to drop back to 4 parents) was driven by the 5-parent fast-path
  assumption. With the actual ceiling clarified, you can keep the
  parents explicit if domain semantics call for it.
- **What does matter**: benchmark the generic rule's per-init cost on
  this Julia 1.12 + RxInfer 4.7.3 stack before committing. If it
  exceeds ~5 s/init in the warm loop, the bulk-run alternative
  (matmul-style direct contraction, as in `flood_bn_ibf_v1.jl`'s
  `infer_soft_matmul()`) is the established fallback.

---

## 8. Why this wasn't already done

`cdi_bn_integration.md` recommended Path B (CDI as parent) months ago.
The reason it stalled, in retrospect, was that the doc framed the work
as "extend the CPT to 5 parents" without noticing the CPT was already
parametric. The post-hoc update path (`cdi_evidence_update.py`) was the
quick win and shipped first. The 13-hour 1981-2024 backfill we just
finished is the post-hoc pipeline's production debut — and the
2012-01 fAPAR cutoff bug is the kind of pain the joint-inference path
makes structurally impossible.

This is the right moment to revisit Path B.

---

## 9. Open questions for the team

1. Is `cdi_weight = 0.40` the right starting point, or should the weight
   be state-dependent (e.g. CDI=Alert weighted more strongly than
   CDI=Watch, mirroring the existing tail-risk `{0.0, 0.10, 0.35, 0.60}`
   pattern)?
2. Should we keep `forecast_agreement` for the planned NOAA GEFS
   integration and instead factorise CDI into the `current_spi3` parent
   (the original Path C in `cdi_bn_integration.md`)? Trade-off:
   structural cleanliness vs CPT redesign cost.
3. Do we need a separate `cdi_source` parent (operational vs MODIS
   backfill vs SPI+SMA-only) to model source heterogeneity explicitly,
   or is the era-weighted σ sufficient?
4. Is the post-hoc diagnostic (pre-CDI vs post-CDI risk levels in the
   output CSV) operationally useful enough to preserve via a "CDI-off"
   shadow run, or can it be retired?

---

## 10. Files touched (preview)

- `drought_bn_ibf_v1.jl` — extend `compute_risk_probs` signature,
  add CDI loop in `build_risk_cpt` and `build_risk_cpt_tensor`
- `drought_bn_ibf_v1_warm.jl` — read CDI soft-evidence columns from
  the prep CSV; pass to inference
- `cdi_data_prep.py` — emit `cdi_p_*` soft-evidence vector with
  era-weighted σ; retire categorical-only output
- `drought_data_prep.py` — merge CDI soft-evidence columns into the
  prep CSV (single source of truth into the BN)
- `run_drought_bn_backfill.py` — collapse to `prep,bn` stages; delete
  `step_cdi`, `cdi_command_for`, `cdi_inputs_dir` plumbing
- `cdi_evidence_update.py` — **delete** (or keep as a "CDI-off shadow"
  diagnostic per question 4 above)
- `evidence_nodes.md` — update §3 BN structure to show 5-parent risk_level
- `cdi_bn_integration.md` — add a "v2 status: implemented via parametric
  scoring rule" note at the top of Path B

No new dependencies; no Julia package additions; no RxInfer version
bump. This is a refactor of existing code, not a new system.
