# Flood IBF Bayesian Network: Python vs Julia Comparison

A detailed comparison of `flood_bn_ibf_v1.py` (pgmpy) and `flood_bn_ibf_v1.jl` (RxInfer.jl) — what is equivalent, what diverges, and where the Julia implementation could drastically differ going forward.

---

## 1. What Is Identical

### 1.1 DAG Structure

Both implementations encode the same directed acyclic graph:

```
antecedent_rainfall ──┐
exceedance_prob ──────┤
spatial_coverage ─────┼──► risk_level ──► action
rainfall_trend ───────┤
forecast_agreement ───┘
```

Six nodes, five edges into `risk_level`, one edge into `action`. The optional `forecast_agreement` toggle (`include_agreement_node`) is preserved.

### 1.2 State Spaces

Every node has the same discrete states in both versions:

| Node | States | Cardinality |
|------|--------|-------------|
| `antecedent_rainfall` | Dry, Normal, Wet, Very_Wet, Saturated | 5 |
| `exceedance_prob` | Very_Low, Low, Medium, High, Very_High | 5 |
| `spatial_coverage` | Localized, Moderate, Widespread | 3 |
| `rainfall_trend` | Decreasing, Stable, Increasing | 3 |
| `forecast_agreement` | Low, Medium, High | 3 |
| `risk_level` | Minimal, Low, Moderate, High, Extreme | 5 |
| `action` | Monitor, Alert, Prepare, Act | 4 |

### 1.3 Expert Rules and CPT Values

The `_compute_risk_probs` logic (Python) maps 1:1 to `compute_risk_probs` (Julia). Every expert rule, threshold, weight, and the agreement-based uniform mixing is numerically identical. The action CPT matrix is the same 4×5 table.

### 1.4 Categorization Functions

`categorize_antecedent_rainfall`, `_categorize` (Python) are replicated by `categorize_antecedent`, `categorize_exceedance`, `categorize_spatial`, `categorize_trend`, `categorize_agreement` (Julia). Same thresholds, same bins.

### 1.5 Boundary Processing Pipeline

Both versions iterate over a list of boundary inputs, categorize continuous values into discrete states, run inference, and return risk level + recommended action with confidence scores.

---

## 2. Where They Differ (Implementation Details)

### 2.1 Inference Algorithm

| Aspect | Python (pgmpy) | Julia (RxInfer.jl) |
|--------|----------------|---------------------|
| **Algorithm** | Variable Elimination | Reactive Message Passing (Belief Propagation) |
| **Mechanism** | Eliminates variables one by one, builds factor products | Passes messages along factor graph edges until convergence |
| **Complexity** | Exact; cost depends on elimination order | Exact for trees; iterative (loopy BP) for graphs with cycles |
| **Object created per query** | New `VariableElimination` instance | Compiled factor graph, reusable across queries |

For this specific DAG (a tree with no loops), both algorithms give **exact identical results**. The difference only matters if the graph structure changes.

### 2.2 Multi-Parent Encoding

**Python (pgmpy):** Handles multiple parents natively. `TabularCPD` accepts `evidence` and `evidence_card` lists, and pgmpy internally manages the flattened CPT indexing.

**Julia (RxInfer.jl):** `DiscreteTransition(y, x, T)` takes a single categorical parent `x`. To handle 5 parents flowing into `risk_level`, the Julia port introduces a **"super-parent" encoding** — a deterministic function that maps the 5 parent indices into a single index over 675 (or 225) combinations:

```julia
function encode_parents(ant, exc, spa, tre, agr)
    return ((agr-1)*225 + (tre-1)*75 + (spa-1)*25 + (exc-1)*5 + (ant-1)) + 1
end
```

This is semantically equivalent but architecturally different — it collapses the multi-parent CPT into a single large transition matrix.

### 2.3 Object Model

**Python:** Class-based (`FloodBayesianNetworkV1`). The pgmpy `BayesianNetwork` object holds edges, CPDs, and validates the model via `check_model()`. Inference creates a separate `VariableElimination` object.

**Julia:** Two-track design:
1. **`infer_direct()`** — Pure matrix indexing, no model object. Pre-builds CPT matrices once, then indexes directly. This is functionally a lookup table, not a graphical model inference.
2. **`@model flood_bn_model`** — Declarative RxInfer model with `Categorical` and `DiscreteTransition` nodes. Creates a factor graph compiled by RxInfer's engine.

### 2.4 Type System

**Python:** Dynamic typing throughout. `boundary_data` is a `Dict[str, Any]`. State names are strings matched at runtime.

**Julia:** Uses concrete structs (`BoundaryInput`, `BoundaryResult`) with typed fields. State indices are `Int` (1-based) during computation, mapped to strings only at output. This catches type errors at compile time and enables better optimization.

### 2.5 CPT Construction Timing

**Python:** CPTs are built in `__init__` → `_setup_cpds()` and stored inside the pgmpy model. Rebuilt each time a new `FloodBayesianNetworkV1` is instantiated.

**Julia:** CPTs are built by `build_risk_cpt()` and `build_action_cpt()` as standalone matrices, passed explicitly to inference. They can be precomputed once and reused across multiple runs without re-instantiating any model object.

### 2.6 The `confidence` column — realigned, and now identical

**This one changed emitted numbers, not just implementation.** Both sides have
been fixed; the entry stays because the old values are still on disk in every
`julia_risk_*.csv` written before this.

**Was**, in both: `confidence = max(action_probs)` — the peak of the **action**
distribution.

**Now**, in both: `posterior_confidence(risk_probs) = 1 − H/H_max` — the
sharpness of the risk posterior.

The old definition read off the action node, which the Julia file marks
`# deprecated (Layer-2 leakage)`. So it reported how peaked a deprecated
distribution was, not how uncertain the hazard belief is: **a boundary with a
flat, genuinely uncertain risk posterior could report high confidence.** The
entropy form scores a flat posterior 0 regardless of which state edges ahead,
and is computed on the same distribution `compute_crma_state` reads, so the
confidence and the state describe one object.

Measured on init 2026-03-03, 275 rows, identical flags — `confidence` is the
**only** column that changes; `risk_level`, `crma_state`, `traffic_light` and
all ten probability columns are byte-identical:

| | old (max action prob) | new (entropy sharpness) |
|---|---|---|
| mean | 0.540 | 0.368 |
| range | 0.372 – 0.760 | 0.148 – 0.509 |
| correlation with posterior entropy | −0.865 | **−1.000** (by construction) |

Rank correlation between old and new is **+0.910**, so the *ordering* of
boundaries by confidence is largely preserved — but the levels drop by ~0.17.
**Anything comparing this column against a fixed numeric threshold, or plotting
it on a fixed 0–1 colour scale, needs recalibrating.** Known consumers —
`summarize_bn.py` (`mean_confidence`), `plot_bn_dag_per_day.py` and
`plot_nairobi_diagnostic.py` — only report or print the value, so none break.

**Parity is verified, not assumed.** `posterior_confidence` in
`flood_bn_ibf_v1.py:219` and in `flood_bn_ibf_v1.jl` agree to **2.2e-16**
(1 ULP) across nine cases, edge cases included:

| input | value |
|---|---|
| flat `[0.2]*5` | 0.000000000000000 |
| one-hot | 1.000000000000000 |
| `[0, 0, .1, .1, .8]` | 0.602947181302750 |
| `[.97, .01, .01, .005, .005]` | 0.891495032336291 |
| all zeros | 0.0 — guarded, not `NaN` |
| unnormalised `[2.0]*5` | 0.0 — normalised first |

Both normalise before taking entropy, both floor the log argument at `1e-12`,
both clamp to `[0, 1]`, and both use natural log — which cancels in `H/log k`,
so the base is not a divergence risk. **If either is edited, re-run this
comparison.** The whole reason this section exists is that the two drifted once
already, and the divergence was invisible because nothing compared them.

---

## 3. Where Julia Could Drastically Diverge

These are not just implementation differences — they represent fundamentally different capabilities that RxInfer.jl enables and pgmpy does not.

### 3.1 Online / Streaming Inference

**Python (pgmpy):** Batch-only. Each call to `VariableElimination.query()` is independent. Processing 300 boundaries means 300 independent inference runs with no shared computation.

**Julia (RxInfer.jl):** The reactive message-passing engine natively supports **streaming data**. As new IMERG observations or GEFS forecasts arrive, the factor graph can be updated incrementally without rebuilding:

```julia
# Hypothetical streaming API
subscription = subscribe!(model, new_evidence_stream) do posterior
    update_dashboard(posterior)
end
```

This is a fundamental architectural difference — the Julia version could run as a **persistent service** that updates risk estimates in real-time as data flows in, rather than batch-processing boundaries one by one.

### 3.2 Learning CPTs from Data (Bayesian Parameter Learning)

**Python (pgmpy):** CPTs are hard-coded expert tables. pgmpy does support `BayesianEstimator` and `MaximumLikelihoodEstimator`, but these are separate workflows that don't integrate with the inference engine.

**Julia (RxInfer.jl):** By replacing fixed CPT matrices with `Dirichlet` priors, the same `@model` can **jointly infer risk levels AND learn CPT parameters** from observed flood outcomes:

```julia
@model function flood_bn_learnable()
    # Dirichlet prior over CPT rows — learns from data
    for col in 1:n_combos
        risk_cpt_col[col] ~ Dirichlet(ones(5))  # uninformative prior
    end
    # ... rest of model
end
```

This means the expert-elicited CPTs could serve as **informative priors** that get refined as real flood events are observed — a critical upgrade path for operational IBF systems. pgmpy has no equivalent within its inference loop.

### 3.3 Soft / Uncertain Evidence

**Python (pgmpy):** Evidence is hard — `evidence={'antecedent_rainfall': 'Wet'}`. You cannot say "70% Wet, 30% Very_Wet" without manually computing a mixture.

**Julia (RxInfer.jl):** Evidence can be a full probability vector (soft evidence):

```julia
# Hard evidence (Python-equivalent)
data = (parent_combo = onehot(idx, n_combos),)

# Soft evidence — e.g., satellite observation is uncertain
data = (parent_combo = [0.0, ..., 0.7, ..., 0.3, ..., 0.0],)
```

This matters for flood IBF because:
- IMERG observations have pixel-level uncertainty
- GEFS ensemble spread represents genuine forecast uncertainty
- Antecedent conditions near category boundaries (e.g., 59mm — is it "Wet" or "Very_Wet"?) should propagate that uncertainty rather than making a hard cut

### 3.4 Hierarchical / Multi-Scale Models

**Python (pgmpy):** Each boundary is processed independently. No mechanism to share information across neighboring boundaries or across time steps.

**Julia (RxInfer.jl):** The `@model` macro supports hierarchical structure natively. A country-level prior could inform admin-1 level estimates:

```julia
@model function hierarchical_flood_bn(n_boundaries)
    # Country-level risk prior
    country_risk ~ Dirichlet(ones(5))

    for i in 1:n_boundaries
        # Boundary-specific risk informed by country prior
        risk[i] ~ DiscreteTransition(parent_combo[i], risk_cpt)
        # Spatial smoothing with neighbors could be added here
    end
end
```

This would allow the system to borrow strength from data-rich regions to improve estimates in data-sparse ones — especially relevant for areas like South Sudan or Djibouti with limited historical observations.

### 3.5 Performance at Scale

| Dimension | Python (pgmpy) | Julia (RxInfer.jl) |
|-----------|----------------|---------------------|
| **Single boundary inference** | ~5-15ms (Python overhead + factor operations) | ~0.01-0.05ms (compiled matrix index) |
| **300 boundaries** | ~3-5 seconds | ~5-15ms (with `infer_direct`) |
| **JIT compilation** | None | First call compiles; subsequent calls are native-speed |
| **Parallelism** | GIL-limited; needs multiprocessing | Native threads via `Threads.@threads` |
| **GPU** | Not supported in pgmpy | RxInfer supports GPU-backed arrays for large-scale models |

For operational IBF running every 6 hours across East Africa (~300 admin-1 boundaries), this means the Julia BN inference step becomes negligible compared to data I/O — which is where the real bottleneck is.

### 3.6 Model Composition and Extensibility

**Python (pgmpy):** Adding a new node (e.g., soil moisture, elevation class) requires modifying `_create_risk_cpt` to handle the new dimension, exponentially growing the CPT.

**Julia (RxInfer.jl):** RxInfer's message-passing factorizes naturally. New factors can be added without rebuilding the entire CPT:

```julia
@model function flood_bn_extended()
    # Original parents
    risk_from_weather ~ DiscreteTransition(weather_combo, weather_cpt)
    # New factor: soil/terrain risk (separate sub-model)
    risk_from_terrain ~ DiscreteTransition(terrain_combo, terrain_cpt)
    # Combine via a mixing node
    combined_risk ~ MixtureOf(risk_from_weather, risk_from_terrain, mixing_weights)
    action ~ DiscreteTransition(combined_risk, action_cpt)
end
```

This avoids the combinatorial explosion of the monolithic CPT approach used in the Python version (currently 675 columns — adding a 4-state soil moisture node would push it to 2700).

---

## 4. What Is NOT Ported

The Julia file covers only the **BN inference engine**. The following Python components have no Julia equivalent yet:

| Component | Python Class/Function | Julia Status |
|-----------|----------------------|--------------|
| Data loading (IMERG) | `FloodDataLoaderV1.load_imerg_observations` | Not ported — needs `Rasters.jl` / `YAXArrays.jl` |
| Data loading (GEFS) | `FloodDataLoaderV1.load_gefs_probabilities` | Not ported — needs `Zarr.jl` or `NCDatasets.jl` |
| Boundary extraction | `FloodDataLoaderV1.extract_boundary_data` | Not ported — needs `GeoDataFrames.jl` + `Rasters.jl` |
| Region masking | `regionmask` usage | Not ported — needs `Rasters.jl` masking or custom implementation |
| Regridding | `xesmf` usage | Not ported — needs `Interpolations.jl` or custom |
| CLI argument parsing | `argparse` | Minimal — needs `Comonicon.jl` or `ArgParse.jl` |

---

## 5. Migration Recommendation

### Phase 1: Validate (current state)
Run both Python and Julia versions on the same boundary inputs. Compare risk probabilities to <1e-6 tolerance. The `self_test()` function in the Julia file covers basic cases.

### Phase 2: Replace inference only
Keep the Python data pipeline (`FloodDataLoaderV1`). Call Julia for BN inference via `PyJulia` or by writing boundary data to CSV → Julia processes → reads results back. This gets the performance and streaming benefits without rewriting the data layer.

### Phase 3: Enable learning
Add `Dirichlet` priors to CPT columns. Feed historical flood event data (EMDAT, FloodList, ICPAC records) to update the expert-elicited CPTs. This is the highest-value divergence from the Python version.

### Phase 4: Full Julia pipeline
Port data loading to `YAXArrays.jl` + `Rasters.jl` + `GeoDataFrames.jl` for an end-to-end Julia system. Only worthwhile if the team commits to Julia as the operational stack.
