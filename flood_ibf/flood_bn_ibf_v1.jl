#=
Flood Impact-Based Forecasting using Bayesian Networks - RxInfer.jl Port

Port of flood_bn_ibf_v1.py Bayesian Network to Julia/RxInfer.jl
Uses reactive message passing for inference instead of variable elimination.

Dependencies (add to Project.toml or install manually):
    using Pkg
    Pkg.add(["RxInfer", "CSV", "DataFrames", "JSON3"])

Usage:
    julia --project flood_bn_ibf_v1.jl --boundaries path/to/boundaries.geojson --date 2026-01-15

Author: ICPAC IBF Team
Date: April 2026
=#

using LinearAlgebra
using Printf
using CSV
using DataFrames
using RxInfer                                                    # required

# ============================================================================
# CONSTANTS
# ============================================================================

# State labels for each node
const ANTECEDENT_STATES = ["Dry", "Normal", "Wet", "Very_Wet", "Saturated"]  # 5
const EXCEEDANCE_STATES = ["Very_Low", "Low", "Medium", "High", "Very_High"]  # 5
const SPATIAL_STATES    = ["Localized", "Moderate", "Widespread"]             # 3
const TREND_STATES      = ["Decreasing", "Stable", "Increasing"]             # 3
const AGREEMENT_STATES  = ["Low", "Medium", "High"]                          # 3
const TAIL_RISK_STATES  = ["Nil", "Low", "Moderate", "High"]                 # 4
const RISK_STATES       = ["Minimal", "Low", "Moderate", "High", "Extreme"]  # 5
const ACTION_STATES     = ["Monitor", "Alert", "Prepare", "Act"]             # 4 (deprecated)
const CRMA_STATES       = ["Monitor", "Evaluate", "Assess", "Actionable_Risk"] # 4 (Layer-1 output)
const TRAFFIC_LIGHT     = Dict(
    "Monitor"         => "Green",
    "Evaluate"        => "Yellow",
    "Assess"          => "Orange",
    "Actionable_Risk" => "Red",
)

# Precipitation thresholds (mm/24h)
const PRECIP_THRESHOLDS_24H = Dict(
    "light"       => 5,
    "moderate"    => 25,
    "heavy"       => 50,
    "very_heavy"  => 75,
    "extreme"     => 100,
    "exceptional" => 125,
)

# Antecedent rainfall thresholds (7-day accumulated, mm)
const ANTECEDENT_THRESHOLDS = Dict(
    "dry"       => 10.0,
    "normal"    => 30.0,
    "wet"       => 60.0,
    "very_wet"  => 100.0,
    "saturated" => Inf,
)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

"""
Categorize 7-day accumulated rainfall into antecedent condition index (1-5).
"""
function categorize_antecedent(rainfall_mm::Float64)::Int
    isnan(rainfall_mm) && return 2  # Normal
    rainfall_mm < ANTECEDENT_THRESHOLDS["dry"]      && return 1  # Dry
    rainfall_mm < ANTECEDENT_THRESHOLDS["normal"]    && return 2  # Normal
    rainfall_mm < ANTECEDENT_THRESHOLDS["wet"]       && return 3  # Wet
    rainfall_mm < ANTECEDENT_THRESHOLDS["very_wet"]  && return 4  # Very_Wet
    return 5  # Saturated
end

"""
Categorize exceedance probability (0-1) into discrete state index (1-5).
"""
function categorize_exceedance(eprob::Float64)::Int
    isnan(eprob) && return 1
    eprob < 0.2 && return 1  # Very_Low
    eprob < 0.4 && return 2  # Low
    eprob < 0.6 && return 3  # Medium
    eprob < 0.8 && return 4  # High
    return 5  # Very_High
end

"""
Categorize spatial coverage fraction (0-1) into discrete state index (1-3).
"""
function categorize_spatial(coverage::Float64)::Int
    isnan(coverage) && return 1
    coverage < 0.3 && return 1  # Localized
    coverage < 0.6 && return 2  # Moderate
    return 3  # Widespread
end

"""
Map trend string to index (1-3).
"""
function categorize_trend(trend::String)::Int
    t = lowercase(trend)
    t == "decreasing" && return 1
    t == "increasing" && return 3
    return 2  # Stable
end

"""
Map agreement string to index (1-3).
"""
function categorize_agreement(agreement::String)::Int
    a = lowercase(agreement)
    a == "low"  && return 1
    a == "high" && return 3
    return 2  # Medium
end

"""
Categorize ensemble-max / threshold ratio into tail-risk index (1-4).
Captures whether any single ensemble member exceeds the RP threshold.
"""
function categorize_tail_risk(max_ratio::Float64)::Int
    isnan(max_ratio) && return 1
    max_ratio < 0.5 && return 1  # None — well below threshold
    max_ratio < 1.0 && return 2  # Low — approaching threshold
    max_ratio < 2.0 && return 3  # Moderate — at least 1 member exceeds
    return 4  # High — member well above threshold (≥ 2× RP)
end

"""
Sharpness of a posterior: 1 - H/H_max in [0,1] (H_max = log k for k states).

This replaces `confidence = maximum(action_probs)`, which was read off the
action node -- a node this file itself marks deprecated (Layer-2 leakage), so
the emitted `confidence` reported how peaked a deprecated distribution was
rather than how uncertain the hazard belief is.  A basin with a flat, genuinely
uncertain risk posterior could report high confidence.

Entropy is the honest measure: a flat posterior (we know nothing) scores 0
regardless of which state happens to edge ahead, and only a genuinely peaked
belief scores near 1.  It is a statement about the *quality of the belief*,
which is what CRMA is accountable for.

Computed on `risk_probs` -- the same distribution `compute_crma_state` reads --
so the confidence and the state it accompanies describe one object.
"""
function posterior_confidence(probs::Vector{Float64})::Float64
    k = length(probs)
    k <= 1 && return 1.0
    s = sum(probs)
    s > 0 || return 0.0
    p = probs ./ s
    H = -sum(x * log(max(x, 1e-12)) for x in p)
    return clamp(1.0 - H / log(k), 0.0, 1.0)
end

"""
Compute the CRMA risk-assessment state from the risk_level posterior
using a cost-loss-ratio based trigger rule.

The cost-loss framing (Murphy 1977, Richardson 2000, Lopez et al. 2020):
rational trigger when P(event) ≥ C/L, where C = cost of acting early,
L = loss from missed event. For FbF cash transfers, C/L ≈ 0.1;
pre-positioned stockpiles, C/L ≈ 0.2 (Weingärtner & Wilkinson 2019).

Four-state rule:
  Actionable_Risk : P(High) + P(Extreme) ≥ cost_loss_ratio
  Assess          : P(Mod) + P(High) + P(Extreme) ≥ max(2·C/L, 0.4)
  Evaluate        : P(Low) + P(Mod) + P(High) + P(Extreme) ≥ max(3·C/L, 0.3)
  Monitor         : otherwise

Returns (state_idx, explanation_string).
"""
function compute_crma_state(risk_probs::Vector{Float64};
                            cost_loss_ratio::Float64=0.2)
    p_minimal  = risk_probs[1]
    p_low      = risk_probs[2]
    p_moderate = risk_probs[3]
    p_high     = risk_probs[4]
    p_extreme  = risk_probs[5]

    p_act      = p_high + p_extreme
    p_assess   = p_moderate + p_high + p_extreme
    p_evaluate = p_low + p_moderate + p_high + p_extreme

    θ_act      = cost_loss_ratio
    θ_assess   = max(2.0 * cost_loss_ratio, 0.40)
    θ_evaluate = max(3.0 * cost_loss_ratio, 0.30)

    if p_act >= θ_act
        expl = "P(High∪Extreme)=$(round(p_act, digits=2)) ≥ C/L=$(round(θ_act, digits=2))"
        return 4, expl
    elseif p_assess >= θ_assess
        expl = "P(Mod∪High∪Extreme)=$(round(p_assess, digits=2)) ≥ $(round(θ_assess, digits=2))"
        return 3, expl
    elseif p_evaluate >= θ_evaluate
        expl = "P(Low∪Mod∪High∪Extreme)=$(round(p_evaluate, digits=2)) ≥ $(round(θ_evaluate, digits=2))"
        return 2, expl
    else
        expl = "all conditional masses below thresholds"
        return 1, expl
    end
end

# ============================================================================
# CPT CONSTRUCTION (mirrors _compute_risk_probs from Python)
# ============================================================================

"""
Compute risk probability vector [5] given parent state indices.
Extends the original expert rules with a tail_risk node that captures
whether any single ensemble member exceeds the RP threshold.
"""
function compute_risk_probs(
    antecedent::Int,  # 1-5
    exceed::Int,      # 1-5
    spatial::Int,     # 1-3
    trend::Int,       # 1-3
    agreement::Int,   # 1-3
    tail::Int=1,      # 1-4 (None, Low, Moderate, High)
)::Vector{Float64}
    # Convert to 0-based for the arithmetic (matching Python)
    a = antecedent - 1
    e = exceed - 1
    s = spatial - 1
    t = trend - 1
    ag = agreement - 1
    tr = tail - 1

    # Base risk score
    base_risk = a * 0.30 + e * 0.55

    # Spatial modifier
    if s == 2       # Widespread
        base_risk += 0.5
    elseif s == 1   # Moderate
        base_risk += 0.25
    end

    # Trend modifier
    if t == 2       # Increasing
        base_risk += 0.35
    elseif t == 0   # Decreasing
        base_risk -= 0.30
    end

    # Tail risk modifier: boost risk when ensemble max exceeds threshold
    # even if mean exceedance probability is low
    if tr == 3       # High: ens_max ≥ 2× threshold
        base_risk += 0.60
    elseif tr == 2   # Moderate: ens_max exceeds threshold (1-2×)
        base_risk += 0.35
    elseif tr == 1   # Low: approaching threshold (0.5-1.0×)
        base_risk += 0.10
    end

    # Expert rules for specific scenarios
    probs = if a == 4 && e >= 3 && t == 2
        # Rule 1: Saturated + High/Very_High + Increasing
        if s >= 1
            [0.0, 0.0, 0.05, 0.20, 0.75]
        else
            [0.0, 0.0, 0.10, 0.40, 0.50]
        end
    elseif a >= 3 && e >= 3 && t == 2
        # Rule 2: Very_Wet/Saturated + High + Increasing
        [0.0, 0.0, 0.10, 0.50, 0.40]
    # Rule T1: Low mean exceedance BUT high tail risk + wet/saturated ground
    elseif tr >= 2 && a >= 3 && e <= 2
        if tr == 3  # High tail risk
            [0.0, 0.10, 0.30, 0.45, 0.15]
        else        # Moderate tail risk
            [0.05, 0.20, 0.45, 0.25, 0.05]
        end
    # Rule T2: Low mean exceedance BUT high tail risk (any antecedent)
    elseif tr == 3 && e <= 1
        [0.05, 0.20, 0.40, 0.30, 0.05]
    # Rule T3: Moderate tail risk + increasing trend
    elseif tr >= 2 && t == 2 && e <= 2
        [0.05, 0.15, 0.45, 0.30, 0.05]
    elseif a == 0 && e <= 2 && tr <= 1
        # Rule 3: Dry + low forecast + no tail risk
        [0.55, 0.35, 0.10, 0.0, 0.0]
    elseif t == 0 && a <= 2 && e <= 1 && tr <= 1
        # Rule 4: Decreasing + low antecedent + no tail risk
        [0.65, 0.30, 0.05, 0.0, 0.0]
    elseif a <= 1 && e >= 3
        # Rule 5: High forecast but dry
        [0.10, 0.25, 0.45, 0.15, 0.05]
    elseif base_risk < 1
        [0.50, 0.40, 0.10, 0.0, 0.0]
    elseif base_risk < 2
        [0.10, 0.35, 0.40, 0.15, 0.0]
    elseif base_risk < 3
        [0.05, 0.15, 0.45, 0.30, 0.05]
    elseif base_risk < 4
        [0.0, 0.05, 0.25, 0.50, 0.20]
    else
        [0.0, 0.0, 0.10, 0.40, 0.50]
    end

    # Forecast agreement modifier
    uniform = fill(0.20, 5)
    if ag == 0       # Low agreement → more uncertainty
        probs = 0.5 .* probs .+ 0.5 .* uniform
    elseif ag == 1   # Medium agreement
        probs = 0.8 .* probs .+ 0.2 .* uniform
    end

    return probs ./ sum(probs)
end

"""
Build the full risk CPT as a 3D tensor for DiscreteTransition.

RxInfer DiscreteTransition expects a transition matrix where:
  T[child_state, parent_combo] maps a flattened parent index to child distribution.

Since risk_level has 5 parents, we flatten them into a single "super-parent"
and build a (5 × N_combos) matrix.

Returns: (risk_tensor, n_combos)
"""
function build_risk_cpt(; include_agreement::Bool=true, include_tail_risk::Bool=false)
    if include_tail_risk
        if include_agreement
            n_combos = 5 * 5 * 3 * 3 * 3 * 4  # 2700
        else
            n_combos = 5 * 5 * 3 * 3 * 4      # 900
        end
    else
        if include_agreement
            n_combos = 5 * 5 * 3 * 3 * 3  # 675
        else
            n_combos = 5 * 5 * 3 * 3      # 225
        end
    end

    cpt = zeros(Float64, 5, n_combos)
    idx = 0

    if include_tail_risk && include_agreement
        for tl in 1:4, ag in 1:3, tr in 1:3, sp in 1:3, ex in 1:5, ant in 1:5
            idx += 1
            cpt[:, idx] = compute_risk_probs(ant, ex, sp, tr, ag, tl)
        end
    elseif include_tail_risk
        for tl in 1:4, tr in 1:3, sp in 1:3, ex in 1:5, ant in 1:5
            idx += 1
            cpt[:, idx] = compute_risk_probs(ant, ex, sp, tr, 3, tl)
        end
    elseif include_agreement
        for ag in 1:3, tr in 1:3, sp in 1:3, ex in 1:5, ant in 1:5
            idx += 1
            cpt[:, idx] = compute_risk_probs(ant, ex, sp, tr, ag, 1)
        end
    else
        for tr in 1:3, sp in 1:3, ex in 1:5, ant in 1:5
            idx += 1
            cpt[:, idx] = compute_risk_probs(ant, ex, sp, tr, 3, 1)
        end
    end

    return cpt, n_combos
end

"""
Build the risk CPT as a tensor, axis order matching the RxInfer
DiscreteTransition call `risk ~ DiscreteTransition(ant, T, exc, spa, trn, tail)`.
Shape: `(risk=5, ant=5, exc=5, spa=3, trn=3, tail=4)` when `include_tail_risk`,
       `(risk=5, ant=5, exc=5, spa=3, trn=3)`          otherwise.
`include_agreement=true` is not supported here — the library's exact
rules top out at 5 conditioning parents; turn agreement into soft evidence
on another node or run the legacy matmul path if you need it.
"""
function build_risk_cpt_tensor(; include_tail_risk::Bool=true)
    if include_tail_risk
        T = zeros(Float64, 5, 5, 5, 3, 3, 4)
        for tl in 1:4, tr in 1:3, sp in 1:3, ex in 1:5, ant in 1:5
            T[:, ant, ex, sp, tr, tl] = compute_risk_probs(ant, ex, sp, tr, 3, tl)
        end
        return T
    else
        T = zeros(Float64, 5, 5, 5, 3, 3)
        for tr in 1:3, sp in 1:3, ex in 1:5, ant in 1:5
            T[:, ant, ex, sp, tr] = compute_risk_probs(ant, ex, sp, tr, 3, 1)
        end
        return T
    end
end

"""
Build the action CPT (4 × 5 matrix): action | risk_level.
"""
function build_action_cpt()::Matrix{Float64}
    # Rows = action states, Cols = risk_level states
    # Matches Python exactly
    return [
        0.95  0.15  0.00  0.00  0.00;  # Monitor
        0.05  0.80  0.20  0.05  0.00;  # Alert
        0.00  0.05  0.75  0.25  0.05;  # Prepare
        0.00  0.00  0.05  0.70  0.95;  # Act
    ]
end

# ============================================================================
# RxInfer MODEL DEFINITION
# ============================================================================

#=
RxInfer.jl uses the @model macro to define probabilistic models.
For discrete BNs with fixed CPTs we use:
  - Categorical(prior) for root nodes
  - DiscreteTransition(parent, cpt) for conditional nodes

Since risk_level depends on 5 parents, we encode the joint parent state
as a single Categorical variable via a deterministic mapping, then use
DiscreteTransition with the full CPT.

NOTE: RxInfer's DiscreteTransition(y, x, T) means:
  y ~ Categorical(T * x)  where x is a one-hot vector.
=#

"""
Encode multiple discrete parent indices into a single flat index (1-based).
Used to collapse 5 parents into one "super-parent" for DiscreteTransition.
"""
function encode_parents(ant::Int, exc::Int, spa::Int, tre::Int, agr::Int;
                        tail::Int=1, include_tail_risk::Bool=false)::Int
    if include_tail_risk
        # Order: tail(outer) > agreement > trend > spatial > exceedance > antecedent(inner)
        return ((tail - 1) * 3 * 3 * 3 * 5 * 5 +
                (agr - 1) * 3 * 3 * 5 * 5 +
                (tre - 1) * 3 * 5 * 5 +
                (spa - 1) * 5 * 5 +
                (exc - 1) * 5 +
                (ant - 1)) + 1
    else
        return ((agr - 1) * 3 * 3 * 5 * 5 +
                (tre - 1) * 3 * 5 * 5 +
                (spa - 1) * 5 * 5 +
                (exc - 1) * 5 +
                (ant - 1)) + 1
    end
end

function encode_parents_no_agreement(ant::Int, exc::Int, spa::Int, tre::Int;
                                     tail::Int=1, include_tail_risk::Bool=false)::Int
    if include_tail_risk
        # Order: tail(outer) > trend > spatial > exceedance > antecedent(inner)
        return ((tail - 1) * 3 * 3 * 5 * 5 +
                (tre - 1) * 3 * 5 * 5 +
                (spa - 1) * 5 * 5 +
                (exc - 1) * 5 +
                (ant - 1)) + 1
    else
        return ((tre - 1) * 3 * 5 * 5 +
                (spa - 1) * 5 * 5 +
                (exc - 1) * 5 +
                (ant - 1)) + 1
    end
end

# ============================================================================
# INFERENCE (Direct matrix multiplication - no need for full RxInfer model
# for this fixed-CPT case, but we show both approaches)
# ============================================================================

"""
Simple direct inference using matrix multiplication.
For fixed expert CPTs this is equivalent to variable elimination.

Args:
    antecedent_idx: 1-5
    exceedance_idx: 1-5
    spatial_idx: 1-3
    trend_idx: 1-3
    agreement_idx: 1-3
    risk_cpt: (5 × N) risk CPT matrix
    action_cpt: (4 × 5) action CPT matrix

Returns:
    (risk_probs, action_probs) - probability vectors
"""
function infer_direct(
    antecedent_idx::Int,
    exceedance_idx::Int,
    spatial_idx::Int,
    trend_idx::Int,
    agreement_idx::Int,
    risk_cpt::Matrix{Float64},
    action_cpt::Matrix{Float64};
    include_agreement::Bool=true,
    tail_risk_idx::Int=1,
    include_tail_risk::Bool=false,
)
    parent_idx = if include_agreement
        encode_parents(antecedent_idx, exceedance_idx, spatial_idx, trend_idx, agreement_idx;
                        tail=tail_risk_idx, include_tail_risk)
    else
        encode_parents_no_agreement(antecedent_idx, exceedance_idx, spatial_idx, trend_idx;
                                     tail=tail_risk_idx, include_tail_risk)
    end

    # P(risk | parents) = column of CPT
    risk_probs = risk_cpt[:, parent_idx]

    # P(action | evidence) = action_cpt * risk_probs (marginalize over risk)
    action_probs = action_cpt * risk_probs

    return risk_probs, action_probs
end

# ============================================================================
# RxInfer MODEL — multi-parent discrete BN with soft-evidence support.
# Verified pattern: each parent receives virtual evidence through a
# `DiscreteTransition(parent, diageye(K))` channel whose observation is either
# a one-hot vector (hard evidence) or a probability vector (soft evidence).
# The risk-level posterior is queried by attaching a `missing` observation on
# `risk_data`, which terminates the half-edge and triggers the forward marginal.
# Ref: https://examples.rxinfer.com/categories/basic_examples/bayesian_networks/
# ============================================================================

@model function flood_bn_model_5parent(T, ant_data, exc_data, spa_data, trn_data, tail_data, risk_data)
    ant  ~ Categorical(fill(1/5, 5))
    exc  ~ Categorical(fill(1/5, 5))
    spa  ~ Categorical(fill(1/3, 3))
    trn  ~ Categorical(fill(1/3, 3))
    tail ~ Categorical(fill(1/4, 4))
    ant_data  ~ DiscreteTransition(ant,  diageye(5))
    exc_data  ~ DiscreteTransition(exc,  diageye(5))
    spa_data  ~ DiscreteTransition(spa,  diageye(3))
    trn_data  ~ DiscreteTransition(trn,  diageye(3))
    tail_data ~ DiscreteTransition(tail, diageye(4))
    risk ~ DiscreteTransition(ant, T, exc, spa, trn, tail)
    risk_data ~ DiscreteTransition(risk, diageye(5))
end

@model function flood_bn_model_4parent(T, ant_data, exc_data, spa_data, trn_data, risk_data)
    ant  ~ Categorical(fill(1/5, 5))
    exc  ~ Categorical(fill(1/5, 5))
    spa  ~ Categorical(fill(1/3, 3))
    trn  ~ Categorical(fill(1/3, 3))
    ant_data  ~ DiscreteTransition(ant,  diageye(5))
    exc_data  ~ DiscreteTransition(exc,  diageye(5))
    spa_data  ~ DiscreteTransition(spa,  diageye(3))
    trn_data  ~ DiscreteTransition(trn,  diageye(3))
    risk ~ DiscreteTransition(ant, T, exc, spa, trn)
    risk_data ~ DiscreteTransition(risk, diageye(5))
end

_rxinfer_init_5 = @initialization begin
    q(ant)  = Categorical(fill(1/5, 5))
    q(exc)  = Categorical(fill(1/5, 5))
    q(spa)  = Categorical(fill(1/3, 3))
    q(trn)  = Categorical(fill(1/3, 3))
    q(tail) = Categorical(fill(1/4, 4))
    q(risk) = Categorical(fill(1/5, 5))
end

_rxinfer_init_4 = @initialization begin
    q(ant)  = Categorical(fill(1/5, 5))
    q(exc)  = Categorical(fill(1/5, 5))
    q(spa)  = Categorical(fill(1/3, 3))
    q(trn)  = Categorical(fill(1/3, 3))
    q(risk) = Categorical(fill(1/5, 5))
end

"""
Soft-evidence inference via RxInfer. Each `*_ev` argument is a probability
vector over that parent's states (one-hot = hard evidence). Returns
`(risk_probs, action_probs)` so the caller can drop-in replace `infer_direct`.
`iterations` controls the fixed number of message-passing rounds; 10 is the
library's default for this idiom and converges on our fixed-CPT DAG.
"""
function infer_rxinfer_soft(
    ant_ev::Vector{Float64},
    exc_ev::Vector{Float64},
    spa_ev::Vector{Float64},
    trn_ev::Vector{Float64};
    tail_ev::Union{Nothing,Vector{Float64}}=nothing,
    risk_cpt_tensor::AbstractArray{Float64},
    action_cpt::Matrix{Float64},
    iterations::Int=10,
)::Tuple{Vector{Float64},Vector{Float64}}
    if tail_ev === nothing
        r = infer(
            model = flood_bn_model_4parent(T = risk_cpt_tensor),
            data  = (ant_data = ant_ev, exc_data = exc_ev, spa_data = spa_ev,
                     trn_data = trn_ev, risk_data = missing),
            iterations     = iterations,
            initialization = _rxinfer_init_4,
        )
    else
        r = infer(
            model = flood_bn_model_5parent(T = risk_cpt_tensor),
            data  = (ant_data = ant_ev, exc_data = exc_ev, spa_data = spa_ev,
                     trn_data = trn_ev, tail_data = tail_ev, risk_data = missing),
            iterations     = iterations,
            initialization = _rxinfer_init_5,
        )
    end
    risk_probs   = Vector{Float64}(last(r.posteriors[:risk]).p)
    action_probs = action_cpt * risk_probs
    return risk_probs, action_probs
end

"""
Build a one-hot probability vector of length `k` with 1.0 at position `idx`.
Used to convert a hard categorical classification into soft-evidence form.
"""
onehot(idx::Int, k::Int) = (v = zeros(Float64, k); v[idx] = 1.0; v)

# ============================================================================
# BOUNDARY PROCESSING
# ============================================================================

"""
Boundary data as a named tuple / struct for type safety.
"""
struct BoundaryInput
    id::String
    name::String
    country::String
    antecedent_rainfall_mm::Float64
    antecedent_category::String
    rainfall_trend::String
    ecmwf_eprob_heavy::Float64
    spatial_coverage::Float64
    forecast_agreement::String
    ens_max_ratio::Float64
    # Optional soft-evidence vectors (nothing => derive one-hot from the hard
    # categorisation above). Lengths must match the node state counts.
    ant_probs::Union{Nothing,Vector{Float64}}
    exc_probs::Union{Nothing,Vector{Float64}}
    spa_probs::Union{Nothing,Vector{Float64}}
    trn_probs::Union{Nothing,Vector{Float64}}
    tail_probs::Union{Nothing,Vector{Float64}}
end

BoundaryInput(id, name, country, ant_mm, ant_cat, trend, eprob, spa_cov, agr, ratio) =
    BoundaryInput(id, name, country, ant_mm, ant_cat, trend, eprob, spa_cov, agr, ratio,
                  nothing, nothing, nothing, nothing, nothing)

struct BoundaryResult
    boundary_id::String
    boundary_name::String
    country::String
    antecedent_category::String
    rainfall_trend::String
    risk_level::String
    risk_probabilities::Vector{Float64}
    recommended_action::String         # deprecated (Layer-2 leakage)
    action_probabilities::Vector{Float64}
    confidence::Float64
    crma_state::String                 # Layer-1 CRMA output
    crma_explanation::String           # rule that fired
    traffic_light::String              # Green / Yellow / Orange / Red
end

"""
Process a single boundary through the BN via the legacy matmul path. Kept
for validation and for the `include_agreement=true` case that exceeds
RxInfer's multi-parent tensor arity.
"""
function process_boundary(
    b::BoundaryInput,
    risk_cpt::Matrix{Float64},
    action_cpt::Matrix{Float64};
    include_agreement::Bool=true,
    include_tail_risk::Bool=false,
    cost_loss_ratio::Float64=0.2,
)::BoundaryResult
    ant_idx = categorize_antecedent(b.antecedent_rainfall_mm)
    exc_idx = categorize_exceedance(b.ecmwf_eprob_heavy)
    spa_idx = categorize_spatial(b.spatial_coverage)
    tre_idx = categorize_trend(b.rainfall_trend)
    agr_idx = categorize_agreement(b.forecast_agreement)
    tl_idx  = categorize_tail_risk(b.ens_max_ratio)

    risk_probs, action_probs = infer_direct(
        ant_idx, exc_idx, spa_idx, tre_idx, agr_idx,
        risk_cpt, action_cpt;
        include_agreement,
        tail_risk_idx=tl_idx,
        include_tail_risk,
    )

    return _assemble_result(b, ant_idx, tre_idx, risk_probs, action_probs, cost_loss_ratio)
end

"""
Process a single boundary through the BN via RxInfer's message passing.
Accepts soft evidence if `BoundaryInput` carries per-state probability
vectors; otherwise constructs one-hot evidence from the hard categorisation.
"""
function process_boundary_rxinfer(
    b::BoundaryInput,
    risk_cpt_tensor::AbstractArray{Float64},
    action_cpt::Matrix{Float64};
    include_tail_risk::Bool=false,
    cost_loss_ratio::Float64=0.2,
    iterations::Int=10,
)::BoundaryResult
    ant_idx = categorize_antecedent(b.antecedent_rainfall_mm)
    exc_idx = categorize_exceedance(b.ecmwf_eprob_heavy)
    spa_idx = categorize_spatial(b.spatial_coverage)
    tre_idx = categorize_trend(b.rainfall_trend)
    tl_idx  = categorize_tail_risk(b.ens_max_ratio)

    ant_ev = b.ant_probs === nothing ? onehot(ant_idx, 5) : b.ant_probs
    exc_ev = b.exc_probs === nothing ? onehot(exc_idx, 5) : b.exc_probs
    spa_ev = b.spa_probs === nothing ? onehot(spa_idx, 3) : b.spa_probs
    trn_ev = b.trn_probs === nothing ? onehot(tre_idx, 3) : b.trn_probs
    tail_ev = include_tail_risk ?
              (b.tail_probs === nothing ? onehot(tl_idx, 4) : b.tail_probs) :
              nothing

    risk_probs, action_probs = infer_rxinfer_soft(
        ant_ev, exc_ev, spa_ev, trn_ev;
        tail_ev = tail_ev,
        risk_cpt_tensor = risk_cpt_tensor,
        action_cpt = action_cpt,
        iterations = iterations,
    )

    return _assemble_result(b, ant_idx, tre_idx, risk_probs, action_probs, cost_loss_ratio)
end

function _assemble_result(b::BoundaryInput, ant_idx::Int, tre_idx::Int,
                           risk_probs::Vector{Float64}, action_probs::Vector{Float64},
                           cost_loss_ratio::Float64)::BoundaryResult
    crma_idx, crma_expl = compute_crma_state(risk_probs; cost_loss_ratio)
    crma_state = CRMA_STATES[crma_idx]
    traffic_light = TRAFFIC_LIGHT[crma_state]
    return BoundaryResult(
        b.id, b.name, b.country,
        ANTECEDENT_STATES[ant_idx],
        TREND_STATES[tre_idx],
        RISK_STATES[argmax(risk_probs)],
        risk_probs,
        ACTION_STATES[argmax(action_probs)],
        action_probs,
        posterior_confidence(risk_probs),
        crma_state, crma_expl, traffic_light,
    )
end

"""
Process all boundaries. Pre-builds CPTs once for efficiency.

`use_rxinfer=true` (default) routes through the reactive message-passing engine
with soft-evidence support. `use_rxinfer=false` falls back to the direct-matmul
path, which is currently the only option when `include_agreement=true` (the
library's exact `DiscreteTransition` rules top out at 5 conditioning parents).
"""
function process_all_boundaries(
    boundaries::Vector{BoundaryInput};
    include_agreement::Bool=true,
    include_tail_risk::Bool=false,
    cost_loss_ratio::Float64=0.2,
    use_rxinfer::Bool=true,
)::Vector{BoundaryResult}
    action_cpt = build_action_cpt()
    results = Vector{BoundaryResult}(undef, length(boundaries))

    if use_rxinfer && !include_agreement
        T = build_risk_cpt_tensor(; include_tail_risk)
        for (i, b) in enumerate(boundaries)
            results[i] = process_boundary_rxinfer(b, T, action_cpt;
                                                  include_tail_risk, cost_loss_ratio)
            if i % 50 == 0
                @info "Processed $i/$(length(boundaries)) boundaries (RxInfer)"
            end
        end
    else
        if use_rxinfer && include_agreement
            @info "include_agreement=true has 6 parents; RxInfer tensor arity insufficient — using matmul path"
        end
        risk_cpt, _ = build_risk_cpt(; include_agreement, include_tail_risk)
        for (i, b) in enumerate(boundaries)
            results[i] = process_boundary(b, risk_cpt, action_cpt;
                                          include_agreement, include_tail_risk, cost_loss_ratio)
            if i % 50 == 0
                @info "Processed $i/$(length(boundaries)) boundaries (matmul)"
            end
        end
    end

    return results
end

# ============================================================================
# CSV / DataFrame integration (optional, requires CSV + DataFrames)
# ============================================================================

"""
Convert results to a DataFrame (requires DataFrames.jl).
"""
function results_to_dataframe(results::Vector{BoundaryResult})
    try
        @eval using DataFrames
    catch
        @warn "DataFrames.jl not available, returning raw results"
        return results
    end

    return DataFrames.DataFrame(
        boundary_id        = [r.boundary_id for r in results],
        boundary_name      = [r.boundary_name for r in results],
        country            = [r.country for r in results],
        antecedent_category = [r.antecedent_category for r in results],
        rainfall_trend     = [r.rainfall_trend for r in results],
        risk_level         = [r.risk_level for r in results],
        recommended_action = [r.recommended_action for r in results],
        confidence         = [r.confidence for r in results],
    )
end

# ============================================================================
# DEMO / SELF-TEST
# ============================================================================

"""
Run a quick self-test to verify CPTs match the Python implementation.
"""
function self_test()
    @info "Running self-test..."

    risk_cpt, _ = build_risk_cpt(; include_agreement=true)
    action_cpt = build_action_cpt()

    # Test case 1: Saturated + Very_High + Widespread + Increasing + High agreement
    # Should give Extreme risk
    rp, ap = infer_direct(5, 5, 3, 3, 3, risk_cpt, action_cpt)
    @info "Test 1 (worst case):" risk=RISK_STATES[argmax(rp)] action=ACTION_STATES[argmax(ap)]
    @assert RISK_STATES[argmax(rp)] == "Extreme" "Expected Extreme risk"
    @assert ACTION_STATES[argmax(ap)] == "Act" "Expected Act action"

    # Test case 2: Dry + Very_Low + Localized + Decreasing + High agreement
    # Should give Minimal risk
    rp2, ap2 = infer_direct(1, 1, 1, 1, 3, risk_cpt, action_cpt)
    @info "Test 2 (best case):" risk=RISK_STATES[argmax(rp2)] action=ACTION_STATES[argmax(ap2)]
    @assert RISK_STATES[argmax(rp2)] == "Minimal" "Expected Minimal risk"
    @assert ACTION_STATES[argmax(ap2)] == "Monitor" "Expected Monitor action"

    # Test case 3: Normal + Medium + Moderate + Stable + Medium agreement
    rp3, ap3 = infer_direct(2, 3, 2, 2, 2, risk_cpt, action_cpt)
    @info "Test 3 (moderate):" risk=RISK_STATES[argmax(rp3)] action=ACTION_STATES[argmax(ap3)]

    # Test case 4: Low agreement should spread probabilities
    rp_high, _ = infer_direct(3, 3, 2, 2, 3, risk_cpt, action_cpt)
    rp_low, _ = infer_direct(3, 3, 2, 2, 1, risk_cpt, action_cpt)
    entropy_high = -sum(p * log(max(p, 1e-10)) for p in rp_high)
    entropy_low  = -sum(p * log(max(p, 1e-10)) for p in rp_low)
    @info "Test 4 (agreement effect):" entropy_high entropy_low
    @assert entropy_low > entropy_high "Low agreement should increase entropy"

    # Test case 5: entropy confidence.  Flat posterior scores 0, one-hot scores
    # 1, and -- the property that matters operationally -- the SAME evidence
    # under low ensemble agreement must score lower than under high agreement.
    # The old `maximum(action_probs)` could not see that: it read a deprecated
    # node and was blind to how spread the risk belief actually was.
    @assert posterior_confidence(fill(0.2, 5)) < 1e-9 "flat posterior must score zero confidence"
    @assert posterior_confidence(onehot(3, 5)) > 1.0 - 1e-9 "one-hot posterior must score full confidence"
    conf_mid = posterior_confidence([0.0, 0.0, 0.1, 0.1, 0.8])
    @assert 0.0 < conf_mid < 1.0 "a peaked-but-spread posterior sits strictly between"
    @assert posterior_confidence(rp_low) < posterior_confidence(rp_high) "Low agreement must lower confidence"
    @info "Test 5 (entropy confidence):" flat=posterior_confidence(fill(0.2, 5)) peaked=round(conf_mid, digits=3) conf_low=round(posterior_confidence(rp_low), digits=3) conf_high=round(posterior_confidence(rp_high), digits=3)

    @info "All self-tests passed!"
end

# ============================================================================
# FAST SOFT-EVIDENCE INFERENCE (tensor contraction — for bulk storyline runs)
# ============================================================================

"""
Direct tensor contraction for soft evidence. Mathematically identical to
RxInfer's message passing but runs in O(prod(state_sizes)) ≈ 22k ops per
boundary — sub-microsecond. Used for the 115k+ per-member storyline
inferences where RxInfer's per-call overhead dominates.
"""
function infer_soft_matmul(
    ant_ev::Vector{Float64}, exc_ev::Vector{Float64},
    spa_ev::Vector{Float64}, trn_ev::Vector{Float64},
    tail_ev::Vector{Float64},
    T::Array{Float64},
    action_cpt::Matrix{Float64},
)::Tuple{Vector{Float64},Vector{Float64}}
    risk_probs = zeros(Float64, 5)
    @inbounds for tl in 1:4, tr in 1:3, sp in 1:3, ex in 1:5, ant in 1:5
        w = ant_ev[ant] * exc_ev[ex] * spa_ev[sp] * trn_ev[tr] * tail_ev[tl]
        for r in 1:5
            risk_probs[r] += T[r, ant, ex, sp, tr, tl] * w
        end
    end
    s = sum(risk_probs)
    if s > 0; risk_probs ./= s; end
    return risk_probs, action_cpt * risk_probs
end

# ============================================================================
# DYNAMIC BAYESIAN NETWORK — temporal chaining across days
# ============================================================================

"""
Blend yesterday's risk posterior with a uniform to control temporal persistence.
`decay=0.6` → 60% yesterday + 40% uniform. `decay=0.0` → no memory (static BN).
"""
function blend_temporal_prior(yesterday::Vector{Float64}; decay::Float64=0.6)::Vector{Float64}
    v = decay .* yesterday .+ (1.0 - decay) .* fill(0.2, 5)
    return v ./ sum(v)
end

"""
Run the BN as a Dynamic Bayesian Network across a sequence of daily input CSVs.
Yesterday's risk posterior is fed as soft virtual evidence on `risk_data` at
time `t`, implementing the temporal link P(risk_t | evidence_t, risk_{t-1}).

The `lookback` parameter controls how many days of accumulated posterior to
carry forward (default 7, matching the forecast horizon). After `lookback` days
the chain resets to a uniform prior.

Returns a single long-format DataFrame with all days.
"""
function run_dbn_sequence(
    input_csvs::Vector{String};
    include_tail_risk::Bool=true,
    cost_loss_ratio::Float64=0.2,
    temporal_decay::Float64=0.6,
    lookback::Int=7,
)
    T = build_risk_cpt_tensor(; include_tail_risk)
    action_cpt = build_action_cpt()

    # boundary_id → yesterday's risk posterior
    prev = Dict{String, Vector{Float64}}()
    # boundary_id → how many consecutive days of posterior we've chained
    chain_len = Dict{String, Int}()

    all_frames = DataFrames.DataFrame[]

    for (day_idx, csv_path) in enumerate(input_csvs)
        df = CSV.read(csv_path, DataFrames.DataFrame)
        colnames = names(df)
        has_ratio = "ens_max_ratio" in colnames
        _soft(prefix, k, row) = all("$(prefix)_p$i" in colnames for i in 1:k) ?
            Float64[row["$(prefix)_p$i"] for i in 1:k] : nothing

        target_date = "target_date" in colnames ? string(df[1, :target_date]) : "day_$day_idx"

        n = DataFrames.nrow(df)
        out_rows = Vector{NamedTuple}(undef, n)

        for (i, row) in enumerate(DataFrames.eachrow(df))
            bid = String(row.id)

            ant_idx = categorize_antecedent(Float64(row.antecedent_rainfall_mm))
            exc_idx = categorize_exceedance(Float64(row.ecmwf_eprob_heavy))
            spa_idx = categorize_spatial(Float64(row.spatial_coverage))
            tre_idx = categorize_trend(String(row.rainfall_trend))
            tl_idx  = has_ratio ? categorize_tail_risk(Float64(row.ens_max_ratio)) : 1

            ant_ev = something(_soft("ant", 5, row), onehot(ant_idx, 5))
            exc_ev = something(_soft("exc", 5, row), onehot(exc_idx, 5))
            spa_ev = something(_soft("spa", 3, row), onehot(spa_idx, 3))
            trn_ev = something(_soft("trn", 3, row), onehot(tre_idx, 3))
            tail_ev = include_tail_risk ?
                      something(_soft("tail", 4, row), onehot(tl_idx, 4)) :
                      onehot(1, 4)

            # Temporal prior from yesterday
            yesterday = get(prev, bid, nothing)
            cl = get(chain_len, bid, 0)
            if yesterday !== nothing && cl < lookback
                risk_ev = blend_temporal_prior(yesterday; decay=temporal_decay)
            else
                risk_ev = nothing  # reset or first day
            end

            # Inference via fast matmul (exact, handles soft evidence)
            risk_probs, action_probs = infer_soft_matmul(
                ant_ev, exc_ev, spa_ev, trn_ev, tail_ev, T, action_cpt)

            # Apply temporal prior as multiplicative virtual evidence
            if risk_ev !== nothing
                risk_probs .*= risk_ev
                s = sum(risk_probs)
                if s > 0; risk_probs ./= s; end
                action_probs = action_cpt * risk_probs
            end

            # Store for tomorrow
            prev[bid] = copy(risk_probs)
            chain_len[bid] = (yesterday !== nothing ? cl + 1 : 1)

            crma_idx, crma_expl = compute_crma_state(risk_probs; cost_loss_ratio)

            out_rows[i] = (
                target_date     = target_date,
                dbn_day         = day_idx,
                boundary_id     = bid,
                boundary_name   = String(row.name),
                country         = String(row.country),
                risk_level      = RISK_STATES[argmax(risk_probs)],
                crma_state      = CRMA_STATES[crma_idx],
                traffic_light   = TRAFFIC_LIGHT[CRMA_STATES[crma_idx]],
                crma_explanation = crma_expl,
                risk_minimal    = risk_probs[1],
                risk_low        = risk_probs[2],
                risk_moderate   = risk_probs[3],
                risk_high       = risk_probs[4],
                risk_extreme    = risk_probs[5],
                temporal_prior  = risk_ev !== nothing,
                p_high_extreme  = risk_probs[4] + risk_probs[5],
            )
        end
        push!(all_frames, DataFrames.DataFrame(out_rows))
        @info "DBN day $day_idx ($target_date): $(n) boundaries"
    end
    return vcat(all_frames...)
end

# ============================================================================
# STORYLINE SELECTION — per-member BN + worst/median/best picker
# ============================================================================

"""
Run the BN on per-member evidence CSV (one row per boundary × member).
Each member gets its own exceedance, spatial coverage, and tail risk;
antecedent and trend are shared (IMERG observations).

Returns a DataFrame with risk posteriors per (boundary, member).
"""
function run_per_member_bn(
    member_csv::String;
    include_tail_risk::Bool=true,
    cost_loss_ratio::Float64=0.2,
)
    df = CSV.read(member_csv, DataFrames.DataFrame)
    T = build_risk_cpt_tensor(; include_tail_risk)
    action_cpt = build_action_cpt()
    colnames = names(df)

    _soft(prefix, k, row) = all("$(prefix)_p$i" in colnames for i in 1:k) ?
        Float64[row["$(prefix)_p$i"] for i in 1:k] : nothing

    n = DataFrames.nrow(df)
    out = Vector{NamedTuple}(undef, n)

    for (i, row) in enumerate(DataFrames.eachrow(df))
        ant_idx = categorize_antecedent(Float64(row.antecedent_rainfall_mm))
        exc_idx = categorize_exceedance(Float64(row.member_exc_frac))
        spa_idx = categorize_spatial(Float64(row.member_spa_cov))
        tre_idx = categorize_trend(String(row.rainfall_trend))
        tl_idx  = categorize_tail_risk(Float64(row.member_max_ratio))

        ant_ev  = something(_soft("ant", 5, row), onehot(ant_idx, 5))
        exc_ev  = something(_soft("exc", 5, row), onehot(exc_idx, 5))
        spa_ev  = something(_soft("spa", 3, row), onehot(spa_idx, 3))
        trn_ev  = something(_soft("trn", 3, row), onehot(tre_idx, 3))
        tail_ev = something(_soft("tail", 4, row), onehot(tl_idx, 4))

        risk_probs, _ = infer_soft_matmul(ant_ev, exc_ev, spa_ev, trn_ev, tail_ev, T, action_cpt)
        crma_idx, _ = compute_crma_state(risk_probs; cost_loss_ratio)

        out[i] = (
            boundary_id    = String(row.boundary_id),
            boundary_name  = String(row.boundary_name),
            country        = String(row.country),
            member         = String(row.member),
            target_date    = string(row.target_date),
            risk_level     = RISK_STATES[argmax(risk_probs)],
            crma_state     = CRMA_STATES[crma_idx],
            p_high_extreme = risk_probs[4] + risk_probs[5],
            risk_minimal   = risk_probs[1],
            risk_low       = risk_probs[2],
            risk_moderate  = risk_probs[3],
            risk_high      = risk_probs[4],
            risk_extreme   = risk_probs[5],
            member_max_ratio = Float64(row.member_max_ratio),
            member_exc_frac  = Float64(row.member_exc_frac),
        )
    end
    return DataFrames.DataFrame(out)
end

"""
Select worst / median / best storylines per boundary from per-member BN results.
"Worst" = member with highest P(High∪Extreme) — the highest-risk plausible future.
"""
function select_storylines(member_results::DataFrames.DataFrame)
    groups = DataFrames.groupby(member_results, [:boundary_id, :target_date])
    rows = NamedTuple[]

    for g in groups
        sorted = sort(g, :p_high_extreme, rev=true)
        n = DataFrames.nrow(sorted)
        picks = [
            ("worst",  sorted[1, :]),
            ("median", sorted[div(n, 2) + 1, :]),
            ("best",   sorted[n, :]),
        ]
        for (stype, r) in picks
            # How likely is a world at least this bad?
            n_ge = sum(sorted.p_high_extreme .>= r.p_high_extreme)
            push!(rows, (
                storyline       = stype,
                boundary_id     = r.boundary_id,
                boundary_name   = r.boundary_name,
                country         = r.country,
                target_date     = r.target_date,
                member          = r.member,
                risk_level      = r.risk_level,
                crma_state      = r.crma_state,
                p_high_extreme  = r.p_high_extreme,
                risk_minimal    = r.risk_minimal,
                risk_low        = r.risk_low,
                risk_moderate   = r.risk_moderate,
                risk_high       = r.risk_high,
                risk_extreme    = r.risk_extreme,
                member_max_ratio = r.member_max_ratio,
                probability     = round(n_ge / n, digits=3),  # P(world ≥ this bad)
                n_members       = n,
            ))
        end
    end
    return DataFrames.DataFrame(rows)
end

# ============================================================================
# CLI ENTRY POINT
# ============================================================================

"""
Parse a flag value from ARGS: --flag value → value (or nothing).
"""
function getarg(flag::String)
    i = findfirst(==(flag), ARGS)
    return i === nothing || i == length(ARGS) ? nothing : ARGS[i + 1]
end

"""
Run CSV-driven inference. Reads a prep CSV (one row per boundary) and writes
a result CSV with the full risk + action probability vectors.
"""
function run_csv(input_csv::String, output_csv::String;
                 include_agreement::Bool, include_tail_risk::Bool,
                 cost_loss_ratio::Float64=0.2,
                 use_rxinfer::Bool=true)
    df = CSV.read(input_csv, DataFrames.DataFrame)
    colnames = names(df)

    has_ratio = "ens_max_ratio" in colnames
    if include_tail_risk && !has_ratio
        @warn "--tail-risk requested but ens_max_ratio column not in CSV; disabling"
        include_tail_risk = false
    end

    # Optional soft-evidence columns: ant_p1..ant_p5, exc_p1..exc_p5,
    # spa_p1..spa_p3, trn_p1..trn_p3, tail_p1..tail_p4. All-or-nothing per node.
    _soft(prefix::String, k::Int, row) = all("$(prefix)_p$i" in colnames for i in 1:k) ?
        [Float64(row["$(prefix)_p$i"]) for i in 1:k] : nothing

    inputs = Vector{BoundaryInput}(undef, DataFrames.nrow(df))
    n_soft_rows = 0
    for (i, row) in enumerate(DataFrames.eachrow(df))
        ant_p  = _soft("ant",  5, row)
        exc_p  = _soft("exc",  5, row)
        spa_p  = _soft("spa",  3, row)
        trn_p  = _soft("trn",  3, row)
        tail_p = _soft("tail", 4, row)
        if any(x -> x !== nothing, (ant_p, exc_p, spa_p, trn_p, tail_p))
            n_soft_rows += 1
        end
        inputs[i] = BoundaryInput(
            String(row.id),
            String(row.name),
            String(row.country),
            Float64(row.antecedent_rainfall_mm),
            "",
            String(row.rainfall_trend),
            Float64(row.ecmwf_eprob_heavy),
            Float64(row.spatial_coverage),
            String(row.forecast_agreement),
            has_ratio ? Float64(row.ens_max_ratio) : 0.0,
            ant_p, exc_p, spa_p, trn_p, tail_p,
        )
    end

    backend = use_rxinfer && !include_agreement ? "RxInfer" : "matmul"
    @info "Processing $(length(inputs)) boundaries (backend=$backend agreement=$include_agreement tail_risk=$include_tail_risk C/L=$cost_loss_ratio soft_rows=$n_soft_rows)"
    results = process_all_boundaries(inputs; include_agreement, include_tail_risk,
                                      cost_loss_ratio, use_rxinfer)

    out = DataFrames.DataFrame(
        boundary_id         = [r.boundary_id for r in results],
        boundary_name       = [r.boundary_name for r in results],
        country             = [r.country for r in results],
        antecedent_category = [r.antecedent_category for r in results],
        rainfall_trend      = [r.rainfall_trend for r in results],
        risk_level          = [r.risk_level for r in results],
        crma_state          = [r.crma_state for r in results],
        traffic_light       = [r.traffic_light for r in results],
        crma_explanation    = [r.crma_explanation for r in results],
        recommended_action  = [r.recommended_action for r in results],
        confidence          = [r.confidence for r in results],
        risk_minimal        = [r.risk_probabilities[1] for r in results],
        risk_low            = [r.risk_probabilities[2] for r in results],
        risk_moderate       = [r.risk_probabilities[3] for r in results],
        risk_high           = [r.risk_probabilities[4] for r in results],
        risk_extreme        = [r.risk_probabilities[5] for r in results],
        action_monitor      = [r.action_probabilities[1] for r in results],
        action_alert        = [r.action_probabilities[2] for r in results],
        action_prepare      = [r.action_probabilities[3] for r in results],
        action_act          = [r.action_probabilities[4] for r in results],
    )

    mkpath(dirname(abspath(output_csv)))
    CSV.write(output_csv, out)
    @info "Wrote $output_csv rows=$(DataFrames.nrow(out))"

    # Brief distribution print
    risk_counts = DataFrames.combine(DataFrames.groupby(out, :risk_level), DataFrames.nrow => :n)
    @info "Risk distribution:" risk_counts
    action_counts = DataFrames.combine(DataFrames.groupby(out, :recommended_action), DataFrames.nrow => :n)
    @info "Action distribution:" action_counts
    crma_counts = DataFrames.combine(DataFrames.groupby(out, :crma_state), DataFrames.nrow => :n)
    @info "CRMA state distribution:" crma_counts
end

function main()
    if "--test" in ARGS
        self_test()
        return
    end

    input_csv = getarg("--input-csv")
    output_csv = getarg("--output-csv")
    include_agreement = !("--no-agreement" in ARGS)
    include_tail_risk = "--tail-risk" in ARGS
    cl_str = getarg("--cost-loss-ratio")
    cost_loss_ratio = cl_str === nothing ? 0.2 : parse(Float64, cl_str)
    use_rxinfer = !("--legacy-inference" in ARGS)

    if input_csv !== nothing && output_csv !== nothing
        run_csv(input_csv, output_csv; include_agreement, include_tail_risk,
                cost_loss_ratio, use_rxinfer)
        return
    end

    @info "Flood BN IBF v1 (Julia/RxInfer)"
    @info "Usage: julia flood_bn_ibf_v1.jl --input-csv IN.csv --output-csv OUT.csv [--no-agreement] [--tail-risk] [--legacy-inference] [--cost-loss-ratio 0.2]"
    @info "       julia flood_bn_ibf_v1.jl --test"

    b = BoundaryInput(
        "KEN.1", "Nairobi", "Kenya",
        45.0, "Wet", "Increasing",
        0.65, 0.4, "High", 1.5,
    )
    risk_cpt, _ = build_risk_cpt()
    action_cpt = build_action_cpt()
    result = process_boundary(b, risk_cpt, action_cpt)
    @info "Demo result:" boundary=result.boundary_id risk=result.risk_level action=result.recommended_action confidence=@sprintf("%.2f", result.confidence)

    println("\nRisk probabilities:")
    for (state, prob) in zip(RISK_STATES, result.risk_probabilities)
        bar = repeat("█", round(Int, prob * 40))
        @printf("  %-10s %5.1f%% %s\n", state, prob * 100, bar)
    end

    println("\nAction probabilities:")
    for (state, prob) in zip(ACTION_STATES, result.action_probabilities)
        bar = repeat("█", round(Int, prob * 40))
        @printf("  %-10s %5.1f%% %s\n", state, prob * 100, bar)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
