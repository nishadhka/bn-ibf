#!/usr/bin/env julia
# drought_bn_ibf_v1_warm.jl
#
# Warm-Julia loop driver for the drought BN. Loads RxInfer + the existing
# drought_bn_ibf_v1.jl module symbols once (paying the ~95 s
# Julia + RxInfer JIT cost a single time), then iterates a manifest of
# (input_csv, output_csv) pairs in the same process — eliminating the
# per-init JIT overhead that dominates a sequential cold run.
#
# The single-init Julia entry point (drought_bn_ibf_v1.jl) is reused
# unchanged via `include` (its `if abspath(PROGRAM_FILE) == @__FILE__`
# guard prevents the demo block from running).
#
# Usage:
#     julia --project=. drought_bn_ibf_v1_warm.jl \
#         --manifest /tmp/drought_manifest.csv \
#         [--no-agreement] [--tail-risk] [--cost-loss-ratio 0.2] \
#         [--legacy-inference]
#
# Manifest format (CSV with header — one row per (init, season) pair):
#     input_csv,output_csv
#     bn_inputs_v2_backfill/drought_inputs_1981-01_MAM.csv,output_v2_notail_backfill/drought_bn_v2_notail_1981-01_MAM.csv
#     bn_inputs_v2_backfill/drought_inputs_1981-02_MAM.csv,output_v2_notail_backfill/drought_bn_v2_notail_1981-02_MAM.csv
#     ...
#
# Performance: the first call still pays the JIT cost (~95 s); every
# subsequent call is sub-second per init for 227 boundaries. For a
# 528-init backfill that's ~8 minutes of useful work + 95 s warm-up
# vs ~14 hours of accumulated JIT in a per-init shell loop.

using CSV
using DataFrames
using Printf
using Dates

# Load the v1 BN entry-point's symbols (run_csv, build_*_cpt, …) into
# this scope. The included file's main() is guarded so it does not run.
include("drought_bn_ibf_v1.jl")


function getarg(flag::AbstractString)
    idx = findfirst(==(flag), ARGS)
    if idx === nothing || idx == length(ARGS)
        return nothing
    end
    return ARGS[idx + 1]
end


function read_manifest(path::String)::DataFrame
    df = CSV.read(path, DataFrame)
    for col in (:input_csv, :output_csv)
        if !(string(col) in names(df))
            error("manifest missing required column '$col' in $path")
        end
    end
    return df
end


function main_warm()
    manifest_path = getarg("--manifest")
    if manifest_path === nothing
        error("--manifest <path> is required")
    end
    if !isfile(manifest_path)
        error("manifest not found: $manifest_path")
    end

    include_agreement = !("--no-agreement" in ARGS)
    include_tail_risk = "--tail-risk" in ARGS
    cl_str = getarg("--cost-loss-ratio")
    cost_loss_ratio = cl_str === nothing ? 0.2 : parse(Float64, cl_str)
    use_rxinfer = !("--legacy-inference" in ARGS)

    manifest = read_manifest(manifest_path)
    n = nrow(manifest)
    @info "Drought BN warm-loop driver" n_inits=n include_agreement include_tail_risk cost_loss_ratio use_rxinfer

    # Per-call timings: the first call includes JIT, the rest are
    # steady-state. We report both separately so the operator can see
    # whether the warm-loop assumption is holding.
    durations = Float64[]
    failures = Tuple{String,String}[]

    t0_total = time()
    for (i, row) in enumerate(eachrow(manifest))
        in_csv  = String(row.input_csv)
        out_csv = String(row.output_csv)

        if !isfile(in_csv)
            push!(failures, (in_csv, "missing input"))
            @warn @sprintf("[%d/%d] SKIP missing %s", i, n, in_csv)
            continue
        end
        mkpath(dirname(out_csv))

        t0 = time()
        try
            run_csv(in_csv, out_csv;
                    include_agreement=include_agreement,
                    include_tail_risk=include_tail_risk,
                    cost_loss_ratio=cost_loss_ratio,
                    use_rxinfer=use_rxinfer)
        catch e
            push!(failures, (in_csv, sprint(showerror, e)))
            @warn @sprintf("[%d/%d] FAIL %s: %s", i, n, basename(in_csv), e)
            continue
        end

        dt = time() - t0
        push!(durations, dt)
        if i == 1 || i % 25 == 0 || i == n
            @info @sprintf("[%d/%d] %.2fs  %s -> %s",
                           i, n, dt, basename(in_csv), basename(out_csv))
        end
    end

    elapsed = time() - t0_total
    @info @sprintf("Warm loop done: %d successes, %d failures, %.1f s wall",
                   length(durations), length(failures), elapsed)
    if !isempty(durations)
        first_dt = durations[1]
        if length(durations) > 1
            rest = durations[2:end]
            @info @sprintf("First call (incl JIT): %.1fs.  Steady-state: mean %.2fs / median %.2fs / max %.2fs over %d calls.",
                           first_dt,
                           sum(rest) / length(rest),
                           sort(rest)[div(length(rest)+1, 2)],
                           maximum(rest),
                           length(rest))
        else
            @info @sprintf("First (and only) call: %.1fs (no steady-state stats).", first_dt)
        end
    end

    if !isempty(failures)
        @warn "Failures:"
        for (path, msg) in failures
            @warn "  $(path): $(msg)"
        end
        exit(1)
    end
end


if abspath(PROGRAM_FILE) == @__FILE__
    main_warm()
end
