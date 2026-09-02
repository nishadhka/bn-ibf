#!/usr/bin/env julia
# Run flood_bn_ibf_v1's BN over every Layer 4 evidence CSV in ONE Julia session.
#
#   julia --project run_layer4_batch.jl <layer4_dir> [cost_loss_ratio]
#
# Invoking the script once per initialisation costs ~90 s of Julia startup and
# RxInfer precompilation EACH TIME -- 2.3 hours for 92 days, of which the actual
# inference is a few seconds.  Including it once and looping inside the session
# pays that cost a single time.
include(joinpath(@__DIR__, "flood_bn_ibf_v1.jl"))
using Logging

dir = length(ARGS) >= 1 ? ARGS[1] : error("usage: run_layer4_batch.jl <dir> [C/L] [suffix] [tail] [pattern]")
cl   = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 0.2
sfx  = length(ARGS) >= 3 ? ARGS[3] : ""
tail = length(ARGS) >= 4 ? ARGS[4] == "true" : true
pat  = length(ARGS) >= 5 ? ARGS[5] : ""

files = sort(filter(f -> startswith(basename(f), "julia_evidence_") &&
                         endswith(f, ".csv") &&
                         (isempty(pat) || occursin(pat, basename(f))),
                    readdir(dir; join=true)))
println("Layer 4 (Julia): $(length(files)) initialisations, C/L=$cl")

for (i, f) in enumerate(files)
    init = replace(replace(basename(f), "julia_evidence_" => ""), ".csv" => "")
    out  = joinpath(dir, "julia_risk_$(init)$(sfx).csv")
    with_logger(NullLogger()) do
        run_csv(f, out; include_agreement=false, include_tail_risk=tail,
                cost_loss_ratio=cl, use_rxinfer=true)
    end
    println("[$i/$(length(files))] $init")
    flush(stdout)
end
println("done")
