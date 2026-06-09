#!/usr/bin/env julia
# run_flood_dbn_window.jl
# Generalized DBN driver: run the DBN sequence over the soft-evidence inputs in
# an arbitrary input directory and write per-day posterior CSVs into <out>/dbn/
# (the layout the parquet + DAG JSON generators consume), plus a combined file.
#
#   julia --project=. run_flood_dbn_window.jl \
#       --input-dir output/events/<key>/bn_inputs \
#       --out-dir   output/events/<key> \
#       [--pattern "_soft\.csv\$"] [--expect 15]
#
# This is the event-window generalization of run_flood_dbn_15day.jl (which was
# hardcoded to the bn_inputs/flood_inputs_2026-03-DD pattern). Same DBN params
# as the Mar-2026 run: temporal_decay=0.6, lookback=7, cost_loss_ratio=0.2,
# tail_risk on. See flood_bn_ibf_run_notes_2026-03.md (Step 3).

using CSV, DataFrames

include("flood_bn_ibf_v1.jl")

# ---- minimal --flag value arg parsing ----
function argval(flag, default)
    i = findfirst(==(flag), ARGS)
    (i === nothing || i == length(ARGS)) ? default : ARGS[i + 1]
end

input_dir = argval("--input-dir", "bn_inputs")
out_dir   = argval("--out-dir", "output")
pattern   = Regex(argval("--pattern", "_soft\\.csv\$"))
expect    = parse(Int, argval("--expect", "0"))   # 0 = no assertion

soft_csvs = sort(filter(f -> occursin(pattern, f), readdir(input_dir, join=true)))
isempty(soft_csvs) && error("no soft-evidence CSVs matching $(pattern) in $(input_dir)")
@info "DBN sequence" n=length(soft_csvs) dir=input_dir first=basename(first(soft_csvs)) last=basename(last(soft_csvs))
expect > 0 && @assert length(soft_csvs) == expect "expected $expect soft input CSVs, found $(length(soft_csvs))"

dbn = run_dbn_sequence(soft_csvs;
    include_tail_risk = true,
    cost_loss_ratio   = 0.20,
    temporal_decay    = 0.60,
    lookback          = 7,
)

mkpath(joinpath(out_dir, "dbn"))
for sub in DataFrames.groupby(dbn, :target_date)
    d = string(sub[1, :target_date])[1:10]
    CSV.write(joinpath(out_dir, "dbn", "flood_bn_v1_$(d).csv"), DataFrames.DataFrame(sub))
end
combined = joinpath(out_dir, "flood_bn_v1_dbn_window.csv")
CSV.write(combined, dbn)

ndays = length(unique(dbn.target_date))
@info "wrote per-day files" dir=joinpath(out_dir, "dbn") combined=combined ndays=ndays rows=nrow(dbn)
