#!/usr/bin/env julia
# run_flood_dbn_15day.jl
# Driver: run the DBN sequence over the Mar 1-15 2026 soft-evidence inputs and
# write per-day posterior CSVs into output/dbn/ (the layout the parquet + DAG
# JSON generators consume), plus a combined 15-day file.
#
#   julia --project=. run_flood_dbn_15day.jl
#
# Mirrors the Step-3 procedure in flood_bn_ibf_run_notes_2026-03.md
# (temporal_decay=0.6, lookback=7, cost_loss_ratio=0.2, tail_risk on).

using CSV, DataFrames

include("flood_bn_ibf_v1.jl")

soft_csvs = sort(filter(f -> occursin(r"flood_inputs_2026-03-\d\d_soft\.csv$", f),
                        readdir("bn_inputs", join=true)))
@info "DBN sequence over $(length(soft_csvs)) days" first=basename(first(soft_csvs)) last=basename(last(soft_csvs))
@assert length(soft_csvs) == 15 "expected 15 soft input CSVs, found $(length(soft_csvs))"

dbn = run_dbn_sequence(soft_csvs;
    include_tail_risk = true,
    cost_loss_ratio   = 0.20,
    temporal_decay    = 0.60,
    lookback          = 7,
)

mkpath("output/dbn")
for sub in DataFrames.groupby(dbn, :target_date)
    d = string(sub[1, :target_date])[1:10]
    CSV.write("output/dbn/flood_bn_v1_$(d).csv", DataFrames.DataFrame(sub))
end
CSV.write("output/flood_bn_v1_dbn_15day.csv", dbn)

ndays = length(unique(dbn.target_date))
@info "wrote $ndays per-day files → output/dbn/  + combined output/flood_bn_v1_dbn_15day.csv  (rows=$(nrow(dbn)))"
