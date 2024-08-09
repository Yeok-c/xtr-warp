#!/usr/bin/bash

# Build Indexes for BEIR/test (nbits=4)
BEIR=("nfcorpus" "fiqa" "scidocs" "scifact")
for dataset in "${BEIR[@]}"; do
    python utils.py index -c beir -d "$dataset" -s test -n 4
done

# Build Indexes for LoTTE.search/test (nbits=4)
LoTTE=("writing" "recreation" "science" "technology" "lifestyle" "pooled")
for dataset in "${LoTTE[@]}"; do
    python utils.py index -c lotte -d "$dataset" -t search -s test -n 4
done