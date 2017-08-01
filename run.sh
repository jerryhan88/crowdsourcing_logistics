#!/usr/bin/env bash


for i in {0..5}; do
    python -c "from experiments import run; run($i, 4)" &
done