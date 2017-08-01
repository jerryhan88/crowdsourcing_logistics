#!/usr/bin/env bash


for i in {0..6}; do
    python -c "from experiments import run; run($i, 7)" &
done