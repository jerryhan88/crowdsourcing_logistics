#!/usr/bin/env bash


for i in {0..10}; do
    python -c "from experiments import run; run($i)" &
done