#!/usr/bin/env bash

python3 -c "from experiments import server_run; server_run()" &

#for i in {0..6}; do
#    python -c "from experiments import run; run($i, 7)" &
#done