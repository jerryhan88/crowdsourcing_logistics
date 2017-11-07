#!/usr/bin/env bash
for i in {0..7}; do
    qsub _cluster_run.sh $i
done