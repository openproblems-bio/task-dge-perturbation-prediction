#!/bin/bash

aws s3 sync \
  s3://openproblems-data/resources/dge_perturbation_prediction/results/ \
  output/benchmark_results/ \
  --delete --dryrun
