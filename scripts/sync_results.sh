#!/bin/bash

aws s3 sync \
  s3://openproblems-data/resources/perturbation_prediction/results/ \
  output/benchmark_results/ \
  --delete --dryrun

# sync back modified results
aws s3 sync \
  output/benchmark_results/ \
  s3://openproblems-data/resources/perturbation_prediction/results/ \
  --delete --dryrun

# sync one run
runid=run_2024-06-01_00-03-09; aws s3 sync \
  output/benchmark_results/${runid}/ \
  s3://openproblems-data/resources/perturbation_prediction/results/${runid}/ \
  --delete --dryrun
