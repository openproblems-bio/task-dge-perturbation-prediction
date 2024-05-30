#!/bin/bash

aws s3 sync \
  s3://openproblems-data/resources/dge_perturbation_prediction/results/ \
  output/benchmark_results/ \
  --delete --dryrun

aws s3 sync \
  output/benchmark_results/ \
  s3://openproblems-data/resources/dge_perturbation_prediction/results/ \
  --delete --dryrun