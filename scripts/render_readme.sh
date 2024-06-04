#!/bin/bash

set -e

[[ ! -d ../openproblems-v2 ]] && echo "You need to clone the openproblems-v2 repository next to this repository" && exit 1

../openproblems-v2/target/docker/common/create_task_readme/create_task_readme \
  --task "dge_perturbation_prediction" \
  --task_dir "src/task" \
  --github_url "https://github.com/openproblems-bio/task-dge-perturbation-prediction/tree/main/" \
  --output "README.md"
