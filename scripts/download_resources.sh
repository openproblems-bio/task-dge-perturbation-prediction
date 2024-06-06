#!/bin/bash

set -e

echo ">> Downloading resources"
# aws s3 sync --no-sign-request \
#   "s3://openproblems-bio/public/neurips-2023-competition/workflow-resources/" \
#   "resources" \
#   --delete

viash run src/common/sync_test_resources/config.vsh.yaml -- \
  --input "s3://openproblems-bio/public/neurips-2023-competition/workflow-resources/" \
  --output "resources" \
  --delete
