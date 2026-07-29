#!/bin/bash

# get the root of the directory
REPO_ROOT=$(git rev-parse --show-toplevel)

# ensure that the command below is run from the root of the repository
cd "$REPO_ROOT"

set -e

echo "Running data preprocessing"
echo "  Make sure to run 'scripts/project/build_all_docker_containers.sh'!"


RAW_DATA="resources/common/others"
DATASET_DIR="resources/datasets/others"

nextflow run . \
  -main-script target/nextflow/workflows/process_datasets/main.nf \
  -profile docker \
  -c new_labels_ci.config \
  --input "$RAW_DATA/ocular_atlas/log_cp10k/dataset.h5ad" \
  --publish_dir "$DATASET_DIR/ocular_atlas/log_cp10k/" \
  --output_dataset dataset.h5ad \
  --output_solution solution.h5ad \
  --output_state state.yaml

sed -i 's|run|ocular_atlas/log_cp10k|g' "$DATASET_DIR/ocular_atlas/log_cp10k/state.yaml"

nextflow run . \
  -main-script target/nextflow/workflows/process_datasets/main.nf \
  -profile docker \
  -c new_labels_ci.config \
  --input "$RAW_DATA/celegans_embryo/log_cp10k/dataset.h5ad" \
  --publish_dir "$DATASET_DIR/celegans_embryo/log_cp10k/" \
  --output_dataset dataset.h5ad \
  --output_solution solution.h5ad \
  --output_state state.yaml

sed -i 's|run|celegans_embryo/log_cp10k|g' "$DATASET_DIR/ocular_atlas/log_cp10k/state.yaml"

nextflow run . \
  -main-script target/nextflow/workflows/process_datasets/main.nf \
  -profile docker \
  -c new_labels_ci.config \
  --input "$RAW_DATA/tabula_muris/log_cp10k/dataset.h5ad" \
  --publish_dir "$DATASET_DIR/tabula_muris/log_cp10k/" \
  --output_dataset dataset.h5ad \
  --output_solution solution.h5ad \
  --output_state state.yaml

sed -i 's|run|tabula_muris/log_cp10k|g' "$DATASET_DIR/ocular_atlas/log_cp10k/state.yaml"

