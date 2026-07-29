#!/bin/bash

# get the root of the directory
REPO_ROOT=$(git rev-parse --show-toplevel)

# ensure that the command below is run from the root of the repository
cd "$REPO_ROOT"

set -e

echo "Running data preprocessing"
echo "  Make sure to run 'scripts/project/build_all_docker_containers.sh'!"


RAW_DATA="resources/common/atac"
DATASET_DIR="resources/datasets/atac"


#3
nextflow run . \
  -main-script target/nextflow/workflows/process_datasets/main.nf \
  -profile docker \
  -c new_labels_ci.config \
  --input "$RAW_DATA/granja_atac/dataset.h5ad" \
  --publish_dir "$DATASET_DIR/granja_atac/log_cp10k/" \
  --output_dataset dataset.h5ad \
  --output_solution solution.h5ad \
  --output_state state.yaml \
  --atac

sed -i 's|run|granja_atac/log_cp10k|g' "$DATASET_DIR/granja_atac/log_cp10k/state.yaml"

#6
nextflow run . \
  -main-script target/nextflow/workflows/process_datasets/main.nf \
  -profile docker \
  -c new_labels_ci.config \
  --input "$RAW_DATA/trevino_atac/dataset.h5ad" \
  --publish_dir "$DATASET_DIR/trevino_atac/log_cp10k/" \
  --output_dataset dataset.h5ad \
  --output_solution solution.h5ad \
  --output_state state.yaml \
  --atac

sed -i 's|run|trevino_atac/log_cp10k|g' "$DATASET_DIR/trevino_atac/log_cp10k/state.yaml"

#7
nextflow run . \
  -main-script target/nextflow/workflows/process_datasets/main.nf \
  -profile docker \
  -c new_labels_ci.config \
  --input "$RAW_DATA/weinand_atac/dataset.h5ad" \
  --publish_dir "$DATASET_DIR/weinand_atac/log_cp10k/" \
  --output_dataset dataset.h5ad \
  --output_solution solution.h5ad \
  --output_state state.yaml \
  --atac

sed -i 's|run|weinand_atac/log_cp10k|g' "$DATASET_DIR/weinand_atac/log_cp10k/state.yaml"

#2
nextflow run . \
  -main-script target/nextflow/workflows/process_datasets/main.nf \
  -profile docker \
  -c new_labels_ci.config \
  --input "$RAW_DATA/cheong_atac/dataset.h5ad" \
  --publish_dir "$DATASET_DIR/cheong_atac/log_cp10k/" \
  --output_dataset dataset.h5ad \
  --output_solution solution.h5ad \
  --output_state state.yaml \
  --atac

sed -i 's|run|cheong_atac/log_cp10k|g' "$DATASET_DIR/cheong_atac/log_cp10k/state.yaml"

#1
nextflow run . \
  -main-script target/nextflow/workflows/process_datasets/main.nf \
  -profile docker \
  -c new_labels_ci.config \
  --input "$RAW_DATA/burkhardt_atac/dataset.h5ad" \
  --publish_dir "$DATASET_DIR/burkhardt_atac/log_cp10k/" \
  --output_dataset dataset.h5ad \
  --output_solution solution.h5ad \
  --output_state state.yaml \
  --atac

sed -i 's|run|burkhardt_atac/log_cp10k|g' "$DATASET_DIR/burkhardt_atac/log_cp10k/state.yaml"

#Stop here
#exit 0

#4
nextflow run . \
  -main-script target/nextflow/workflows/process_datasets/main.nf \
  -profile docker \
  -c new_labels_ci.config \
  --input "$RAW_DATA/kanemaru_atac/dataset.h5ad" \
  --publish_dir "$DATASET_DIR/kanemaru_atac/log_cp10k/" \
  --output_dataset dataset.h5ad \
  --output_solution solution.h5ad \
  --output_state state.yaml \
  --atac

sed -i 's|run|kanemaru_atac/log_cp10k|g' "$DATASET_DIR/kanemaru_atac/log_cp10k/state.yaml"

#5
nextflow run . \
  -main-script target/nextflow/workflows/process_datasets/main.nf \
  -profile docker \
  -c new_labels_ci.config \
  --input "$RAW_DATA/morabito_atac/dataset.h5ad" \
  --publish_dir "$DATASET_DIR/morabito_atac/log_cp10k/" \
  --output_dataset dataset.h5ad \
  --output_solution solution.h5ad \
  --output_state state.yaml \
  --atac

sed -i 's|run|morabito_atac/log_cp10k|g' "$DATASET_DIR/morabito_atac/log_cp10k/state.yaml"

#8
nextflow run . \
  -main-script target/nextflow/workflows/process_datasets/main.nf \
  -profile docker \
  -c new_labels_ci.config \
  --input "$RAW_DATA/domcke_atac/dataset.h5ad" \
  --publish_dir "$DATASET_DIR/domcke_atac/log_cp10k/" \
  --output_dataset dataset.h5ad \
  --output_solution solution.h5ad \
  --output_state state.yaml \
  --atac

sed -i 's|run|domcke_atac/log_cp10k|g' "$DATASET_DIR/domcke_atac/log_cp10k/state.yaml"

#9
nextflow run . \
  -main-script target/nextflow/workflows/process_datasets/main.nf \
  -profile docker \
  -c new_labels_ci.config \
  --input "$RAW_DATA/liang_atac/dataset.h5ad" \
  --publish_dir "$DATASET_DIR/liang_atac/log_cp10k/" \
  --output_dataset dataset.h5ad \
  --output_solution solution.h5ad \
  --output_state state.yaml \
  --atac

sed -i 's|run|liang_atac/log_cp10k|g' "$DATASET_DIR/liang_atac/log_cp10k/state.yaml"