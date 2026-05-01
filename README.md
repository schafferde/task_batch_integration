# Batch Integration with BatchRefiner
## Description

This branch of this fork of `openprobems/task_batch_integration` contains code and materials associated with BatchRefiner:

Schäffer, D. E, Kang, H., Aksu, E. D., Edelman, D., Berger, B.: Significantly enhanced batch integration of scRNA-seq embeddings. *In preparation*

## Methods and Modifications for OpenProblems Pipeline
All of our benchmarking was done using the OpenProblems pipeline, and this repository contains our modifications:
- `src/methods/` contains updated and additional baseline methods, as well as BatchRefiner-modified methods.
    - We used a total of nine baseline methods: CONCORD, Harmony, LIGER, NMF, PCA, SCA, Scanorama, scVI, Seurat.
        - CONCORD, NMF, SCA, and Seurat CCA are new. 
        - PCA is newly added as a method with paramertized dimensions, but was previously included in the preprocessing and used as a control method.
        - LIGER and Harmony (harmonypy) are modified to produce output with paramaterized dimensions.
        - We also fixed a rare issue with the LIGER wrapper that arrises when unique cell identifiers are prefixed with batch labels contrart to expectations. 
        - CONCORD, scVI, and Seurat are modified to load in embeddings computed elsewhere, as our piepline deployment did not support GPU usage and Seurat had a long runtime in some cases.
        - Scanorama (`scanorama_integrate`) is modified to produce only embedding output; we also include a `scanorama_correct` method for corrected-count output.
        - Please see each method script for implementation details. 
        - Our pipeline-compatible implementations of SCA, as well as our split of Scanorama into two methods,
          are also available as standalone branches of this repository. 
    - The BatchRefiner methods are named as `Baseline_mode_metric`. 
        - Baseline is the baseline method (above).
        - Mode is either `scale`, `sel` (for filtering by SELecting dimensions), or `subbm` (for centering by SUBtracting Batch Means). 
        - Metric is either `pcr` or `ilisi`. The `pcr` metric corresponds to using batch $R^2$, and is so named because it uses part of the principal component reregression implementation from `scib`. 
        - BatchRefiner methods for CONCORD, LIGER, NMF, scVI, and Seurat are modified to load in embeddings computed while running the baseline methods. 
        - The `ilisi_sel` implementations for those methods are also modified to load dimension scores saved by the correspdoning `ilisi_scale` methods. Loading intermediate results is accomplished by manually moving files into the the method's working space while they are suspended.
- `src/control_methods/` contains a panel of seven control methods used to calculate empirical minimum and maximum ranges for each metric and dataset.
- `src/workflow/run_benchmark` contains the top-level script for benchmarking. We modified this to include the expanded set of methods we used. 
- `src/metrics/` contains the various OpenProblems metrics. We modified the resource usage tags of a few based on our observations and the values we chose for each tag (below).
   We also modified `kbet` to use [our slight modification to the scib implementation](https://github.com/schafferde/scib/tree/kbet_memory). It changes some data types in a deterministic preprocessing step to greatly reduce time and memory usage, especially for numberous cell types. We used `scib` version 1.1.7 for all benchmarking; we note that this modification has recently been merged into `scib` version `1.2.0`.
- `scripts/` contains some added scripts that we used for running the pipeline and associated data handling. Scripts not mentioned are included from
    the original repository and may or may not work out of the box.
    - Added: `scripts/create_resources/download_resources.sh` contains the command we used to download processed CELLxGENE-origin datasets from OpenProblems' AWS storage.
    - Added: `scripts/run_benchmark/revised_run_local.sh` runs the pipeline on the full datasets locally.
    - Added: `new_labels_ci.config`, giving limiting values for nextflow resource usage labels used by above. We tuned these values, and the resource usages of a few methods and metrics, in cases where more resources (time, CPUs, or memory) were needed. The current values and labelings of each componenet were sufficient for us to run the pipeline. In many cases, the current resource limits are likely not tight bounds.
    - Added: `scripts/generate_br_table.py` to generate a CSV of score outputs from many method runs. This script takes a list of output `score_uns.yaml` files, followed by `-o <output>csv`. It also tries to read a lookup table `kbet_lookup_table.csv` to fill in any missing KBET values in the `yaml` files, which would be marked with `-1`. If multiple values for the same metric, method, and dataset are provided, the earliest-occuring is used. 
    - Used unmodified: `scripts/project/build_all_docker_containers.sh` is used to build the pipeline before running.
    - Used unmodified: `scripts/create_resources/test_resources.sh` downloads the small test dataset.
    - Used unmodified: `scripts/run_benchmark/run_test_local.sh` runs the pipeline on the small test dataset.
    
## Data
``br_results`` contains three data files and two scripts:
- Three CSV files contain the accumulated results from runs of OpenProblems benchmarking, scaled using control metrics. In general, we renamed outputs from multiple benchmarking runs with different parameters to generate unqiue names. 
    - Relative to the names of each method as implemented (above), we renamed `seurat_cca` to `seurat` and `scanorama_integrate` to `scanorama`. 
    - Methods filtering one half, third, or fifth of dimensions are named `sel50`, `sel67`, and `sel80`, respectivly. 
    - Methods filtering with a fixed batch $R^2$ threshold are named `selfix01` and `selfix10` for thresholds of 0.01 (max. 50 dimensions filtered) and 0.10. 
    - Methods filtering based on Q3+IQR are named `seliqr`. 
    - Methods centering by subtracting scaled batch means are named `subbm`. 
- A script, `plot_benchmarking_results.py`, that generates all figure panels used to visualize OpenProblems benchmarking results. 
- A script, `example_plot_umap.py`, that demonstrates plotting side-by-side UMAPs for a baseline method and BatchRefiner approaches. 

---
## The original README from the OpenProblems repository follows below.


# Batch Integration


<!--
This file is automatically generated from the tasks's api/*.yaml files.
Do not edit this file directly.
-->

Remove unwanted batch effects from scRNA-seq data while retaining
biologically meaningful variation.

Repository:
[openproblems-bio/task_batch_integration](https://github.com/openproblems-bio/task_batch_integration)

## Description

As single-cell technologies advance, single-cell datasets are growing
both in size and complexity. Especially in consortia such as the Human
Cell Atlas, individual studies combine data from multiple labs, each
sequencing multiple individuals possibly with different technologies.
This gives rise to complex batch effects in the data that must be
computationally removed to perform a joint analysis. These batch
integration methods must remove the batch effect while not removing
relevant biological information. Currently, over 200 tools exist that
aim to remove batch effects scRNA-seq datasets \[@zappia2018exploring\].
These methods balance the removal of batch effects with the conservation
of nuanced biological information in different ways. This abundance of
tools has complicated batch integration method choice, leading to
several benchmarks on this topic \[@luecken2020benchmarking;
@tran2020benchmark; @chazarragil2021flexible; @mereu2020benchmarking\].
Yet, benchmarks use different metrics, method implementations and
datasets. Here we build a living benchmarking task for batch integration
methods with the vision of improving the consistency of method
evaluation.

In this task we evaluate batch integration methods on their ability to
remove batch effects in the data while conserving variation attributed
to biological effects. As input, methods require either normalised or
unnormalised data with multiple batches and consistent cell type labels.
The batch integrated output can be a feature matrix, a low dimensional
embedding and/or a neighbourhood graph. The respective batch-integrated
representation is then evaluated using sets of metrics that capture how
well batch effects are removed and whether biological variance is
conserved. We have based this particular task on the latest, and most
extensive benchmark of single-cell data integration methods.

## Authors & contributors

| name              | roles              |
|:------------------|:-------------------|
| Michaela Mueller  | maintainer, author |
| Malte Luecken     | author             |
| Daniel Strobl     | author             |
| Robrecht Cannoodt | contributor        |
| Scott Gigante     | contributor        |
| Kai Waldrant      | contributor        |
| Nartin Kim        | contributor        |

## API

``` mermaid
flowchart TB
  file_common_dataset("<a href='https://github.com/openproblems-bio/task_batch_integration#file-format-common-dataset'>Common Dataset</a>")
  comp_process_dataset[/"<a href='https://github.com/openproblems-bio/task_batch_integration#component-type-data-processor'>Data processor</a>"/]
  file_dataset("<a href='https://github.com/openproblems-bio/task_batch_integration#file-format-dataset'>Dataset</a>")
  file_solution("<a href='https://github.com/openproblems-bio/task_batch_integration#file-format-solution'>Solution</a>")
  comp_control_method[/"<a href='https://github.com/openproblems-bio/task_batch_integration#component-type-control-method'>Control method</a>"/]
  comp_method[/"<a href='https://github.com/openproblems-bio/task_batch_integration#component-type-method'>Method</a>"/]
  comp_process_integration[/"<a href='https://github.com/openproblems-bio/task_batch_integration#component-type-process-integration'>Process integration</a>"/]
  comp_metric[/"<a href='https://github.com/openproblems-bio/task_batch_integration#component-type-metric'>Metric</a>"/]
  file_integrated("<a href='https://github.com/openproblems-bio/task_batch_integration#file-format-integration'>Integration</a>")
  file_integrated_processed("<a href='https://github.com/openproblems-bio/task_batch_integration#file-format-processed-integration-output'>Processed integration output</a>")
  file_score("<a href='https://github.com/openproblems-bio/task_batch_integration#file-format-score'>Score</a>")
  file_common_dataset---comp_process_dataset
  comp_process_dataset-->file_dataset
  comp_process_dataset-->file_solution
  file_dataset---comp_control_method
  file_dataset---comp_method
  file_dataset---comp_process_integration
  file_solution---comp_control_method
  file_solution---comp_metric
  comp_control_method-->file_integrated
  comp_method-->file_integrated
  comp_process_integration-->file_integrated_processed
  comp_metric-->file_score
  file_integrated---comp_process_integration
  file_integrated_processed---comp_metric
```

## File format: Common Dataset

A subset of the common dataset.

Example file: `resources_test/common/cxg_immune_cell_atlas/dataset.h5ad`

Format:

<div class="small">

    AnnData object
     obs: 'cell_type', 'batch'
     var: 'hvg', 'hvg_score', 'feature_name', 'feature_id'
     obsm: 'X_pca'
     obsp: 'knn_distances', 'knn_connectivities'
     layers: 'counts', 'normalized'
     uns: 'dataset_id', 'dataset_name', 'dataset_url', 'dataset_reference', 'dataset_summary', 'dataset_description', 'dataset_organism', 'normalization_id', 'knn'

</div>

Data structure:

<div class="small">

| Slot | Type | Description |
|:---|:---|:---|
| `obs["cell_type"]` | `string` | Cell type information. |
| `obs["batch"]` | `string` | Batch information. |
| `var["hvg"]` | `boolean` | Whether or not the feature is considered to be a ‘highly variable gene’. |
| `var["hvg_score"]` | `double` | A ranking of the features by hvg. |
| `var["feature_name"]` | `string` | A human-readable name for the feature, usually a gene symbol. |
| `var["feature_id"]` | `string` | A database identifier for the feature, usually an ENSEMBL ID. |
| `obsm["X_pca"]` | `double` | The resulting PCA embedding. |
| `obsp["knn_distances"]` | `double` | K nearest neighbors distance matrix. |
| `obsp["knn_connectivities"]` | `double` | K nearest neighbors connectivities matrix. |
| `layers["counts"]` | `integer` | Raw counts. |
| `layers["normalized"]` | `double` | Normalized expression values. |
| `uns["dataset_id"]` | `string` | A unique identifier for the dataset. |
| `uns["dataset_name"]` | `string` | Nicely formatted name. |
| `uns["dataset_url"]` | `string` | (*Optional*) Link to the original source of the dataset. |
| `uns["dataset_reference"]` | `string` | (*Optional*) Bibtex reference of the paper in which the dataset was published. |
| `uns["dataset_summary"]` | `string` | Short description of the dataset. |
| `uns["dataset_description"]` | `string` | Long description of the dataset. |
| `uns["dataset_organism"]` | `string` | (*Optional*) The organism of the sample in the dataset. |
| `uns["normalization_id"]` | `string` | Which normalization was used. |
| `uns["knn"]` | `object` | (*Optional*) Supplementary K nearest neighbors data. |

</div>

## Component type: Data processor

A label projection dataset processor.

Arguments:

<div class="small">

| Name | Type | Description |
|:---|:---|:---|
| `--input` | `file` | A subset of the common dataset. |
| `--output_dataset` | `file` | (*Output*) Unintegrated AnnData HDF5 file. |
| `--output_solution` | `file` | (*Output*) Uncensored dataset containing the true labels. |
| `--hvgs` | `integer` | (*Optional*) NA. Default: `2000`. |

</div>

## File format: Dataset

Unintegrated AnnData HDF5 file.

Example file:
`resources_test/task_batch_integration/cxg_immune_cell_atlas/dataset.h5ad`

Format:

<div class="small">

    AnnData object
     obs: 'cell_type', 'batch'
     var: 'hvg', 'hvg_score', 'feature_name', 'feature_id'
     obsm: 'X_pca'
     obsp: 'knn_distances', 'knn_connectivities'
     layers: 'counts', 'normalized'
     uns: 'dataset_id', 'normalization_id', 'dataset_organism', 'knn'

</div>

Data structure:

<div class="small">

| Slot | Type | Description |
|:---|:---|:---|
| `obs["cell_type"]` | `string` | Cell type information. |
| `obs["batch"]` | `string` | Batch information. |
| `var["hvg"]` | `boolean` | Whether or not the feature is considered to be a ‘highly variable gene’. |
| `var["hvg_score"]` | `double` | A ranking of the features by hvg. |
| `var["feature_name"]` | `string` | A human-readable name for the feature, usually a gene symbol. |
| `var["feature_id"]` | `string` | A database identifier for the feature, usually an ENSEMBL ID. |
| `obsm["X_pca"]` | `double` | The resulting PCA embedding. |
| `obsp["knn_distances"]` | `double` | K nearest neighbors distance matrix. |
| `obsp["knn_connectivities"]` | `double` | K nearest neighbors connectivities matrix. |
| `layers["counts"]` | `integer` | Raw counts. |
| `layers["normalized"]` | `double` | Normalized expression values. |
| `uns["dataset_id"]` | `string` | A unique identifier for the dataset. |
| `uns["normalization_id"]` | `string` | Which normalization was used. |
| `uns["dataset_organism"]` | `string` | (*Optional*) The organism of the sample in the dataset. |
| `uns["knn"]` | `object` | Supplementary K nearest neighbors data. |

</div>

## File format: Solution

Uncensored dataset containing the true labels.

Example file:
`resources_test/task_batch_integration/cxg_immune_cell_atlas/solution.h5ad`

Format:

<div class="small">

    AnnData object
     obs: 'cell_type', 'batch'
     var: 'feature_name', 'feature_id', 'hvg', 'hvg_score', 'batch_hvg'
     obsm: 'X_pca'
     obsp: 'knn_distances', 'knn_connectivities'
     layers: 'counts', 'normalized'
     uns: 'dataset_id', 'dataset_name', 'dataset_url', 'dataset_reference', 'dataset_summary', 'dataset_description', 'dataset_organism', 'normalization_id', 'knn'

</div>

Data structure:

<div class="small">

| Slot | Type | Description |
|:---|:---|:---|
| `obs["cell_type"]` | `string` | Cell type information. |
| `obs["batch"]` | `string` | Batch information. |
| `var["feature_name"]` | `string` | A human-readable name for the feature, usually a gene symbol. |
| `var["feature_id"]` | `string` | A database identifier for the feature, usually an ENSEMBL ID. |
| `var["hvg"]` | `boolean` | Whether or not the feature is considered to be a ‘highly variable gene’. |
| `var["hvg_score"]` | `double` | A ranking of the features by hvg. |
| `var["batch_hvg"]` | `boolean` | Whether or not the feature is considered to be a batch-aware ‘highly variable gene’. |
| `obsm["X_pca"]` | `double` | The resulting PCA embedding. |
| `obsp["knn_distances"]` | `double` | K nearest neighbors distance matrix. |
| `obsp["knn_connectivities"]` | `double` | K nearest neighbors connectivities matrix. |
| `layers["counts"]` | `integer` | Raw counts. |
| `layers["normalized"]` | `double` | Normalized expression values. |
| `uns["dataset_id"]` | `string` | A unique identifier for the dataset. |
| `uns["dataset_name"]` | `string` | Nicely formatted name. |
| `uns["dataset_url"]` | `string` | (*Optional*) Link to the original source of the dataset. |
| `uns["dataset_reference"]` | `string` | (*Optional*) Bibtex reference of the paper in which the dataset was published. |
| `uns["dataset_summary"]` | `string` | Short description of the dataset. |
| `uns["dataset_description"]` | `string` | Long description of the dataset. |
| `uns["dataset_organism"]` | `string` | (*Optional*) The organism of the sample in the dataset. |
| `uns["normalization_id"]` | `string` | Which normalization was used. |
| `uns["knn"]` | `object` | Supplementary K nearest neighbors data. |

</div>

## Component type: Control method

A control method for the batch integration task.

Arguments:

<div class="small">

| Name               | Type   | Description                                    |
|:-------------------|:-------|:-----------------------------------------------|
| `--input_dataset`  | `file` | Unintegrated AnnData HDF5 file.                |
| `--input_solution` | `file` | Uncensored dataset containing the true labels. |
| `--output`         | `file` | (*Output*) An integrated AnnData dataset.      |

</div>

## Component type: Method

A method for the batch integration task.

Arguments:

<div class="small">

| Name       | Type   | Description                               |
|:-----------|:-------|:------------------------------------------|
| `--input`  | `file` | Unintegrated AnnData HDF5 file.           |
| `--output` | `file` | (*Output*) An integrated AnnData dataset. |

</div>

## Component type: Process integration

Process output from an integration method to the format expected by
metrics

Arguments:

<div class="small">

| Name | Type | Description |
|:---|:---|:---|
| `--input_dataset` | `file` | Unintegrated AnnData HDF5 file. |
| `--input_integrated` | `file` | An integrated AnnData dataset. |
| `--expected_method_types` | `string` | NA. |
| `--expected_method_types` | `string` | NA. |
| `--expected_method_types` | `string` | NA. |
| `--output` | `file` | (*Output*) An integrated AnnData dataset with additional outputs. |

</div>

## Component type: Metric

A metric for evaluating batch integration methods.

Arguments:

<div class="small">

| Name | Type | Description |
|:---|:---|:---|
| `--input_integrated` | `file` | An integrated AnnData dataset with additional outputs. |
| `--input_solution` | `file` | Uncensored dataset containing the true labels. |
| `--output` | `file` | (*Output*) Metric score file. |

</div>

## File format: Integration

An integrated AnnData dataset.

Example file:
`resources_test/task_batch_integration/cxg_immune_cell_atlas/integrated.h5ad`

Description:

Must contain at least one of:

- Feature: the corrected_counts layer
- Embedding: the X_emb obsm
- Graph: the connectivities and distances obsp

Format:

<div class="small">

    AnnData object
     obsm: 'X_emb'
     obsp: 'connectivities', 'distances'
     layers: 'corrected_counts'
     uns: 'dataset_id', 'normalization_id', 'dataset_organism', 'method_id', 'neighbors'

</div>

Data structure:

<div class="small">

| Slot | Type | Description |
|:---|:---|:---|
| `obsm["X_emb"]` | `double` | (*Optional*) Embedding output - 2D coordinate matrix. |
| `obsp["connectivities"]` | `double` | (*Optional*) Graph output - neighbor connectivities matrix. |
| `obsp["distances"]` | `double` | (*Optional*) Graph output - neighbor distances matrix. |
| `layers["corrected_counts"]` | `double` | (*Optional*) Feature output - corrected counts. |
| `uns["dataset_id"]` | `string` | A unique identifier for the dataset. |
| `uns["normalization_id"]` | `string` | Which normalization was used. |
| `uns["dataset_organism"]` | `string` | (*Optional*) The organism of the sample in the dataset. |
| `uns["method_id"]` | `string` | A unique identifier for the method. |
| `uns["neighbors"]` | `object` | (*Optional*) Supplementary K nearest neighbors data. |

</div>

## File format: Processed integration output

An integrated AnnData dataset with additional outputs.

Example file:
`resources_test/task_batch_integration/cxg_immune_cell_atlas/integrated_processed.h5ad`

Description:

Must contain at least one of:

- Feature: the corrected_counts layer
- Embedding: the X_emb obsm
- Graph: the connectivities and distances obsp

The Graph should always be present, but the Feature and Embedding are
optional.

Format:

<div class="small">

    AnnData object
     obsm: 'X_emb', 'clustering'
     obsp: 'connectivities', 'distances'
     layers: 'corrected_counts'
     uns: 'dataset_id', 'normalization_id', 'dataset_organism', 'method_id', 'neighbors'

</div>

Data structure:

<div class="small">

| Slot | Type | Description |
|:---|:---|:---|
| `obsm["X_emb"]` | `double` | (*Optional*) Embedding output - 2D coordinate matrix. |
| `obsm["clustering"]` | `integer` | Leiden clustering results at different resolutions. |
| `obsp["connectivities"]` | `double` | Graph output - neighbor connectivities matrix. |
| `obsp["distances"]` | `double` | Graph output - neighbor distances matrix. |
| `layers["corrected_counts"]` | `double` | (*Optional*) Feature output - corrected counts. |
| `uns["dataset_id"]` | `string` | A unique identifier for the dataset. |
| `uns["normalization_id"]` | `string` | Which normalization was used. |
| `uns["dataset_organism"]` | `string` | (*Optional*) The organism of the sample in the dataset. |
| `uns["method_id"]` | `string` | A unique identifier for the method. |
| `uns["neighbors"]` | `object` | Supplementary K nearest neighbors data. |

</div>

## File format: Score

Metric score file

Example file: `score.h5ad`

Format:

<div class="small">

    AnnData object
     uns: 'dataset_id', 'normalization_id', 'method_id', 'metric_ids', 'metric_values'

</div>

Data structure:

<div class="small">

| Slot | Type | Description |
|:---|:---|:---|
| `uns["dataset_id"]` | `string` | A unique identifier for the dataset. |
| `uns["normalization_id"]` | `string` | Which normalization was used. |
| `uns["method_id"]` | `string` | A unique identifier for the method. |
| `uns["metric_ids"]` | `string` | One or more unique metric identifiers. |
| `uns["metric_values"]` | `double` | The metric values obtained for the given prediction. Must be of same length as ‘metric_ids’. |

</div>

