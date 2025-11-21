#I used I/O from the proposed addition of Seurat:
#https://github.com/openproblems-bio/task_batch_integration/blob/add-seurat-methods/src/methods/seurat_cca/script.R
#However, I used Seurat v5 commands for the actual computation as described:
# https://satijalab.org/seurat/articles/integration_introduction#perform-integration 

cat("Loading dependencies\n")
suppressPackageStartupMessages({
  requireNamespace("anndata", quietly = TRUE)
  library(Matrix, warn.conflicts = FALSE)
  library(Seurat, warn.conflicts = FALSE)
  library(SeuratObject, warn.conflicts = FALSE)
})

packageVersion("Seurat")

## VIASH START
par <- list(
  input = 'resources_test/task_batch_integration/cxg_immune_cell_atlas/dataset.h5ad',
  output = 'output.h5ad',
  dims = 100L,
  kweight=100L,
)
meta <- list(
  name = "seurat_cca"
)
## VIASH END

#Bypass, allowing one to copy output file in
#Sys.sleep(300)
#q()


cat("Read input\n")
adata <- anndata::read_h5ad(par$input)

cat("Create Seurat object using precomputed data\n")
# Extract preprocessed data
raw_data <- t(adata$layers[["counts"]])
norm_data <- t(adata$layers[["normalized"]])
obs <- adata$obs
var <- adata$var

# Convert to dgCMatrix (Seurat v5 compatibility)
if (inherits(raw_data, "dgRMatrix")) {
  dense_temp <- as.matrix(raw_data)
  raw_data <- as(dense_temp, "dgCMatrix")
}
if (inherits(norm_data, "dgRMatrix")) {
  dense_temp <- as.matrix(norm_data)
  norm_data <- as(dense_temp, "dgCMatrix")
}

# Ensure proper dimnames for other matrix types
rownames(norm_data) <- rownames(var)
colnames(norm_data) <- rownames(obs)

# Create Seurat object
seurat_obj <- CreateSeuratObject(
  counts = raw_data,
  meta.data = obs,
  assay = "RNA"
)

# In Seurat v5, we need to set the data layer for normalized data
seurat_obj[["RNA"]]$data <- norm_data

cat("Set highly variable genes from input\n")
hvg_genes <- rownames(adata$var)[adata$var$hvg]
cat("Using", length(hvg_genes), "HVGs from input dataset\n")
VariableFeatures(seurat_obj) <- hvg_genes

cat("Split by batch and perform CCA integration\n")
seurat_obj[["RNA"]] <- split(seurat_obj[["RNA"]], f = seurat_obj$batch)
seurat_obj <- ScaleData(seurat_obj)
seurat_obj <- RunPCA(seurat_obj, npcs=par$dims)

#For GTEx, we get 
#Error: k.weight (100) is set larger than the number of cells in the smallest object (40).
#Please choose a smaller k.weight.
c <- table(obs$batch)
k.weight = min(min(c), par$kweight)

seurat_obj <- IntegrateLayers(object = seurat_obj, method = CCAIntegration, orig.reduction = "pca", new.reduction = "integrated.cca", verbose = FALSE, k.weight = k.weight)
seurat_obj[["RNA"]] <- JoinLayers(seurat_obj[["RNA"]])
X_emb = Embeddings(seurat_obj, reduction = "integrated.cca")

cat("Store outputs\n")
output <- anndata::AnnData(
  obs = adata$obs,
  var = adata$var,
  obsm = list(
    X_emb = X_emb
  ),
  uns = list(
    dataset_id = adata$uns[["dataset_id"]],
    normalization_id = adata$uns[["normalization_id"]],
    method_id = meta$name
  )
)

cat("Write output to file\n")
zzz <- output$write_h5ad(par$output, compression = "gzip")

cat("Finished\n")
