import numpy as np
import scanpy as sc
#from time import time
import pickle
import scanorama
from harmony import harmonize


total_dims = 500

adata = sc.read_h5ad('large_dataset/large_data_26dataset.h5ad')
#Normalize?
adata.layers['counts']=adata.X.copy()
sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e4)
sc.pp.log1p(adata)
adata.layers["logcounts"] = adata.X.copy()

sc.tl.pca(adata, n_comps=500)
batch_key = "batch"

# List of adata per batch
batch_cats = adata.obs.batch.cat.categories
adata_list = [adata[adata.obs.batch == b].copy() for b in batch_cats]

for d in [100, 300, 500]:
    scanorama.integrate_scanpy(adata_list, dimred=d)

    adata.obsm[f"Scanorama_{d}"] = np.zeros((adata.shape[0], adata_list[0].obsm["X_scanorama"].shape[1]))
    for i, b in enumerate(batch_cats):
        adata.obsm[f"Scanorama_{d}"][adata.obs.batch == b] = adata_list[i].obsm["X_scanorama"]

    adata.obsm[f"Harmony_{d}"] = harmonize(adata.obsm["X_pca"][:,:d], adata.obs, batch_key= batch_key)


adata.write_h5ad('large_dataset/checkpoint_allgenes_scan+harm.h5ad')

