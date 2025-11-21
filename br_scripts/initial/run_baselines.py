import numpy as np
import scanpy as sc
import scanorama
from harmony import harmonize
from shannonca.dimred import reduce_scanpy


adata = sc.read_h5ad('large_dataset/large_data_26dataset.h5ad')
#Normalize?
adata.layers['counts']=adata.X.copy()
sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e4)
sc.pp.log1p(adata)
adata.layers["logcounts"] = adata.X.copy()

#PCA
sc.tl.pca(adata, n_comps=500)
batch_key = "batch"

# List of adata per batch
batch_cats = adata.obs.batch.cat.categories
adata_list = [adata[adata.obs.batch == b].copy() for b in batch_cats]

for d in [100, 200, 300, 500]:
    #Scanorama
    scanorama.integrate_scanpy(adata_list, dimred=d)
    adata.obsm[f"Scanorama_{d}"] = np.zeros((adata.shape[0], adata_list[0].obsm["X_scanorama"].shape[1]))
    for i, b in enumerate(batch_cats):
        adata.obsm[f"Scanorama_{d}"][adata.obs.batch == b] = adata_list[i].obsm["X_scanorama"]

    #Harmony
    adata.obsm[f"Harmony_{d}"] = harmonize(adata.obsm["X_pca"][:,:d], adata.obs, batch_key= batch_key)

    #SCA
    reduce_scanpy(adata, keep_scores=False, keep_loadings=False, keep_all_iters=True, layer=None, key_added=f'{d}_Dim_SCA', iters=5, n_comps=d)


adata.write_h5ad('large_dataset/checkpoint_allgenes_baselines.h5ad')

