import numpy as np
import scanpy as sc
from scipy.stats import describe
from scib_metrics.benchmark import Benchmarker, BioConservation, BatchCorrection
from time import time
import pickle
from shannonca.dimred import reduce_scanpy
import sys


start = time()
"""
adata = sc.read_h5ad('large_dataset/large_data_26dataset.h5ad')
print("Data loaded with elapsed time", time() - start)
#Normalize?
adata.layers['counts']=adata.X.copy()
sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e4)
sc.pp.log1p(adata)
adata.layers["logcounts"] = adata.X.copy()
print("Data normalized with elapsed time", time() - start)

#Keep only highly variable genes
sc.pp.highly_variable_genes(adata, n_top_genes=2000, batch_key= batch_key)
adata = adata[:, adata.var.highly_variable].copy()
print("Subset to HVGs with elapsed time", time() - start)

#Initial PCA
sc.tl.pca(adata, n_comps=200)
print("Completed SCA with elapsed time", time() - start)

adata.write_h5ad('large_dataset/checkpoint1.h5ad')
print("Data saved with elapsed time", time() - start)
"""
total_dims = 500
"""
adata = sc.read_h5ad('large_dataset/checkpoint1.h5ad')
reduce_scanpy(adata, keep_scores=False, keep_loadings=False, keep_all_iters=False, layer=None, key_added='High_Dim_SCA', iters=1, n_comps=total_dims)


print("Completed SCA with elapsed time", time() - start) #667

adata.write_h5ad('large_dataset/checkpoint_sca1.h5ad')
"""
#adata = sc.read_h5ad('large_dataset/checkpoint_sca5.h5ad')
adata = sc.read_h5ad('large_dataset/checkpoint_allgenes_sca5+nmf.h5ad')


batch_key = "batch"
label_key = "cell_types"

n = int(sys.argv[1])

#Score columns
keys = []
for i in range(50*(n-1), 50*(n)):
    key = f"SCA_dim_{i}"
    adata.obsm[key] = adata.obsm["X_High_Dim_SCA_5"][:,i].reshape((-1,1))
    keys.append(key)

print("Starting column benchmarking")
sc_bm = Benchmarker(
    adata,
    batch_key= batch_key,
    label_key= label_key,
    embedding_obsm_keys=keys,
    bio_conservation_metrics = BioConservation(),
    batch_correction_metrics = BatchCorrection(),    
    n_jobs=48
)
sc_bm.benchmark()
print("Completed column scoring with elapsed time", time() - start)

score_df = sc_bm.get_results(min_max_scale=False)
score_df.to_pickle(f'500_column_sca5_allgenes_scores_{n}_really_unscaled.pkl') #elapsed time ?


