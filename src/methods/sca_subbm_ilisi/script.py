import sys
import anndata as ad
from shannonca.dimred import reduce_scanpy
from scib.metrics.lisi import lisi_graph_py
import scanpy as sc
import numpy as np
import pandas as pd
## VIASH START
par = {
    "input": "resources_test/task_batch_integration/cxg_immune_cell_atlas/dataset.h5ad",
    "output": "output.h5ad",
    "iters": 5,
    "n_comps": 100
}
meta = {
    "name": "sca_subbm_ilisi",
}
## VIASH END

sys.path.append(meta["resources_dir"])
from read_anndata_partial import read_anndata
print(">> Read input", flush=True)
adata = read_anndata(
    par["input"],
    X='layers/normalized',    
    obs="obs",
    var="var",
    uns="uns"
)

print(">> Run SCA", flush=True)
reduce_scanpy(adata, keep_loadings=False, layer=None, key_added='sca', iters=par['iters'], n_comps=par['n_comps'])
def column_ilisi(i):
    adata_tmp = ad.AnnData(X=adata.obsm['X_sca'][:, i].reshape((-1,1)), obs={"batch":adata.obs['batch'].values})
    sc.pp.neighbors(adata_tmp, n_neighbors=15, copy=False)
    ilisi_scores = lisi_graph_py(
        adata=adata_tmp,
        obs_key='batch',
        n_cores=20,
    )
    ilisi = np.nanmedian(ilisi_scores)
    ilisi = (ilisi - 1)# / (adata.obs['batch'].nunique() - 1)
    return ilisi

print(">> Compute iLISI for SCA Columns", flush=True)
scores = np.asarray([column_ilisi(i) for i in range(par['n_comps'])])
scores = -scores #Here, we want a bad column to have a higher score
scores -= np.min(scores)
max_val = np.max(scores)
scores /= max_val if max_val > 0 else 1 #Becomes a no-op if all the same
embedding = adata.obsm['X_sca']
category_means = pd.DataFrame(embedding).groupby(adata.obs['batch'].values).transform('mean').values
print("Means:", category_means.shape)
result = embedding - (scores * category_means)
print("Result:", result.shape)
   
print("Store output", flush=True)
output = ad.AnnData(
    obs=adata.obs[[]],
    var=adata.var[[]],
    obsm={
        "X_emb": result
    },
    shape=adata.shape,
    uns={
        "dataset_id": adata.uns["dataset_id"],
        "normalization_id": adata.uns["normalization_id"],
        "method_id": meta["name"],
    }
)

print("Write output to file", flush=True)
output.write_h5ad(par["output"], compression="gzip")
