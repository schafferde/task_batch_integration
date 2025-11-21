import sys
import anndata as ad
from shannonca.dimred import reduce_scanpy
from scib.metrics.pcr import pc_regression
from multiprocessing import Pool
import numpy as np
import warnings

## VIASH START
par = {
    "input": "resources_test/task_batch_integration/cxg_immune_cell_atlas/dataset.h5ad",
    "output": "output.h5ad",
    "iters": 5,
    "n_comps": 100
}
meta = {
    "name": "sca_scale_pcr",
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

def column_pcr_reg(i):
    return pc_regression(adata.obsm['X_sca'][:, i].reshape((-1,1)), adata.obs['batch'])

print(">> Compute PCR for SCA Columns", flush=True)
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)
    with Pool(20) as p:
        pcr_afters = np.asarray(p.map(column_pcr_reg, range(par['n_comps'])))


#We are not using this
"""
#Want to reuse this, and base on existing PCA
pcr_before = pc_regression(
        adata.obsm["X_pca"],
        adata.obs['batch'],
        #Remark: this key, from OP docs, differs from hardcoded default of SCIB.pcr
        #So, without changeding (from ["pcr"]["variance"]), would recompute always?
        pca_var=adata.uns["pca_variance"],
        n_threads=20,
    )
scores = (pcr_before - pcr_afters) / pcr_before
"""    

#We flip and scale based only on pcr_after, so the highest column gets 1 and the lowest gets 0.
#Alternately, we could use the scores (already normalized and with a floor for bad columns) to scale
scores = -pcr_afters #Because lower is better
scores -= np.min(scores)
max_val = np.max(scores)
scores /= max_val if max_val > 0 else 1 #Becomes a no-op if all the same
    
print("Store output", flush=True)
output = ad.AnnData(
    obs=adata.obs[[]],
    var=adata.var[[]],
    obsm={
        "X_emb": adata.obsm['X_sca'] * scores
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
