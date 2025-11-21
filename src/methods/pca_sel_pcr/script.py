import sys
import anndata as ad
import scanpy as sc
from scib.metrics.pcr import pc_regression
from multiprocessing import Pool
import numpy as np
import warnings

## VIASH START
par = {
    "input": "resources_test/task_batch_integration/cxg_immune_cell_atlas/dataset.h5ad",
    "output": "output.h5ad",
    "n_comps": 100,
    "n_comps_init": 300
}
meta = {
    "name": "pca_sel_pcr",
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

print(">> Run PCA", flush=True)
sc.pp.pca(adata, n_comps=par["n_comps_init"])

def column_pcr_reg(i):
    return pc_regression(adata.obsm['X_pca'][:, i].reshape((-1,1)), adata.obs['batch'])

print(">> Compute PCR for PCA Columns", flush=True)
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)
    with Pool(20) as p:
        pcr_afters = np.asarray(p.map(column_pcr_reg, range(par['n_comps_init'])))


#Note that we want to minimize pcr_after, which would maximize score
#columns = np.argpartition(scores, -par["n_comps"])[-par["n_comps"]:]
columns = np.argpartition(pcr_afters, par["n_comps"])[:par["n_comps"]]
    
print("Store output", flush=True)
output = ad.AnnData(
    obs=adata.obs[[]],
    var=adata.var[[]],
    obsm={
        "X_emb": adata.obsm['X_pca'][:, columns]
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
