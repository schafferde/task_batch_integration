import sys
import anndata as ad
from scib.metrics.pcr import pc_regression
from multiprocessing import Pool
import numpy as np
import warnings
import time

## VIASH START
par = {
    "input": "resources_test/task_batch_integration/cxg_immune_cell_atlas/dataset.h5ad",
    "output": "output.h5ad",
    "n_comps": 50,
    "n_comps_init": 100
}
meta = {
    "name": "seurat_sel_pcr",
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
time.sleep(60*5)
adata_res = read_anndata(par["output"].replace(".h5ad", ".fromSeurat.h5ad"), obsm="obsm")
embedding = adata_res.obsm["X_emb"]
def column_pcr_reg(i):
    return pc_regression(embedding[:, i].reshape((-1,1)), adata.obs['batch'])

print(">> Compute PCR for PCA Columns", flush=True)
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)
    with Pool(20) as p:
        pcr_afters = np.asarray(p.map(column_pcr_reg, range(embedding.shape[1])))

#Note that we want to minimize pcr_after, which would maximize score
columns = np.argpartition(pcr_afters, par["n_comps"])[:par["n_comps"]]
    
print("Store output", flush=True)
output = ad.AnnData(
    obs=adata.obs[[]],
    var=adata.var[[]],
    obsm={
        "X_emb": embedding[:, columns]
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
