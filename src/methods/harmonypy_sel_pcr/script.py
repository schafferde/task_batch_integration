import sys
import anndata as ad
import harmonypy as hm
import scanpy as sc
from scib.metrics.pcr import pc_regression
from multiprocessing import Pool
import numpy as np
import warnings

## VIASH START
par = {
    "input": "resources_test/task_batch_integration/cxg_immune_cell_atlas/dataset.h5ad",
    "output": "output.h5ad",
    "dimred_init": 100,
    "dimred": 50
}
meta = {
    "name": "harmonypy_sel_pcr",
    "resources_dir": "src/utils"
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
sc.pp.pca(adata, n_comps=par["dimred_init"])
print(">> Run harmonypy", flush=True)
print(adata.obsm["X_pca"].shape, flush=True)
out = hm.run_harmony(
  adata.obsm["X_pca"], #Overwritten by above
  adata.obs,
  "batch"
).Z_corr.transpose()

def column_pcr_reg(i):
    return pc_regression(out[:, i].reshape((-1,1)), adata.obs['batch'])

print(">> Compute PCR for Harmony Columns", flush=True)
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)
    with Pool(20) as p:
        pcr_afters = np.asarray(p.map(column_pcr_reg, range(par['dimred_init'])))


#Note that we want to minimize pcr_after, which would maximize score
columns = np.argpartition(pcr_afters, par["dimred"])[:par["dimred"]]

print("Store output", flush=True)
output = ad.AnnData(
    obs=adata.obs[[]],
    var=adata.var[[]],
    obsm={
        "X_emb": out[:, columns]
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
