import sys
import anndata as ad
from scib.metrics.pcr import pc_regression
from multiprocessing import Pool
import numpy as np
import warnings
import time

## VIASH START
par = {
    'input': 'resources_test/task_batch_integration/cxg_immune_cell_atlas/dataset.h5ad',
    'output': 'output.h5ad',
    'n_hvg': 2000,
    'n_latent': 100,
    'n_hidden': 128,
    'n_layers': 2,
    'max_epochs': 400,
    'n_final': 50
}
meta = {
    'name' : 'scvi_sel_pcr',
}
## VIASH END

sys.path.append(meta["resources_dir"])
from read_anndata_partial import read_anndata

print('Read input', flush=True)
adata = read_anndata(
    par['input'],
    X='layers/counts',
    obs='obs',
    var='var',
    uns='uns'
)

if par["n_hvg"]:
    print(f"Select top {par['n_hvg']} high variable genes", flush=True)
    idx = adata.var["hvg_score"].to_numpy().argsort()[::-1][:par["n_hvg"]]
    adata = adata[:, idx].copy()

time.sleep(60*5)
#Load pre-computed data
resname = par["output"].replace(".h5ad", ".fromSCVI.npy")
results = np.load(resname)
def column_pcr_reg(i):
    return pc_regression(results[:, i].reshape((-1,1)), adata.obs['batch'])

print(">> Compute PCR for scVI Columns", flush=True)
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)
    with Pool(20) as p:
        pcr_afters = np.asarray(p.map(column_pcr_reg, range(results.shape[1])))

#Note that we want to minimize pcr_after, which would maximize score
columns = np.argpartition(pcr_afters, par["n_final"])[:par["n_final"]]
    
print("Store output", flush=True)
output = ad.AnnData(
    obs=adata.obs[[]],
    var=adata.var[[]],
    obsm={
        "X_emb": results[:, columns]
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
