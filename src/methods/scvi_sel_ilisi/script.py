import sys
import anndata as ad
import scanpy as sc
from scib.metrics.lisi import lisi_graph_py
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
    'name' : 'scvi_sel_ilisi',
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

print("Processing data", flush=True)
time.sleep(60*5)
#Load pre-computed data
resname = par["output"].replace(".h5ad", ".fromSCVI.npy")
results = np.load(resname)

def column_ilisi(i):
    adata_tmp = ad.AnnData(X=results[:, i].reshape((-1,1)), obs={"batch":adata.obs['batch']})
    sc.pp.neighbors(adata_tmp, n_neighbors=15, copy=False)
    ilisi_scores = lisi_graph_py(
        adata=adata_tmp,
        obs_key='batch',
        n_cores=20,
    )
    ilisi = np.nanmedian(ilisi_scores)
    ilisi = (ilisi - 1)# / (adata.obs['batch'].nunique() - 1)
    return ilisi

print(">> Compute iLISI for scVI Columns", flush=True)
#scores = np.asarray([column_ilisi(i) for i in range(results.shape[1])])
#Read in pre-computed iLISI scores
scores = np.load(par["output"].replace(".h5ad", ".ilisiScores.npy"))

columns = np.argpartition(scores, -par["n_final"])[-par["n_final"]:]
    
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
