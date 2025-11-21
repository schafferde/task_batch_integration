import sys
import anndata as ad
#from scvi.model import SCVI
import time
import numpy as np

## VIASH START
par = {
    'input': 'resources_test/task_batch_integration/cxg_immune_cell_atlas/dataset.h5ad',
    'output': 'output.h5ad',
    'n_hvg': 2000,
    'n_latent': 100,
    'n_hidden': 128,
    'n_layers': 2,
    'max_epochs': 400
}
meta = {
    'name' : 'scvi',
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
#Skip actually running SCVI here
#Instead, save what would be the input
fname = par["output"].replace("h5ad", ".forSCVI.h5ad")
print(fname, flush=True)
adata.write_h5ad(fname)

print("Run scVI", flush=True)
model_kwargs = {
    key: par[key]
    for key in ["n_latent", "n_hidden", "n_layers"]
    if par[key] is not None
}

print(model_kwargs)
resname = par["output"].replace("h5ad", ".fromSCVI.npy")
print(resname, flush=True)
"""
# SCVI.setup_anndata(adata, batch_key="batch")
#vae = SCVI(adata, **model_kwargs)
#vae.train(max_epochs=par["max_epochs"], train_size=1.0)
#results = vae.get_latent_representation()
"""

time.sleep(60*45)
#Now, read in pre-computed embedding
resname = par["output"].replace(".h5ad", ".fromSCVI.npy")

results = np.load(resname)

print("Store outputs", flush=True)
output = ad.AnnData(
    obs=adata.obs[[]],
    var=adata.var[[]],
    obsm={
        "X_emb": results,
    },
    uns={
        "dataset_id": adata.uns["dataset_id"],
        "normalization_id": adata.uns["normalization_id"],
        "method_id": meta["name"],
    },
)

print("Write output to file", flush=True)
output.write_h5ad(par["output"], compression="gzip")
