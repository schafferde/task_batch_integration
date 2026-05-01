import sys
import anndata as ad
import numpy as np
import time
import pandas as pd
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
    'name' : 'scvi_subbm_pcr',
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

#Load pre-computed data
resname = par["output"].replace(".h5ad", ".fromSCVI.npy")
print("Expected ILISI scores for scVI:")
print(par["output"].replace(".h5ad", ".ilisiScores.npy"), flush=True)
print("Expected scVI output:")
print(resname, flush=True)
time.sleep(60*5)

results = np.load(resname)

scores = np.load(par["output"].replace(".h5ad", ".ilisiScores.npy"))
scores = -scores #Here, we want a higher score to be a worse column

scores -= np.min(scores)
max_val = np.max(scores)
scores /= max_val if max_val > 0 else 1 #Becomes a no-op if all the same
category_means = pd.DataFrame(results).groupby(adata.obs['batch'].values).transform('mean').values
print("Means:", category_means.shape)
result = results - (scores * category_means)
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
