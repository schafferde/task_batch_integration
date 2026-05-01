import sys
import anndata as ad
import time
import numpy as np
import pandas as pd

## VIASH START
par = {
    "input": "resources_test/task_batch_integration/cxg_immune_cell_atlas/dataset.h5ad",
    "output": "output.h5ad",
}
meta = {
    "name": "concord_subbm_ilisi",
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

print("Expected CONCORD output:")
print(par["output"].replace(".h5ad", ".fromConcord.h5ad"), flush=True)
print("Expected CONCORD iLISI scores:")
print(par["output"].replace(".h5ad", ".ilisiScores.npy"), flush=True)
time.sleep(60*5)
adata_res = read_anndata(par["output"].replace(".h5ad", ".fromConcord.h5ad"), obsm="obsm")
embedding = adata_res.obsm["X_emb"]


scores = np.load(par["output"].replace(".h5ad", ".ilisiScores.npy"))
scores = -scores #Here, we want a higher score to be a worse column
scores -= np.min(scores)
max_val = np.max(scores)
scores /= max_val if max_val > 0 else 1 #Becomes a no-op if all the same
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
