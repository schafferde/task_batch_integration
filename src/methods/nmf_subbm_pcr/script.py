import sys
import anndata as ad
import time
import warnings
from multiprocessing import Pool
import numpy as np
from scib.metrics.pcr import pc_regression
import pandas as pd

## VIASH START
par = {
    "input": "resources_test/task_batch_integration/cxg_immune_cell_atlas/dataset.h5ad",
    "output": "output.h5ad",
}
meta = {
    "name": "nmf_subbm_pcr",
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

print("Expected NMF output:")
print(par["output"].replace(".h5ad", ".fromNMF.h5ad"), flush=True)
time.sleep(60*4)
adata_res = read_anndata(par["output"].replace(".h5ad", ".fromNMF.h5ad"), obsm="obsm")
embedding = adata_res.obsm["X_emb"]
def column_pcr_reg(i):
    return pc_regression(embedding[:, i].reshape((-1,1)), adata.obs['batch'])

print(">> Compute PCR for NMF Columns", flush=True)
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)
    with Pool(20) as p:
        pcr_afters = np.asarray(p.map(column_pcr_reg, range(embedding.shape[1])))

#We flip and scale based only on pcr_after, so the highest column gets 1 and the lowest gets 0.
#Alternately, we could use the scores (already normalized and with a floor for bad columns) to scale
scores = pcr_afters #Because lower is better
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
