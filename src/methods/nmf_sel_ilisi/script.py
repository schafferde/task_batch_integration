import sys
import anndata as ad
import scanpy as sc
from scib.metrics.lisi import lisi_graph_py
import numpy as np
import time

## VIASH START
par = {
    "input": "resources_test/task_batch_integration/cxg_immune_cell_atlas/dataset.h5ad",
    "output": "output.h5ad",
    "max_iter": 500,
    "n_comps": 67,
    "n_comps_init": 100
}
meta = {
    "name": "nmf_sel_ilisi",
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
print("Expected NMF iLISI scores:")
print(par["output"].replace(".h5ad", ".ilisiScores.npy"), flush=True)
time.sleep(60*5)
#Read in pre-computed embedding
adata_res = read_anndata(par["output"].replace(".h5ad", ".fromNMF.h5ad"), obsm="obsm")
embedding = adata_res.obsm["X_emb"]

scores = np.load(par["output"].replace(".h5ad", ".ilisiScores.npy"))
columns = np.argpartition(scores, -par["n_comps"])[-par["n_comps"]:]
    
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
