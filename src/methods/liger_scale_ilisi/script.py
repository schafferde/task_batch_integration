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
    "n_comps": 100
}
meta = {
    "name": "liger_scale_ilisi",
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
adata_res = read_anndata(par["output"].replace(".h5ad", ".fromLiger.h5ad"), obsm="obsm")
embedding = adata_res.obsm["X_emb"]
def column_ilisi(i):
    adata_tmp = ad.AnnData(X=embedding[:, i].reshape((-1,1)), obs={"batch":adata.obs['batch']})
    sc.pp.neighbors(adata_tmp, n_neighbors=15, copy=False)
    ilisi_scores = lisi_graph_py(
        adata=adata_tmp,
        obs_key='batch',
        n_cores=20,
    )
    ilisi = np.nanmedian(ilisi_scores)
    ilisi = (ilisi - 1)# / (adata.obs['batch'].nunique() - 1)
    return ilisi

print(">> Compute iLISI for LIGER Columns", flush=True)
scores = np.asarray([column_ilisi(i) for i in range(embedding.shape[1])])
np.save(par["output"].replace(".h5ad", ".ilisiScores.h5ad"), scores)
scores -= np.min(scores)
max_val = np.max(scores)
scores /= max_val if max_val > 0 else 1 #Becomes a no-op if all the same
    
print("Store output", flush=True)
output = ad.AnnData(
    obs=adata.obs[[]],
    var=adata.var[[]],
    obsm={
        "X_emb": embedding * scores
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
