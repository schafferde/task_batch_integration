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
    "n_comps": 50,
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
#nmf_model = NMF(n_components=par["n_comps"], init='nndsvda', random_state=0, max_iter=par["max_iter"], solver='mu')
#W = nmf_model.fit_transform(adata.X) # W: cell x program matrix

time.sleep(60*5)
adata_res = read_anndata(par["output"].replace(".h5ad", ".fromNMF.h5ad"), obsm="obsm")
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

print(">> Compute iLISI for NMF Columns", flush=True)
scores = np.asarray([column_ilisi(i) for i in range(embedding.shape[1])])
np.save(par["output"].replace(".h5ad", ".ilisiScores.h5ad"), scores)
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
