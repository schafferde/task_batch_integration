import sys
import anndata as ad
import harmonypy as hm
import scanpy as sc
from scib.metrics.lisi import lisi_graph_py
import numpy as np

## VIASH START
par = {
    "input": "resources_test/task_batch_integration/cxg_immune_cell_atlas/dataset.h5ad",
    "output": "output.h5ad",
    "dimred_init": 100,
    "dimred": 50
}
meta = {
    "name": "harmonypy_sel_ilisi",
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

def column_ilisi(i):
    adata_tmp = ad.AnnData(X=out[:, i].reshape((-1,1)), obs={"batch":adata.obs['batch']})
    sc.pp.neighbors(adata_tmp, n_neighbors=15, copy=False)
    ilisi_scores = lisi_graph_py(
        adata=adata_tmp,
        obs_key='batch',
        n_cores=20,
    )
    ilisi = np.nanmedian(ilisi_scores)
    ilisi = (ilisi - 1)# / (adata.obs['batch'].nunique() - 1)
    return ilisi

print(">> Compute iLISI for Harmony Columns", flush=True)
scores = np.asarray([column_ilisi(i) for i in range(par['dimred_init'])])

columns = np.argpartition(scores, -par["dimred"])[-par["dimred"]:]

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
