import sys
import anndata as ad
import scanpy as sc
from sklearn.decomposition import NMF

## VIASH START
par = {
    "input": "resources_test/task_batch_integration/cxg_immune_cell_atlas/dataset.h5ad",
    "output": "output.h5ad",
    "n_comps": 100,
    "max_iter": 500
}
meta = {
    "name": "nmf",
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

print(">> Run NMF", flush=True)
nmf_model = NMF(n_components=par["n_comps"], init='nndsvda', random_state=0, max_iter=par["max_iter"], solver='mu')
W = nmf_model.fit_transform(adata.X) # W: cell x program matrix
print("NMF ran for", nmf_model.n_iter_, "iterations.")

print("Store output", flush=True)
output = ad.AnnData(
    obs=adata.obs[[]],
    var=adata.var[[]],
    obsm={
        "X_emb": W
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
