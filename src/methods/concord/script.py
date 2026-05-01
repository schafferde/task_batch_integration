import sys
import anndata as ad
import scanpy as sc
import concord as ccd
import torch
import time

## VIASH START
par = {
    "input": "resources_test/task_batch_integration/cxg_immune_cell_atlas/dataset.h5ad",
    "output": "output.h5ad",
}
meta = {
    "name": "concord",
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

print(">> Run Concord", flush=True)
#Skip running Concord
"""
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
cur_ccd = ccd.Concord(adata=adata, domain_key='batch', device=device, preload_dense=True)
cur_ccd.fit_transform(output_key='Concord')
embedding = adata.obsm["Concord"]
"""
print("Input file:")
print(par["input"])
print("Output file to paste:")
print(par["output"].replace(".h5ad", ".fromConcord.h5ad"), flush=True)
time.sleep(60*5)
adata_res = read_anndata(par["output"].replace(".h5ad", ".fromConcord.h5ad"), obsm="obsm")
embedding = adata_res.obsm["X_emb"]



print("Store output", flush=True)
output = ad.AnnData(
    obs=adata.obs[[]],
    var=adata.var[[]],
    obsm={
        "X_emb": embedding
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
