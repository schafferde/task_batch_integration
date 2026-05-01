import sys
import anndata as ad
import scanpy as sc
#import concord as ccd
#import torch
import time
import warnings
from multiprocessing import Pool
import numpy as np
from scib.metrics.pcr import pc_regression

## VIASH START
par = {
    "input": "resources_test/task_batch_integration/cxg_immune_cell_atlas/dataset.h5ad",
    "output": "output.h5ad",
    "r2_thresh": 0.01,
}
meta = {
    "name": "concord_selfix01_pcr",
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
#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#cur_ccd = ccd.Concord(adata=adata, domain_key='batch', device=device, preload_dense=True)
#cur_ccd.fit_transform(output_key='Concord')
#embedding = adata.obsm["Concord"]
print(par["output"].replace(".h5ad", ".fromConcord.h5ad"), flush=True)
time.sleep(60*4)
adata_res = read_anndata(par["output"].replace(".h5ad", ".fromConcord.h5ad"), obsm="obsm")
embedding = adata_res.obsm["X_emb"]
def column_pcr_reg(i):
    return pc_regression(embedding[:, i].reshape((-1,1)), adata.obs['batch'])

print(">> Compute PCR for NMF Columns", flush=True)
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)
    with Pool(20) as p:
        pcr_afters = np.asarray(p.map(column_pcr_reg, range(embedding.shape[1])))

#Alternately, we could use the scores (already normalized and with a floor for bad columns) to scale
#Note that we want the lowest-scoring columns
columns = pcr_afters < par["r2_thresh"]
with open(par["output"].rsplit("/", 1)[0]+f"_{meta['name']}_columns.txt", "w") as f:
    print("Mean score", np.mean(pcr_afters), file=f)
    print("Median score", np.median(pcr_afters), file=f)
    print("Initial columns", columns.shape[0], file=f)
    print("Columns kept with", par['r2_thresh'], np.sum(columns), file=f)


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
