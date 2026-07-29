import sys
import anndata as ad
import scanpy as sc

## VIASH START
par = {
    "input": "resources_test/task_batch_integration/cxg_immune_cell_atlas/dataset.h5ad",
    "output": "output.h5ad",
    "harmony": False,
    "br_mode": 0,
}
meta = {
    "name": "seurat",
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
    obsm="obsm",
    var="var",
    uns="uns"
)

method_name = meta["name"].split("_")[0]
poss_keys = [key for key in adata.obsm_keys() if method_name in key.lower()]
if len(poss_keys) == 0:
    print(f"No match for {method_name} found in {adata.obsm_keys()}")
    sys.exit(1)
elif len(poss_keys) > 1:
    print(f"Found multiple matches for {method_name} : {poss_keys}")
    sys.exit(1)

embedding = adata.obsm[poss_keys[0]]

if par["harmony"]:
    import harmonypy as hm
    harmony_out = hm.run_harmony(embedding, adata.obs, "batch")
    embedding = harmony_out.Z_corr

if par["br_mode"]:
    from scib.metrics.pcr import pc_regression
    from multiprocessing import Pool
    import numpy as np
    import warnings
    print(">> Compute PCR for Harmony Columns", flush=True)

    def column_pcr_reg(i):
        return pc_regression(embedding[:, i].reshape((-1,1)), adata.obs['batch'])
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        with Pool(20) as p:
            pcr_afters = np.asarray(p.map(column_pcr_reg, range(embedding.shape[1])))

    if par["br_mode"] > 0:
        columns = np.argpartition(pcr_afters, par["br_mode"])[:par["br_mode"]]
        embedding = embedding[:, columns]
    elif par["br_mode"] == -1:
        scores = -pcr_afters #Because lower is better
        scores -= np.min(scores)
        max_val = np.max(scores)
        scores /= max_val if max_val > 0 else 1 #Becomes a no-op if all the same
        embedding = embedding * scores
    elif par["br_mode"] == -2:
        import pandas as pd
        scores = pcr_afters #Note reverse to correct high-scoring columns more
        scores -= np.min(scores)
        max_val = np.max(scores)
        scores /= max_val if max_val > 0 else 1 #Becomes a no-op if all the same
        #Now, have scores that are 0 (best) to 1 (worst)
        category_means = pd.DataFrame(embedding).groupby(adata.obs['batch'].values).transform('mean').values
        print("Means:", category_means.shape)
        embedding = embedding - (scores * category_means)
  
    
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
