import sys
import anndata as ad
import scanpy as sc
import scanorama
from multiprocessing import Pool
import numpy as np
import warnings
from scib.metrics.lisi import lisi_graph_py


## VIASH START
par = {
    'input': 'resources_test/task_batch_integration/cxg_immune_cell_atlas/dataset.h5ad',
    'output': 'output.h5ad',
    'dimred': 50,
    'dimred_init': 100
}
meta = {
    'name': 'scanorama_sel_ilisi',
}
## VIASH END

sys.path.append(meta["resources_dir"])
from read_anndata_partial import read_anndata


print('Read input', flush=True)
adata = read_anndata(
    par['input'],
    X='layers/normalized',
    obs='obs',
    var='var',
    uns='uns'
)

print('Run scanorama', flush=True)
split = []
batch_categories = adata.obs['batch'].cat.categories
for b in batch_categories:
    split.append(adata[adata.obs['batch'] == b].copy())
scanorama.integrate_scanpy(split, dimred=par["dimred_init"])

#From https://colab.research.google.com/drive/1CebA3Ow4jXITK0dW5el320KVTX_szhxG
result = np.zeros((adata.shape[0], split[0].obsm["X_scanorama"].shape[1]))
for i, b in enumerate(batch_categories):
    result[adata.obs['batch'] == b] = split[i].obsm["X_scanorama"]

def column_ilisi(i):
    adata_tmp = ad.AnnData(X=result[:, i].reshape((-1,1)), obs={"batch":adata.obs['batch']})
    sc.pp.neighbors(adata_tmp, n_neighbors=15, copy=False)
    ilisi_scores = lisi_graph_py(
        adata=adata_tmp,
        obs_key='batch',
        n_cores=10,
    )
    ilisi = np.nanmedian(ilisi_scores)
    ilisi = (ilisi - 1)# / (adata.obs['batch'].nunique() - 1)
    return ilisi

print(">> Compute iLISI for Scanorama Columns", flush=True)
scores = np.asarray([column_ilisi(i) for i in range(par['dimred_init'])])

columns = np.argpartition(scores, -par["dimred"])[-par["dimred"]:]


print("Store output", flush=True)
output = ad.AnnData(
    obs=adata.obs[[]],
    var=adata.var[[]],
    uns={
        'dataset_id': adata.uns['dataset_id'],
        'normalization_id': adata.uns['normalization_id'],
        'method_id': meta['name'],
    },
    obsm={
        'X_emb': result[:, columns]
    },
    shape=adata.shape,
)

print("Write output to file", flush=True)
output.write(par['output'], compression='gzip')
