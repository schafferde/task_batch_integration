import sys
import anndata as ad
import scanorama
from multiprocessing import Pool
import numpy as np
import warnings
from scib.metrics.pcr import pc_regression


## VIASH START
par = {
    'input': 'resources_test/task_batch_integration/cxg_immune_cell_atlas/dataset.h5ad',
    'output': 'output.h5ad',
    'dimred': 100,
    'dimred_init': 300
}
meta = {
    'name': 'scanorama_sel_pcr',
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

def column_pcr_reg(i):
    return pc_regression(result[:, i].reshape((-1,1)), adata.obs['batch'])

print(">> Compute PCR for Scanorama Columns", flush=True)
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)
    with Pool(20) as p:
        pcr_afters = np.asarray(p.map(column_pcr_reg, range(par['dimred_init'])))


#Note that we want to minimize pcr_after, which would maximize score
columns = np.argpartition(pcr_afters, par["dimred"])[:par["dimred"]]


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
