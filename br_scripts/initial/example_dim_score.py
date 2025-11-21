import scanpy as sc
from scib_metrics.benchmark import Benchmarker, BioConservation, BatchCorrection
import sys
from time import time

#Example: benchmark 100-dimensions Scanorama, 25 columns at a time

adata = sc.read_h5ad('large_dataset/checkpoint_allgenes_baselines.h5ad')


batch_key = "batch"
label_key = "cell_types"
method = "Scan" #Harm, PCA, SCA, Seurat
obsm_key = "Scanorama_100"

n = int(sys.argv[1]) #1-4, for groups of 25

#Score columns
keys = []
for i in range(25*(n-1), 25*(n)):
    key = f"{method}_dim_{i}"
    adata.obsm[key] = adata.obsm[obsm_key][:,i].reshape((-1,1))
    keys.append(key)

start = time()
print("Starting column benchmarking")
sc_bm = Benchmarker(
    adata,
    batch_key= batch_key,
    label_key= label_key,
    embedding_obsm_keys=keys,
    bio_conservation_metrics = BioConservation(),
    batch_correction_metrics = BatchCorrection(),    
    n_jobs=48
)
sc_bm.benchmark()
print("Completed column scoring with elapsed time", time() - start)

score_df = sc_bm.get_results(min_max_scale=False)
score_df.to_pickle(f'100_column_Scan_allgenes_scores_{n}_unscaled.pkl')


