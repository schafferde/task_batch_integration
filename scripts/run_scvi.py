import torchvision
import sys
import scanpy as sc
from scvi.model import SCVI
import numpy as np

adata = sc.read_h5ad(sys.argv[1])

print("Run scVI", flush=True)
model_kwargs = {'n_latent': 100, 'n_hidden': 128, 'n_layers': 2}

SCVI.setup_anndata(adata, batch_key="batch")
vae = SCVI(adata, **model_kwargs)
vae.train(max_epochs=None, train_size=1.0)
results = vae.get_latent_representation()

np.save(sys.argv[2], results)
