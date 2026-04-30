import concord as ccd
import scanpy as sc
import anndata as ad
import torch

in_adata = sys.argv[1]
out_adata = sys.argv[2]
dim = sys.argv[3]
adata = sc.read_h5ad(in_adata)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
adata.X = adata.layers['normalized']
cur_ccd = ccd.Concord(adata=adata, domain_key='batch', latent_dim=dim, device=device, preload_dense=True)
cur_ccd.fit_transform(output_key='Concord')
output = ad.AnnData(obs=adata.obs[[]], 
                    var=adata.var[[]], 
                    obsm={"X_emb": adata.obsm["Concord"]}, 
                    uns={"dataset_id": adata.uns["dataset_id"], 
                         "normalization_id": adata.uns["normalization_id"], 
                         "method_id": "concord"}
                    )
output.write_h5ad(out_adata, compression="gzip")


