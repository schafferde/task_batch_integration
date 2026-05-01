import anndata as ad
import scanpy as sc
import matplotlib.pyplot as plt
import sys
import numpy as np

# --- Inputs ---
FILE1 = sys.argv[1]  # Baseline Embeddings file
FILE2 = sys.argv[2]  # Scaled Embeddings File
FILE3 = sys.argv[3]  # Centered Embeddings File
FILE4 = sys.argv[4]  # Shifted Embeddings File
FILE_REF = sys.argv[5] # Input data file with cell labels
OUTPUT_FILE = sys.argv[6]
COLOR_KEY = "cell_type" # The column in .obs to color the cells by (cell_type or batch)
METHOD = "Harmony" #Name of Method

HD_EMBEDDING_KEY = "X_emb"  # The key for the high-D embedding in .obsm
UMAP_KEY = "X_umap_euc"          # The standard key where UMAP results are stored
POINT_SIZE = 5               # Small point size for large cell numbers
FIGSIZE = (14, 4) # Figure size for four side-by-side plots

# --- Helper Function to Check, Compute, and Save UMAP ---
def process_data_for_umap(file_path: str, hd_key: str, umap_key: str) -> ad.AnnData:
    """
    Loads AnnData, checks for UMAP, computes if missing, and saves the updated file.
    """
    print(f"\n--- Processing {file_path} ---")
    
    # Load the AnnData file
    try:
        adata = ad.read_h5ad(file_path)
    except FileNotFoundError:
        print(f"Error: AnnData file not found at {file_path}. Skipping UMAP check/computation.")
        # Return a placeholder or handle error appropriately if files are critical
        return None 

    # 1. Check if UMAP already exists
    if umap_key in adata.obsm:
        print(f"UMAP results found in .obsm['{umap_key}']. Skipping computation.")
        if "cell_type" not in adata.obs_keys():
            ad_ref = sc.read_h5ad(FILE_REF)
            adata.obs["cell_type"] = ad_ref.obs.cell_type
            adata.write_h5ad(file_path)
        return adata

    print(f"UMAP not found. Computing UMAP from .obsm['{hd_key}']...")

    #Update batch info
    ad_ref = sc.read_h5ad(FILE_REF)
    adata.obs["batch_full"] = ad_ref.obs.batch
    #parse formats X__b__X for MPA, b+x for TS
    adata.obs["batch"] = [x.rsplit("__",1)[0].split("__")[-1].split("+")[::-1][0] for x in adata.obs["batch_full"]]

    adata.obs["cell_type"] = ad_ref.obs.cell_type

    del ad_ref
    
    # 2. Compute UMAP if it does not exist
    if hd_key not in adata.obsm:
        raise ValueError(f"High-D embedding key '{hd_key}' not found in .obsm. Cannot compute UMAP.")
    # Compute the neighborhood graph (required for UMAP)
    sc.pp.neighbors(adata, n_neighbors=15, use_rep=hd_key, metric='euclidean', key_added="euc_neighbors")

    # Compute the UMAP projection, saving it to adata.obsm['X_umap']
    sc.tl.umap(adata, neighbors_key="euc_neighbors", key_added=umap_key)
    print(f"UMAP computation complete. Result is in .obsm['{umap_key}'].")
   
    # 3. Save the updated AnnData file
    adata.write_h5ad(file_path)
    print(f"Saved updated AnnData file to {file_path} for future use.")

    return adata

# --- Load and Process Data (Placeholder files will trigger computation) ---

adata1 = process_data_for_umap(FILE1, HD_EMBEDDING_KEY, UMAP_KEY)
adata2 = process_data_for_umap(FILE2, HD_EMBEDDING_KEY, UMAP_KEY)
adata3 = process_data_for_umap(FILE3, HD_EMBEDDING_KEY, UMAP_KEY)
adata4 = process_data_for_umap(FILE4, HD_EMBEDDING_KEY, UMAP_KEY)


# Handle case where file loading failed
if adata1 is None or adata2 is None or adata3 is None or adata4 is None:
    print("One or more AnnData objects could not be loaded/processed. Exiting plot generation.")
    exit()

#Do a little swap for plotting. This change is not saved.
adata1.obsm["X_umap"] = adata1.obsm[UMAP_KEY]
adata2.obsm["X_umap"] = adata2.obsm[UMAP_KEY]
adata3.obsm["X_umap"] = adata3.obsm[UMAP_KEY]
adata4.obsm["X_umap"] = adata4.obsm[UMAP_KEY]

# --- Plotting ---
#We first plot the UMAPs as a PNG, and then the frames/labels in an SVG
print("\n--- Generating Plot ---")
# 1. Create a figure and a set of subplots (1 row, 2 columns)
fig, axes = plt.subplots(nrows=1, ncols=4, figsize=FIGSIZE, constrained_layout=True) 
axes = axes.flatten()

# 2. Plot the first computed UMAP
# sc.pl.umap automatically plots the result in .obsm['X_umap']
sc.pl.umap(
    adata1, 
    color=COLOR_KEY, 
    #title=FILE1.split(".")[1], #f'UMAP 1 (Source: {HD_EMBEDDING_KEY})', 
    ax=axes[0], 
    show=False, 
    s=POINT_SIZE, 
    legend_loc='none' # Suppress individual plot legend
)
axes[0].set_title(METHOD + " (Baseline)", fontsize=16)

# 3. Plot the second computed UMAP
sc.pl.umap(
    adata2, 
    color=COLOR_KEY, 
    ax=axes[1], 
    show=False, 
    s=POINT_SIZE, 
    legend_loc='lower center', # Display the single legend here,
    legend_fontsize=12, legend_fontweight='normal'
)
axes[1].set_title(METHOD + " w/ BR Scale", fontsize=16)

# 3. Plot the third computed UMAP
sc.pl.umap(
    adata3, 
    color=COLOR_KEY, 
    ax=axes[2], 
    show=False, 
    s=POINT_SIZE, 
    legend_loc='none' 
)
axes[2].set_title(METHOD + " w/ BR Center", fontsize=16)
plt.tight_layout()

# 3. Plot the fourth computed UMAP
sc.pl.umap(
    adata4, 
    color=COLOR_KEY, 
    ax=axes[3], 
    show=False, 
    s=POINT_SIZE, 
    legend_loc='none' 
)
axes[3].set_title(METHOD + " w/ BR Filter", fontsize=16)
plt.tight_layout()

# Position the legend outside of the subplots
handles, labels = axes[1].get_legend_handles_labels()

#Supress legend in PNG
axes[1].get_legend().remove()


# 4. Adjust layout and save
plt.tight_layout()
plt.savefig(OUTPUT_FILE, dpi=300)
print(f"Plot saved successfully to {OUTPUT_FILE}")

# REPLOT TO GET AN SVG WITH FRAMES

fig, axes = plt.subplots(nrows=1, ncols=4, figsize=FIGSIZE, constrained_layout=True) 
sc.pl.umap(
    adata1[:10], 
    color=COLOR_KEY, 
    #title=FILE1.split(".")[1], #f'UMAP 1 (Source: {HD_EMBEDDING_KEY})', 
    ax=axes[0], 
    show=False, 
    s=POINT_SIZE, 
    legend_loc='none' # Suppress individual plot legend
)
axes[0].set_title(METHOD + " (Baseline)", fontsize=16)

# 3. Plot the second computed UMAP
sc.pl.umap(
    adata2[:10], 
    color=COLOR_KEY, 
    ax=axes[1], 
    show=False, 
    s=POINT_SIZE, 
    legend_loc='none', # Display the single legend here,
)
axes[1].set_title(METHOD + " w/ BR Scale", fontsize=16)

# 3. Plot the third computed UMAP
sc.pl.umap(
    adata3[:10], 
    color=COLOR_KEY, 
    ax=axes[2], 
    show=False, 
    s=POINT_SIZE, 
    legend_loc='none' # Display the single legend here
)
axes[2].set_title(METHOD + " w/ BR Center", fontsize=16)

# 3. Plot the fourth computed UMAP
sc.pl.umap(
    adata4[:10], 
    color=COLOR_KEY, 
    ax=axes[3], 
    show=False, 
    s=POINT_SIZE, 
    legend_loc='none' # Display the single legend here
)
axes[3].set_title(METHOD + " w/ BR Filter", fontsize=16)

fig.legend(handles, labels, loc='lower center', 
            ncol=len(labels), 
            #title="GTEx V9 Cell Type", 
            title="HypoMap Cell Type", 
            #title="Mouse Pancreas Atlas Cell Type", 
            #title="Immune Cell Atlas Batch Label",
            frameon=False, title_fontsize=16, fontsize=12, handletextpad=0.2) 
plt.tight_layout()
plt.savefig(OUTPUT_FILE.replace(".png", ".svg"))

