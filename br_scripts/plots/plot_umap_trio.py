import anndata as ad
import scanpy as sc
import matplotlib.pyplot as plt
import sys

# --- Configuration ---
FILE_REF = sys.argv[4]
FILE1 = sys.argv[1]  # Replace with your first AnnData file path
FILE2 = sys.argv[2]  # Replace with your second AnnData file path
FILE3 = sys.argv[3]
HD_EMBEDDING_KEY = "X_emb"  # The key for the high-D embedding in .obsm
UMAP_KEY = "X_umap"          # The standard key where UMAP results are stored
COLOR_KEY = "batch" #"cell_type"          # The column in .obs to color the cells by
OUTPUT_FILE = sys.argv[5]
POINT_SIZE = 5               # Small point size for large cell numbers
FIGSIZE = (14, 5)            # Figure size for two side-by-side plots
METHOD = "SCA" #"Scanorama"

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
    sc.pp.neighbors(adata, n_neighbors=15, use_rep=hd_key, metric='cosine')

    # Compute the UMAP projection, saving it to adata.obsm['X_umap']
    sc.tl.umap(adata)
    print(f"UMAP computation complete. Result is in .obsm['{umap_key}'].")
   
    # 3. Save the updated AnnData file
    adata.write_h5ad(file_path)
    print(f"Saved updated AnnData file to {file_path} for future use.")

    return adata

# --- Load and Process Data (Placeholder files will trigger computation) ---

# NOTE: If your actual files don't exist, this script will fail unless
# you run the placeholder generation code from the previous response here.
# Assuming files exist or placeholder generation is integrated.

adata1 = process_data_for_umap(FILE1, HD_EMBEDDING_KEY, UMAP_KEY)
adata2 = process_data_for_umap(FILE2, HD_EMBEDDING_KEY, UMAP_KEY)
adata3 = process_data_for_umap(FILE3, HD_EMBEDDING_KEY, UMAP_KEY)


# Handle case where file loading failed
if adata1 is None or adata2 is None or adata3 is None:
    print("One or more AnnData objects could not be loaded/processed. Exiting plot generation.")
    exit()

# --- Plotting ---

print("\n--- Generating Plot ---")
# 1. Create a figure and a set of subplots (1 row, 2 columns)
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=FIGSIZE, constrained_layout=True) 
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

# 3. Plot the second computed UMAP
sc.pl.umap(
    adata3, 
    color=COLOR_KEY, 
    ax=axes[2], 
    show=False, 
    s=POINT_SIZE, 
    legend_loc='none' # Display the single legend here
)
axes[2].set_title(METHOD + " w/ BR Filter", fontsize=16)
plt.tight_layout()
plt.savefig(OUTPUT_FILE, dpi=300)


# Position the legend outside of the subplots
handles, labels = axes[1].get_legend_handles_labels()

axes[1].get_legend().remove()


##REPLOT TO GET AN SVG WITH FRAMES
# 4. Adjust layout and save
plt.tight_layout()
plt.savefig(OUTPUT_FILE, dpi=300)
print(f"Plot saved successfully to {OUTPUT_FILE}")
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=FIGSIZE, constrained_layout=True) 
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
axes[2].set_title(METHOD + " w/ BR Filter", fontsize=16)
fig.legend(handles, labels, loc='lower center', 
            #bbox_to_anchor=(0.5, -0.05),
            ncol=len(labels), 
            #title="Mouse Pancreas Atlas Study Label", 
            title="Immune Cell Atlas Batch Label",
            frameon=False, title_fontsize=16, fontsize=12, handletextpad=0.2) 
plt.tight_layout()
plt.savefig(OUTPUT_FILE.replace(".png", ".svg"))

