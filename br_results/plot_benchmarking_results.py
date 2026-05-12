import pandas as pd
import numpy as np
import math
from scipy import stats
import colorsys
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.patches import FancyArrowPatch
import matplotlib.colors as mc
import matplotlib.ticker as plticker
import matplotlib.transforms as mtrans
import matplotlib.ticker as mtick

# Metric lists
biocons_metrics = ["isolated_label_asw", "clisi", 
                 "nmi", "ari", "asw_label",
                 "isolated_label_f1", "cell_cycle_conservation"]
batch_metrics = ["asw_batch", "ilisi", "kbet",
                   "graph_connectivity", "pcr"]
#                    "ari_batch", "nmi_batch"]

df = pd.read_csv("final_100d_table_combined_scale6.csv")
# Define the required order for the jitter plots
all_metrics = batch_metrics + biocons_metrics
baselines = ['concord', 'harmonypy', 'liger', 'nmf', 'pca', 'sca', 'scanorama', 'scvi', 'seurat']

# --- 1. Data Preprocessing and Aggregation ---

def parse_method_name(method_name):
    """Parses method name into components: Baseline, Type, B_modifier, C_modifier."""
    parts = method_name.split('_')
    if len(parts) == 1:
        # Baseline method
        return parts[0], 'baseline', "", "" 
    elif len(parts) == 3:
        # Modified method
        baseline = parts[0]
        B = parts[1]
        C = parts[2]
        return baseline, 'modified', B, C
    return method_name, 'unknown', None, None

df_long = df.melt(
    id_vars=['Metric', 'Dataset'],
    var_name='Method',
    value_name='Value'
)

df_long[['Baseline', 'Type', 'B_modifier', 'C_modifier']] = df_long['Method'].apply(
    lambda x: pd.Series(parse_method_name(x))
)

def calculate_aggregate(df_group, metric_list):
    """Calculates the mean score for a given list of metrics."""
    if not metric_list:
        return np.nan
    score = df_group[df_group['Metric'].isin(metric_list)]['Value'].mean()
    return score

# Per-dataset aggregation (required for both per-dataset plots and global average)
df_agg = df_long.groupby(['Dataset', 'Method', 'Baseline', 'Type', 'B_modifier', 'C_modifier']).apply(
    lambda x: pd.Series({
        'batch_score': calculate_aggregate(x, batch_metrics),
        'biocons_score': calculate_aggregate(x, biocons_metrics)
    })
).reset_index()

df_agg.to_csv("agg_df.csv", index=False)

# Global Aggregate Score aggregation (for Dot Plot panel)
df_agg_global = df_agg.groupby(
    ['Method', 'Baseline', 'Type', 'B_modifier', 'C_modifier']
)[['batch_score', 'biocons_score']].mean().reset_index()

# Global Long Aggregation (Individual Metric Scores averaged across datasets for Global Jitter Plot)
df_long_global = df_long.groupby([
    'Metric', 'Method', 'Baseline', 'Type', 'B_modifier', 'C_modifier'
])['Value'].mean().reset_index()


# --- 2. Plotting Setup and Helpers (Color and Marker Maps) ---

colors_to_use = plt.get_cmap('tab10', 10).colors
#blue, orange, green, red, purple, brown, pink, grey, olive, cyan
#0     1       2      3    4       5      6     7     8      9
color_map = {
    "harmonypy": colors_to_use[3],#red
    "pca": colors_to_use[8], #olive
    "sca": colors_to_use[5], #brown
    "scanorama": colors_to_use[2], #green
    "seurat": colors_to_use[1], #orange
    "scvi": colors_to_use[0], #blue
    "liger": colors_to_use[6], #pink
    "nmf": colors_to_use[4], #purple
    "concord": colors_to_use[7], #grey
    }

print(color_map)

# Define markers based on B_modifier and C_modifier combination
def get_marker(B, C, type_):
    if type_ == 'baseline':
        return 'o' # Circle for baseline
    
    key = (nice_approach(B), C)
    marker_map_new = {
        ('scaled', 'pcr'): 's',   # Square
        ('scaled', 'ilisi'): 'D', # Triangle
        ('filtered', 'pcr'): '^',     # Diamond 
        ('filtered', 'ilisi'): 'v',   # Upside-down Triangle 
        ('centered', 'pcr'): 'P',    # Filled Plus
        ('centered', 'ilisi'): 'X',    # X marker
    }
    return marker_map_new.get(key, 'X')

def nice_dataset(dataset):
    dataset = dataset.replace("_", " ").title()
    dataset = dataset.replace("Dkd", "DKD").replace("Gtex", "GTEx").replace("Hypomap", "HypoMap").replace("Celegans", "C elegans")
    return dataset

def short_dataset(dataset):
    nice = nice_dataset(dataset)
    if "Pancreas" in nice or "Immune" in nice:
        return nice.replace(" Atlas", "")
    elif "elegans" in nice:
        return nice.replace(" Embryo", "")
    else:
        return nice

def nice_metric(metric):
    metric = metric.replace("_", " ").title()
    if metric == "Ilisi":
        return "iLISI"
    if metric == "Clisi":
        return "cLISI"
    if metric == "Pcr":
        return "PCR Comparison"
    metric = metric.replace("Asw", "ASW").replace("Nmi", "NMI").replace("Kbet", "KBET").replace("Ari", "ARI").replace("Conservation", "Cons.")
    return metric

def nice_method(method):
    if method == "harmonypy":
        return "Harmony"
    if method == "scvi":
        return "scVI"
    if method == "scanorama":
        return "Scanorama"
    if method == "seurat":
        return "Seurat"
    return method.upper()

def nice_approach(b_mod, sel_only=False):
    if b_mod == "scale":
        approach = "scaled"
    elif b_mod in ['sel50', 'sel67', 'sel80']:
        approach = "filtered"
    elif b_mod in ["selfix01", "selfix10", "seliqr"]:
        approach =  "filtered"
    elif b_mod == "subbm":
        approach = "centered"
    else:
        approach = b_mod
    return approach

def title_approach(b_mod, c_mod):
    if c_mod == "pcr":
        score = "Batch R²"
    elif c_mod == "ilisi":
        score = "iLISI"
    else:
        score = c_mod
    score = " w/ " + score
    if b_mod == "scale":
        approach = "BatchRefiner " +"Scaling (linearly)" + score
    elif b_mod == 'sel50':
        approach = "BatchRefiner " +"Filtering (½ of dims.)" + score
    elif b_mod == 'sel67':
        approach = "BatchRefiner " +"Filtering (⅓ of dims.)" + score
    elif b_mod == 'sel80':
        approach = "BatchRefiner " +"Filtering (⅕ of dims)" + score
    elif b_mod == 'selfix01':
        approach = "BR " +"Filtering [Batch R² < min(0.01, median)]"   
    elif b_mod == 'selfix10':
        approach = "BatchRefiner " +"Filtering [Batch R² < 0.1]"       
    elif b_mod == 'seliqr':
        approach = "BatchRefiner " +"Filtering [Batch R² < (Q3+IQR)]"       
    elif b_mod == 'subbm':
        approach = "BR " +"Centering (-scaled batch mean)" + score
    else:
        #approach = b_mod + score
        approach = "BR Scaling or Filtering (⅓ of dims.)" + score
    return approach

def darken_color(color, amount=0.75):
    c = colorsys.rgb_to_hls(*mc.to_rgb(color))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])    

# --- 3. Shared Legend Creation ---

def create_shared_legend(fig, filter_c, b_filter_list, arrow_space=True):
    """Creates and positions the shared figure legend using Proxy Artists, dynamically filtering shapes by C-modifier and B-modifier set."""
    
    # 1. Color Legend (Baseline Method)
    legend_elements_color = [
        mlines.Line2D([0], [0], marker='o', color='w', label=nice_method(base),
                   markerfacecolor=color_map.get(base), markersize=15)
        for base in baselines
    ]

    # 2. Shape Legend (Method Type: Dynamically filtered)
    if arrow_space:
        legend_elements_shape = [
            # Always include Baseline marker
            mlines.Line2D([0], [0], marker='o', color='w', label='', markerfacecolor='w', markersize=15),
        ]
    else:
        legend_elements_shape = []

    # Define all possible B-modifier markers based on the C-modifier
    if filter_c == 'pcr':
        marker_defs = [
            ('scaled', 's'), ('filtered', '^'), ('centered', 'P') #, ('sel3', 'X')
        ]
        label_c = " w/ Batch R²"
    elif filter_c == 'ilisi':
        marker_defs = [
            ('scaled', 'D'), ('filtered', 'v'), ('centered', 'X') #, ('sel3', '>')
        ]
        label_c = " w/ iLISI"
    else:
        marker_defs = []
        label_c = " w/ " + filter_c

    # Filter marker definitions based on the requested b_filter_list
    approaches = set([nice_approach(b_mod) for b_mod in b_filter_list])
    for approach, marker in marker_defs:
        if approach in approaches:
            # Create a proxy artist for the B-modifier
            legend_elements_shape.append(
                mlines.Line2D([0], [0], marker=marker, color='w', label=approach+label_c, markerfacecolor='k', markersize=15)
            )
    
    # 3. Arrow Legend 
    #arrow_proxy = mlines.Line2D([0], [0], color='gray', linestyle='-', linewidth=2, alpha=0.5, 
    #                            label='Arrow (Baseline $\\rightarrow$ Modified)')
    
    # Construct title and legend
    if arrow_space:
        legend_title = f"Baseline Method (Color) | BR Shift | BatchRefiner Approach (Shape)"
    else:
        legend_title = f"Baseline Method (Color) | BatchRefiner Approach (Shape)"
        
    full_legend_elements = legend_elements_color + legend_elements_shape #+ [arrow_proxy]
    labels = [e.get_label() for e in full_legend_elements]

    # Calculate required columns for the legend (Baseline colors + Shapes + Arrow)
    ncol_count = len(baselines) + len(legend_elements_shape) + 1 
    
    # Position the legend outside of the subplots
    fig.legend(full_legend_elements, labels, loc='lower center', 
               bbox_to_anchor=(0.5, -0.05),
               ncol=ncol_count, 
               title=legend_title, frameon=False, title_fontsize=16, fontsize=12, handletextpad=0.2) 
    


# --- 4. Per-Dataset Plotting Function (Aggregate) - Individual Axis Scaling Applied ---

def create_dot_plot(df_data, filter_c, b_filter_list, filename, b_arrow_list=None, mod_method_list=None, p_value=False):
    """Generates the multi-panel plot for a specific C-modifier, showing per-dataset scores, with arrows."""
    
    df_plot = df_data[
        (df_data['Type'] == 'baseline') | 
        ((df_data['C_modifier'] == filter_c) & (df_data['B_modifier'].isin(b_filter_list)))
    ].copy()
      
    datasets_to_plot = sorted(df_plot['Dataset'].unique())
    n_datasets = len(datasets_to_plot)
    if n_datasets == 0:
        print(f"No data to plot for filter_c='{filter_c}' and b_filter_list='{b_filter_list}'.")
        return

    # Use a fixed 3x3 grid 
    nrows = 3
    ncols = 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 14), constrained_layout=True) #Increased to 14x14 for 9 datasets
    axes = axes.flatten()

    b_mods_str = ", ".join(b_filter_list)
    fig.suptitle(f'Aggregate Scores (C = "{filter_c}", B = "{b_mods_str}")', fontsize=16, fontweight='bold')

    paired_by_method = {}
    # Plotting Loop
    for i, dataset in enumerate(datasets_to_plot):
        ax = axes[i]
        df_subset = df_plot[df_plot['Dataset'] == dataset]

        # --- INDIVIDUAL AXIS SCALING APPLIED HERE ---
        # Calculate local min/max for the current dataset subset
        x_offset = max(0.1, df_subset['batch_score'].max() - df_subset['batch_score'].min()) * 0.15
        y_offset = max(0.1, df_subset['biocons_score'].max() - df_subset['biocons_score'].min()) * 0.15
        x_min, x_max = df_subset['batch_score'].min() -x_offset, df_subset['batch_score'].max() +x_offset
        y_min, y_max = df_subset['biocons_score'].min() -y_offset, df_subset['biocons_score'].max() +y_offset
        
        # Handle cases where min/max might be identical (single point or zero scores)       
        if not (np.isfinite(x_min) and np.isfinite(x_max)): x_min, x_max = 0, 1.1
        if not (np.isfinite(y_min) and np.isfinite(y_max)): y_min, y_max = 0, 1.1


        # 1. Store baseline coordinates for arrows
        baseline_coords = {}
        df_baseline_subset = df_subset[df_subset['Type'] == 'baseline']
        for _, row in df_baseline_subset.iterrows():
            baseline_coords[row['Baseline']] = (row['batch_score'], row['biocons_score'])
        
            
        # 2. Plot points and collect modified coordinates
        modified_points = []
        for _, row in df_subset.iterrows():
            baseline = row['Baseline']
            method_type = row['Type']
            approach = row['B_modifier']
            if method_type == 'modified':
                if mod_method_list and baseline not in mod_method_list:
                    continue
                elif b_arrow_list and approach in b_arrow_list:
                    modified_points.append(row)
            
            color = color_map.get(baseline, 'gray')
            marker = get_marker(approach, row['C_modifier'], method_type)
            
            # Plotting (AXES SWAPPED: X=Batch, Y=Biocons)
            ax.plot(
                row['batch_score'],    # X-axis
                row['biocons_score'],  # Y-axis
                marker=marker,
                color=color, 
                markersize=15,         # Increased size
                linestyle='',
                alpha=0.8,
                zorder=2,
            )
            

        # 3. Draw arrows from baseline to modified points
        #Also calculate p-values for the approaches we draw arrows for
        for row in modified_points:
            baseline_name = row['Baseline']
            modified_x, modified_y = row['batch_score'], row['biocons_score']

            if baseline_name in baseline_coords:
                baseline_x, baseline_y = baseline_coords[baseline_name]

                if p_value:
                    if baseline_name not in paired_by_method:
                        paired_by_method[baseline_name] = {approach: []}
                    elif approach not in paired_by_method[baseline_name]:
                        paired_by_method[baseline_name][approach] = []
                    paired_by_method[baseline_name][approach].append((baseline_x, modified_x))
                
                color = color_map.get(baseline_name, 'gray')

                fancy_arrow = FancyArrowPatch(
                    posA=(baseline_x, baseline_y),
                    posB=(modified_x, modified_y),
                    arrowstyle="simple", 
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.5,
                    mutation_scale=20,
                    shrinkA=15, shrinkB=15
                )
                try:
                    ax.add_patch(fancy_arrow)
                except:
                    print("Unable to add arrow for", baseline_name)

        # Panel settings (AXIS LABELS SWAPPED)
        ax.set_title(f'Dataset: {nice_dataset(dataset)}', fontsize=16)
        ax.set_xlabel('Batch Correction Score', fontsize=16) 
        ax.set_ylabel('Bio Conservation Score', fontsize=16) 
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Apply local axis limits
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        
        ax.tick_params(axis='both', which='major', labelsize=12)

        #Prevent intervals of 0.025, observed once
        y_ticks = ax.get_yticks()
        inferred_interval = round(y_ticks[1] - y_ticks[0], 4)
        if inferred_interval == 0.025:
            print("Changing Y intervals")
            ax.yaxis.set_major_locator(plticker.MultipleLocator(base=0.04))

        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%4.2f'))
        ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%4.2f'))


        #ax.tick_params(axis='y', which='major', pad=10) 
        # Optional: Force labels to align to the right within that padded space
        #for label in ax.get_yticklabels():
        #    label.set_horizontalalignment('right')
                    
    # Remove unused subplots
    for j in range(len(datasets_to_plot), nrows * ncols):
        fig.delaxes(axes[j])

    create_shared_legend(fig, filter_c, b_filter_list)

    plt.savefig(filename, format='svg', bbox_inches='tight')
    plt.close(fig)

    p_by_approach = {}
    if p_value:
        for method, d in paired_by_method.items():
            for approach, pairs in d.items():
                base_list, mod_list = zip(*pairs)
                t, p = stats.ttest_rel(base_list, mod_list)
                print(method, "with", approach + ": p =", p)
                if approach not in p_by_approach:
                    p_by_approach[approach] = []
                p_by_approach[approach].append(p)
    for approach, pvals in p_by_approach.items():
        statistic, combined_p = stats.combine_pvalues(pvals, method='fisher')
        print("Fisher's combined p for", approach, ": p=", combined_p)


# --- 5. Global Summary Plotting Function  ---

def calculate_difference(df_agg_global, filter_c, b_filter_list):
    """Calculates the batch score difference: Modified - Baseline, filtered by C and B-list."""
    # Filter for the specific C-modifier and B-list (and include baselines for joining)
    df_filtered = df_agg_global[
        (df_agg_global['Type'] == 'baseline') | 
        ((df_agg_global['C_modifier'] == filter_c) & (df_agg_global['B_modifier'].isin(b_filter_list)))
    ].copy()

    # Get baseline scores (Type='baseline')
    df_baseline = df_filtered[df_filtered['Type'] == 'baseline'][['Baseline', 'batch_score', 'biocons_score']].rename(
        columns={'batch_score': 'baseline_batch_score', 'biocons_score': 'baseline_biocons_score'}
    )

    # Get modified scores (Type='modified')
    df_modified = df_filtered[df_filtered['Type'] == 'modified'].copy()

    # Merge to calculate difference
    df_diff = pd.merge(
        df_modified,
        df_baseline,
        on='Baseline',
        how='left'
    )

    # Calculate the difference
    df_diff['batch_diff'] = df_diff['batch_score'] - df_diff['baseline_batch_score']
    df_diff['biocons_diff'] = df_diff['biocons_score'] - df_diff['baseline_biocons_score']
    
    return df_diff[['Baseline', 'B_modifier', 'batch_diff', 'biocons_diff']]


def create_global_summary_two_panel(df_data_global, filter_c, b_filter_list, filename, b_arrow_list=None, mod_method_list=None, tall=False, fixed_bounds=False):
    """Generates a two-panel plot: Dot Plot (L) and Diff Bar Plot (R) with arrows."""
    
    # Filter for baseline methods OR methods where C_modifier matches the filter AND B_modifier is in b_filter_list
    df_dot_plot = df_data_global[
        (df_data_global['Type'] == 'baseline') | 
        ((df_data_global['C_modifier'] == filter_c) & (df_data_global['B_modifier'].isin(b_filter_list)))
    ].copy()
    
    # Setup the figure for two subplots (1 row, 2 columns)
    fig, axes = plt.subplots(1, 2, figsize=(14, (8 if tall else 7)), gridspec_kw={'width_ratios': [1.1, 1]}) #, constrained_layout=True)
    #New: swapped left and right to lead with the bar plot
    ax1 = axes[1] # Left: Dot Plot (Batch vs Biocons)
    ax2 = axes[0] # Right: Bar Plot (Batch Difference)
    
    b_mods_str = ", ".join(b_filter_list)
    fig.suptitle(f'Global Aggregate Scores (C = "{filter_c}", B = "{b_mods_str}")', fontsize=16, fontweight='bold')

    fig.subplots_adjust(left=0.05, bottom=0.08, right=1.0, top=0.92, wspace=0.2, hspace=0.0)

       
    # --- AX1: Dot Plot Logic (Uses global scale for global average data) ---
    
    x_offset = max(0.1, df_dot_plot['batch_score'].max() - df_dot_plot['batch_score'].min()) * 0.15
    y_offset = max(0.1, df_dot_plot['biocons_score'].max() - df_dot_plot['biocons_score'].min()) * 0.15
    x_min, x_max = df_dot_plot['batch_score'].min() -x_offset, df_dot_plot['batch_score'].max() +x_offset
    y_min, y_max = df_dot_plot['biocons_score'].min() -y_offset, df_dot_plot['biocons_score'].max() +y_offset

    if not (np.isfinite(x_min) and np.isfinite(x_max)): x_min, x_max = 0, 1.1
    if not (np.isfinite(y_min) and np.isfinite(y_max)): y_min, y_max = 0, 1.1

    # ... plotting logic for ax1 (same as before) ...
    # 1. Store baseline coordinates for arrows
    baseline_coords = {}
    df_baseline_subset = df_dot_plot[df_dot_plot['Type'] == 'baseline']
    for _, row in df_baseline_subset.iterrows():
        baseline_coords[row['Baseline']] = (row['batch_score'], row['biocons_score'])

    # 2. Plot points and collect modified coordinates
    modified_points = []
    for _, row in df_dot_plot.iterrows():
        baseline = row['Baseline']
        method_type = row['Type']
        if method_type == 'modified':
            if mod_method_list and baseline not in mod_method_list:
                continue
            elif b_arrow_list and row['B_modifier'] in b_arrow_list:
                modified_points.append(row)
        
        color = color_map.get(baseline, 'gray')
        marker = get_marker(row['B_modifier'], row['C_modifier'], method_type)
        
        ax1.plot(
            row['batch_score'],     # X-axis
            row['biocons_score'],   # Y-axis
            marker=marker,
            color=color, 
            markersize=15,
            linestyle='',
            alpha=0.8,
            zorder=2,
        )

    # 3. Draw arrows
    for row in modified_points:
        baseline_name = row['Baseline']
        modified_x, modified_y = row['batch_score'], row['biocons_score']
        
        if baseline_name in baseline_coords:
            baseline_x, baseline_y = baseline_coords[baseline_name]
            color = color_map.get(baseline_name, 'gray')

            fancy_arrow = FancyArrowPatch(
                posA=(baseline_x, baseline_y),
                posB=(modified_x, modified_y),
                arrowstyle="simple", 
                facecolor=color,
                edgecolor=color,
                alpha=0.5,
                mutation_scale=20,
                shrinkA=15, shrinkB=15
            )
            ax1.add_patch(fancy_arrow)

    # Panel settings
    ax1.set_title('Average Benchmarks on 9 Datasets', fontsize=16) 
    ax1.set_xlabel('Batch Correction Score', fontsize=16) 
    ax1.set_ylabel('Bio Conservation Score', fontsize=16) 
    ax1.grid(True, linestyle='--', alpha=0.6)
    if fixed_bounds:
        ax1.set_xlim(0.42, 0.79) #0.33 -> 0.37
        ax1.set_ylim(0.21, 0.63) #0.38 -> 0.42
        ax1.set_aspect('equal', adjustable='box')
    else:
        ax1.set_xlim(x_min, x_max)
        ax1.set_ylim(y_min, y_max)
    ax1.tick_params(axis='both', which='major', labelsize=12)

    # --- AX2: Bar Plot Logic (Uses global average data) ---   

    pos1 = ax2.get_position()
    label_height_shift = 0.18 
    new_y0 = pos1.y0 + label_height_shift
    new_height = pos1.height - label_height_shift
    ax2.set_position([pos1.x0, new_y0, pos1.width, new_height])

    df_diff = calculate_difference(df_data_global, filter_c, b_filter_list)
    baselines_for_diff = df_diff['Baseline'].unique()
    if mod_method_list:
        baselines_for_diff = set(baselines_for_diff) & set(mod_method_list)
    baselines_for_diff = sorted(list(baselines_for_diff))
    b_modifiers_in_plot = [b for b in b_filter_list if b in df_diff['B_modifier'].unique()]
    width = 0.25
    y_max_diff = 0
    x = np.arange(len(baselines_for_diff))
    num_groups = len(baselines_for_diff) * len(b_modifiers_in_plot)
    names = []
    colors = []
    doing_sel_plot = "sel2" in b_modifiers_in_plot
    batch_bars = []
    biocons_bars = []
    for b_mod in b_modifiers_in_plot:
        df_subset = df_diff[df_diff['B_modifier'] == b_mod]
        
        batch_diff_values = []
        biocons_diff_values = []
        for baseline in baselines_for_diff:
            value_series = df_subset[df_subset['Baseline'] == baseline]['batch_diff']
            value = value_series.iloc[0] if not value_series.empty else 0
            batch_diff_values.append(value)
            value_series = df_subset[df_subset['Baseline'] == baseline]['biocons_diff']
            value = value_series.iloc[0] if not value_series.empty else 0
            biocons_diff_values.append(-value)
        diff_values = batch_diff_values + biocons_diff_values
        if diff_values:
            y_max_diff = max(y_max_diff, max(diff_values))
        
        multiplier = 0
        offset = multiplier * width
        rects = ax2.bar(x + offset, [max(v, 0) for v in batch_diff_values], width, color="green")
        batch_bars.append((rects, batch_diff_values))
        #ax2.bar_label(rects, padding=3, fmt='%.2f')

        multiplier += 1
        offset = multiplier * width
        rects = ax2.bar(x + offset, [max(v, 0) for v in biocons_diff_values], width, color="red")
        biocons_bars.append((rects, biocons_diff_values))
        #ax2.bar_label(rects, padding=3, fmt='%.2f')

        approach = nice_approach(b_mod, sel_only=doing_sel_plot)
        for baseline in baselines_for_diff:
            if doing_sel_plot:
                names.append(nice_method(baseline))                             
            else:
                names.append(f"{nice_method(baseline)} {approach}")
            colors.append(color_map.get(baseline, 'gray'))
        if doing_sel_plot:
            y_pos_rel = -0.38 #***
            ax2.text(
                (x[0] + x[-1]) / 2, 
                y_pos_rel, 
                approach,
                transform=ax2.get_xaxis_transform(),
                ha='center', va='bottom', fontsize=16, color='black') #,fontweight='bold')

        x += len(baselines_for_diff)

    ax2.set_ylabel('Δ Score from Baseline Method', fontsize=16)
    rotation = 45
    if doing_sel_plot:
        rotation = 90
    elif tall:
        rotation = 60
    ax2.set_xticks(np.arange(num_groups) + width/2, names, rotation=rotation, ha='right', fontsize=16)

    trans = mtrans.Affine2D().translate(6, 0) #Can be tuned
    for xtick, color in zip(ax2.get_xticklabels(), colors):
        xtick.set_color(darken_color(color))
        xtick.set_transform(xtick.get_transform()+trans)

    ax2.tick_params(axis='y', which='major', labelsize=12)
    ax2.set_xlim(-2*width, num_groups-width)
    ax2.set_title(f"Approach: {title_approach(b_mods_str, filter_c)}", fontsize=16)

    y_max_diff*=1.2
    if fixed_bounds:
        ax2.set_ylim(0, 0.34) 
    else:
        ax2.set_ylim(0, y_max_diff)
    bar_ofset = 0.01 * y_max_diff
    for rects, batch_diff_values in batch_bars:
        for bar, v  in zip(rects, batch_diff_values):
            bar_height = bar.get_height()
            # Shift text left by reducing the x coordinate slightly (e.g., 0.1 units)
            # The x position is the center of the bar by default, so a small offset works.
            x_pos = bar.get_x() + bar.get_width() - 0.01
            y_pos = bar_height + bar_ofset # Position the text at the top of the bar
            ax2.text(x_pos, y_pos, f'{v:.2f}', ha='right', va='bottom', color='black', fontsize=10)

    for rects, biocons_diff_values in biocons_bars:
        for bar, v in zip(rects, biocons_diff_values):
            bar_height = bar.get_height()
            # Shift text left by reducing the x coordinate slightly (e.g., 0.1 units)
            # The x position is the center of the bar by default, so a small offset works.
            x_pos = bar.get_x() + 0.01
            y_pos = bar_height + bar_ofset # Position the text at the top of the bar
            ax2.text(x_pos, y_pos, f'{v:.2f}', ha='left', va='bottom', color='black', fontsize=10)        

    for i, _ in enumerate(b_modifiers_in_plot[:-1]):
        ax2.vlines(x=((i+1)*len(baselines_for_diff))-width*1.5, ymin=0, ymax=y_max_diff, colors='gray', linestyles='--', linewidth=2)

    #create_shared_legend(fig, filter_c, b_filter_list)

    plt.savefig(filename, format='svg', bbox_inches='tight')
    plt.close(fig)


# --- 6. Jitter Plotting Functions (Metric vs. Metric Axis - Global Average) ---

def plot_jitter_data_on_ax(ax, df_data, filter_c, b_filter_list):
    """Shared function to plot jittered metric scores on a single axis (Metrics on X-axis), uses global average data."""
    
    # Filter data for the current C_modifier and B-list, including baselines
    df_plot = df_data[
        (df_data['Type'] == 'baseline') | 
        ((df_data['C_modifier'] == filter_c) & (df_data['B_modifier'].isin(b_filter_list)))
    ].copy()
    
    if df_plot.empty:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
        return

    # 1. Map metrics to numerical x-positions
    x_map = {metric: i for i, metric in enumerate(all_metrics)}
    
    # 2. Set X-axis tick locations and labels
    ax.set_xticks(list(x_map.values()))
    ax.set_xticklabels([nice_metric(x) for x in all_metrics], rotation=45, ha='right', fontsize=16) #***

    # 3. Add jitter and plot
    for metric, df_subset in df_plot.groupby('Metric'):
        if metric not in x_map: continue
        
        x_base = x_map[metric]
        
        for _, row in df_subset.iterrows():
            # Apply Jitter (random noise)
            jitter = np.random.uniform(-0.25, 0.25) #Changed from .15
            x_pos = x_base + jitter

            baseline = row['Baseline']
            method_type = row['Type']
            
            color = color_map.get(baseline, 'gray')
            marker = get_marker(row['B_modifier'], row['C_modifier'], method_type)

            ax.plot(
                x_pos,
                row['Value'],
                marker=marker,
                color=color,
                markersize=10,#8,
                linestyle='',
                alpha=0.8,
                zorder=2,
            )

    # 4. Add custom group labels above the axis 
    batch_end_x = len(batch_metrics) - 0.5
    y_pos_rel = 1.01 #***
    
    ax.text(
        (len(batch_metrics) - 1) / 2, 
        y_pos_rel, 
        'Batch Correction Metrics (9-Dataset Average)',
        transform=ax.get_xaxis_transform(),
        ha='center', va='bottom', fontsize=16, color='darkblue'
    )

    ax.text(
        (len(batch_metrics) + len(all_metrics) - 1) / 2, 
        y_pos_rel, 
        'Bio Conservation Metrics (9-Dataset Average)', 
        transform=ax.get_xaxis_transform(),
        ha='center', va='bottom', fontsize=16, color='darkgreen'  
    )
    
    ax.axvline(x=batch_end_x, color='gray', linestyle='--', linewidth=1, zorder=1)
    
    ax.set_xlim(-0.5, len(all_metrics) - 0.5)
    ax.set_ylabel('Individual Metric Score', fontsize=16) 
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    
    # Set Y-axis limits based on all data
    y_offset = max(0.1, df_plot['Value'].max() - df_plot['Value'].min()) * 0.15
    y_min, y_max = df_plot['Value'].min() -y_offset, df_plot['Value'].max() +y_offset    
    y_min = y_min if np.isfinite(y_min) else 0
    y_max = y_max if np.isfinite(y_max) else 1.1
    ax.set_ylim(y_min, y_max)
    
    ax.tick_params(axis='y', which='major', labelsize=12)


def create_jitter_plot_global(df_data_global_long, filter_c, b_filter_list, filename):
    """Generates a single panel jitter plot for global individual metric scores (averaged across datasets)."""
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 7), constrained_layout=True)
    b_mods_str = ", ".join(b_filter_list)
    fig.suptitle(f'Global Individual Metric Scores (C = "{filter_c}", B = "{b_mods_str}")', fontsize=16, fontweight='bold')
    
    plot_jitter_data_on_ax(ax, df_data_global_long, filter_c, b_filter_list)

    ax.set_xlabel('')
        
    create_shared_legend(fig, filter_c, b_filter_list, arrow_space=False)

    plt.savefig(filename, format='svg', bbox_inches='tight')
    plt.close(fig)


# --- 7. Jitter Plotting Functions (Metric vs. Dataset Axis) - Individual Axis Scaling Applied ---

def plot_jitter_data_dataset_on_ax(ax, df_data, filter_c, b_filter_list, metric, datasets=None, is_last_row=True):
    """Helper to plot a single metric's scores across all datasets (Datasets on X-axis), filtered by C and B-list."""
    
    # Filter for the current metric and C_modifier, including baselines
    df_plot = df_data[
        (df_data['Metric'] == metric) & 
        ((df_data['Type'] == 'baseline') | ((df_data['C_modifier'] == filter_c) & (df_data['B_modifier'].isin(b_filter_list))))
    ].copy()
    
   
    if df_plot.empty:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        return

    # 3. Map Datasets to numerical x-positions
    if datasets is not None:
        datasets_ordered = sorted(datasets)
    else:
        datasets_ordered = sorted(df_plot['Dataset'].unique())
    x_map = {ds: i for i, ds in enumerate(datasets_ordered)}
    
    ax.set_xticks(list(x_map.values()))
    if is_last_row:
        ax.set_xticklabels([short_dataset(x) for x in datasets_ordered], rotation=60, ha='right', fontsize=16) 
        trans = mtrans.Affine2D().translate(6, 0) #Can be tuned
        for xtick in ax.get_xticklabels():
            xtick.set_transform(xtick.get_transform()+trans)
    else:
        ax.set_xticklabels([]) 

    # --- INDIVIDUAL AXIS SCALING APPLIED HERE ---
    # Set Y-axis limits based only on the current metric's data
    y_offset = max(0.1, df_plot['Value'].max() - df_plot['Value'].min()) * 0.15
    y_min, y_max = df_plot['Value'].min() -y_offset, df_plot['Value'].max() +y_offset    
    
    y_min = y_min if np.isfinite(y_min) else -0.05
    y_max = y_max if np.isfinite(y_max) else 1.05
    ax.set_ylim(y_min, y_max)
    
    # 4. Plotting
    for dataset, df_subset in df_plot.groupby('Dataset'):
        if dataset not in x_map: continue
        
        x_base = x_map[dataset]
        
        for _, row in df_subset.iterrows():
            # Apply Jitter
            jitter = np.random.uniform(-0.2, 0.2) 
            x_pos = x_base + jitter

            baseline = row['Baseline']
            method_type = row['Type']
            
            color = color_map.get(baseline, 'gray')
            marker = get_marker(row['B_modifier'], row['C_modifier'], method_type)

            ax.plot(
                x_pos,
                row['Value'],
                marker=marker,
                color=color,
                markersize=8,
                linestyle='',
                alpha=0.8,
                zorder=2,
            )

    ax.set_xlim(-0.5, len(datasets_ordered) - 0.5)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    ax.tick_params(axis='y', which='major', labelsize=12)


def create_jitter_plot_per_metric(df_data_long, filter_c, b_filter_list, filename):
    """Generates the multi-panel jitter plot with one panel per metric (Datasets on X-axis)."""
    
    # Filter for baseline methods OR methods where C_modifier matches the filter
    df_plot_all = df_data_long[
        (df_data_long['Type'] == 'baseline') | 
        ((df_data_long['C_modifier'] == filter_c) & (df_data_long['B_modifier'].isin(b_filter_list)))
    ].copy()
     
    n_metrics = len(all_metrics)
    ncols = 3 
    nrows = int(math.ceil(n_metrics / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 18), constrained_layout=True) #Height increased from 14
    axes = axes.flatten()

    b_mods_str = ", ".join(b_filter_list)
    fig.suptitle(f'Individual Metric Scores (C = "{filter_c}", B = "{b_mods_str}") - Grouped by Metric', fontsize=18, fontweight='bold')

    # Add shared X and Y axis labels
    fig.text(0.0, 0.5, 'Metric Score', va='center', rotation='vertical', fontsize=16)
    
    # Plotting Loop
    for i, metric in enumerate(all_metrics):
        if i >= len(axes): 
            break
            
        ax = axes[i]

        last_row = i >= ncols * (nrows-1)
        
        # Plot data for the specific metric (Individual Scaling applied inside helper)
        plot_jitter_data_dataset_on_ax(ax, df_plot_all, filter_c, b_filter_list, metric, datasets=df_plot_all['Dataset'].unique(), is_last_row=last_row)

        # Panel settings
        is_batch = metric in batch_metrics
        color = 'darkblue' if is_batch else 'darkgreen'
        
        # Set title (Metric name)
        ax.set_title(f'{nice_metric(metric)} ({ "Batch Cor." if is_batch else "Bio. Cons." })', fontsize=16, color=color)
        
        ax.set_xlabel('') 
        ax.set_ylabel('')

    # Remove unused subplots
    for j in range(n_metrics, len(axes)):
        fig.delaxes(axes[j])

    #create_shared_legend(fig, filter_c, b_filter_list, arrow_space=False)

    plt.savefig(filename, format='svg', bbox_inches='tight')
    plt.close(fig)


# --- 8. Generate All Plots ---

print("Starting plot generation...")

b_modifier_groups = { #BR modes
    'scale': ['scale'],
    #'sel50': ['sel50'], #sel = filtering
    #'selfix01': ['selfix01'], 
    #'selfix10': ['selfix10'],
    #'seliqr': ['seliqr'],
    #'sel80': ['sel80'],
    #'sel67': ['sel67'],
    'subbm': ['subbm'],
    #'combo': ['scale', 'sel67']
    }
c_modifiers = ['pcr', 'ilisi'] 
method_subsets = {"combo": ['harmonypy', 'sca', 'pca', 'scanorama', 'concord']}

for c_mod in c_modifiers:
    for group_name, b_list in b_modifier_groups.items():
        # 1. Global Aggregate Plots (Dot Plots + Difference Bar Plot) - Two Panel, Filtered by C and B-list - No Exclusion/Global Scale
        create_global_summary_two_panel(
            df_agg_global, 
            filter_c=c_mod, 
            b_filter_list=b_list, 
            filename=f'plot_global_summary_{c_mod}_{group_name}.svg',
            b_arrow_list=['scale', 'sel50', 'selfix01', 'seliqr', 'sel80', 'sel67', 'selfix10', 'subbm'],
            mod_method_list=method_subsets.get(group_name, None),
            fixed_bounds=True, 
            tall=True
        )
        
        # 2. Aggregate Plots (Dot Plots) - Per Dataset, Filtered by C and B-list - Individual Scaling and Exclusion Applied
        create_dot_plot(
            df_agg, 
            filter_c=c_mod, 
            b_filter_list=b_list, 
            filename=f'plot_dot_{c_mod}_{group_name}.svg',
            b_arrow_list=['scale', 'sel50', 'selfix01', 'seliqr', 'sel80', 'sel67', 'selfix10', 'subbm'],
            mod_method_list=method_subsets.get(group_name, None), 
            p_value=True
        )
        # 3. Jitter Plots (Individual Metrics, Global Average) - Single Panel, Filtered by C and B-list - No Exclusion/Global Scale
        create_jitter_plot_global(
            df_long_global, 
            filter_c=c_mod, 
            b_filter_list=b_list, 
            filename=f'plot_jitter_global_{c_mod}_{group_name}.svg'
        ) 
        # 4. Jitter Plots (Individual Metrics, Grouped by Metric) - Filtered by C and B-list - Individual Scaling and Exclusion Applied
        create_jitter_plot_per_metric(
            df_long, 
            filter_c=c_mod, 
            b_filter_list=b_list, 
            filename=f'plot_jitter_per_metric_{c_mod}_{group_name}.svg'
        )
