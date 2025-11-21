import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.patches import FancyArrowPatch
import colorsys
import matplotlib.colors as mc
import numpy as np

# Metric lists (YOU MUST FILL THESE IN BASED ON YOUR DATA)
# Example: batch metrics (first 8) and biocons metrics (last 7)

biocons_metrics = ["isolated_label_asw", "clisi", 
                 "nmi", "ari", "asw_label",
                 "isolated_label_f1", "cell_cycle_conservation"]
batch_metrics = ["asw_batch", "ilisi", "kbet",
                   "graph_connectivity", "pcr"]

# Save the placeholder data to a temporary CSV for simulation
# df_raw.to_csv('placeholder_data.csv', index=False)
df = pd.read_csv('tables/Combined_100_scale_batch_scores_v1.csv')
df.drop(df.tail(1).index,inplace=True)
df['Batch correction'] = df['Batch correction'].astype(np.float64)
df['Bio conservation'] = df['Bio conservation'].astype(np.float64)
df['Batch Δ'] = df['Batch Δ'].astype(np.float64)
df['BioCons Δ'] = df['BioCons Δ'].astype(np.float64)

baseline_df = df[df['Embedding'].str.contains('Baseline')]
modified_df = df[(df['Batch Δ'].abs() > 0.03) | (df['BioCons Δ'].abs() > 0.03) ]
modified_df_scale = modified_df[~(modified_df['Embedding'].str.contains('_Batch'))]

df = pd.read_csv('tables/Combined_100_select_batch_scores_v1.csv')
df.drop(df.tail(1).index,inplace=True)
df['Batch correction'] = df['Batch correction'].astype(np.float64)
df['Bio conservation'] = df['Bio conservation'].astype(np.float64)
df['Batch Δ'] = df['Batch Δ'].astype(np.float64)
df['BioCons Δ'] = df['BioCons Δ'].astype(np.float64)
modified_df = df[(df['Batch Δ'].abs() > 0.03) | (df['BioCons Δ'].abs() > 0.03) ]
modified_df_select = modified_df[~(modified_df['Embedding'].str.contains('_Batch'))]

# Define the required order for the jitter plots
all_metrics = batch_metrics + biocons_metrics

# --- 2. Data Preprocessing and Aggregation ---

def parse_method_name(method_name):
    """Parses method name into components: Baseline, Type, B_modifier, C_modifier."""
    parts = method_name.split('_')
    if len(parts) == 1:
        # Baseline method (e.g., "Method1")
        return parts[0], 'baseline', "", "" 
    elif len(parts) == 3:
        # Modified method (e.g., "Method1_scale_pcr")
        baseline = parts[0]
        B = parts[1]
        C = parts[2]
        return baseline, 'modified', B, C
    return method_name, 'unknown', None, None

# --- 3. Plotting Setup (Color and Marker Maps) ---

baseline_methods = sorted([x.split()[0] for x in baseline_df['Embedding'].unique()])
colors = plt.cm.get_cmap('Set1', len(baseline_methods)+4)
color_map = {name: colors(i) for i, name in enumerate(baseline_methods)}

marker_map = {
    'baseline': 'o',      # Circle for baseline
    'pcrcomp': 's',           # Square for C='pcr'
    'ilisi': '^',          # Triangle for C='ilisi'
    'silbatch': 'P',
    'kbet': 'X'
}

# --- 4. Shared Legend Creation Function ---

def darken_color(color, amount=0.75):
    c = colorsys.rgb_to_hls(*mc.to_rgb(color))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])    

def create_shared_legend(fig):
    """Creates and positions the shared figure legend using Proxy Artists."""
    
    # 1. Color Legend (Baseline Method)
    legend_elements_color = [
        mlines.Line2D([0], [0], marker='o', color='w', label=base,
                   markerfacecolor=color_map.get(base), markersize=15) # markersize=15
        for base in baseline_methods
    ]

    # 2. Shape Legend (Method Type)
    legend_elements_shape = [
        mlines.Line2D([0], [0], marker=marker_map['baseline'], color='w', label='', markerfacecolor='w', markersize=15), # markersize=15
        mlines.Line2D([0], [0], marker=marker_map['pcrcomp'], color='w', label=f'PCR Comp.', markerfacecolor='k', markersize=15), # markersize=15
        mlines.Line2D([0], [0], marker=marker_map['ilisi'], color='w', label=f'iLISI', markerfacecolor='k', markersize=15), # markersize=15
        mlines.Line2D([0], [0], marker=marker_map['silbatch'], color='w', label=f'Sil. Batch', markerfacecolor='k', markersize=15), # markersize=15
        mlines.Line2D([0], [0], marker=marker_map['kbet'], color='w', label=f'KBET', markerfacecolor='k', markersize=15) # markersize=15
    ]
    
    legend_title = "Baseline Method (Color) | BR Shift | BatchRefiner Metric (Shape)"
    full_legend_elements = legend_elements_color + legend_elements_shape
    labels = [e.get_label() for e in full_legend_elements]

    # Position the legend outside of the subplots
    fig.legend(full_legend_elements, labels, loc='lower center', 
               bbox_to_anchor=(0.5, -0.05),
               ncol=len(baseline_methods) + 5,
               title=legend_title, frameon=False, fontsize=12, title_fontsize=16, handletextpad=0.2) # Fontsize increased
    

# Setup the figure for two subplots (1 row, 2 columns)
fig, axes = plt.subplots(1, 2, figsize=(14, 7), constrained_layout=True)
fig.suptitle('Selected Benchmarks with BatchRefiner on Preliminary Dataset', fontsize=16, fontweight='bold')


# Calculate global min/max for consistent axes across both panels (X=Batch, Y=Biocons)
df_temp = pd.concat([modified_df_scale, modified_df_select, baseline_df])
x_min, x_max = df_temp['Batch correction'].min() * 0.9, df_temp['Batch correction'].max() * 1.1
y_min, y_max = df_temp['Bio conservation'].min() * 0.9, df_temp['Bio conservation'].max() * 1.1

if not (np.isfinite(x_min) and np.isfinite(x_max)): x_min, x_max = 0, 1.1
if not (np.isfinite(y_min) and np.isfinite(y_max)): y_min, y_max = 0, 1.1

for i, df_panel in enumerate([modified_df_scale, modified_df_select]):
    ax = axes[i]
    

    # Replace "" B_modifier with the filter_b value
    #df_panel.loc[df_panel['B_modifier'] == "", 'B_modifier'] = filter_b
    baseline_dict = {}
    for _, row in baseline_df.iterrows():
        baseline = row["Embedding"].split()[0]
        color = color_map.get(baseline, 'gray')
        marker = marker_map['baseline']
        # Plotting (AXES SWAPPED: X=Batch, Y=Biocons)
        ax.plot(
            row['Batch correction'],     # X-axis
            row['Bio conservation'],   # Y-axis
            marker=marker,
            color=color, 
            markersize=15,          # Increased size
            linestyle='',
            alpha=0.8,              # Set alpha
            zorder=2,
        )
        baseline_dict[baseline] = (row['Batch correction'], row['Bio conservation'])

    for _, row in df_panel.iterrows():
        baseline, _, metric = row["Embedding"].split("_")[0:3]
        if baseline == "Harm":
            baseline = "Harmony"
        elif baseline == "Scan":
            baseline = "Scanorama"
        color = color_map.get(baseline, 'gray')
        marker = marker_map[metric.lower().replace("shil", "sil")]
        ax.plot(
            row['Batch correction'],     # X-axis
            row['Bio conservation'],   # Y-axis
            marker=marker,
            color=color, 
            markersize=15,          # Increased size
            linestyle='',
            alpha=0.8,              # Set alpha
            zorder=2,
        )
        if metric.lower() == "pcrcomp":
            xb, yb = baseline_dict[baseline]
            # 3. Create the FancyArrowPatch object
            fancy_arrow = FancyArrowPatch(
                posA=(xb, yb),            # Start position
                posB=(row['Batch correction'], row['Bio conservation']),            # End position
                arrowstyle="simple",         # Defines the shape and head size
                facecolor=color,                # Fill color of the arrow and head
                edgecolor=color,                # Outline color (set same as facecolor)
                alpha=0.5,                      # Transparency
                mutation_scale=20,               # Scale factor for arrow components (default is fine)
                shrinkA=15, shrinkB=15           # Ensures the arrow starts and ends exactly at posA/posB
            )
            ax.add_patch(fancy_arrow)
             



    # Panel settings (AXIS LABELS SWAPPED)
    ax.set_xlabel('Batch Correction Score', fontsize=16) # Increased font size
    ax.set_ylabel('Bio Conservation Score', fontsize=16) # Increased font size
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    # Increase tick label font size
    ax.tick_params(axis='both', which='major', labelsize=12)
axes[0].set_title(f'Approach: Scaling (100d)', fontsize=16)
axes[1].set_title(f'Approach: Filtering (50d from 100d)', fontsize=16)
    
# Add the shared legend
create_shared_legend(fig)

# Save the figure as SVG
plt.savefig('plot_prelim_summary.svg', format='svg', bbox_inches='tight')
plt.close(fig)


df_both = pd.concat([modified_df_scale, modified_df_select])

x = np.arange(len(df_both))  # the label locations

fig, ax = plt.subplots(layout='constrained', figsize=(14, 8))

width = 0.25  # the width of the bars
multiplier = 0
offset = width * multiplier
rects = ax.bar(x + offset, df_both["Batch Δ"], width, color="green", label="Increase in Batch Integration score")
#ax.bar_label(rects, padding=3, fmt='%.2f')
for bar in rects:
    bar_height = bar.get_height()
    # Shift text left by reducing the x coordinate slightly (e.g., 0.1 units)
    # The x position is the center of the bar by default, so a small offset works.
    x_pos = bar.get_x() + bar.get_width() - 0.01
    y_pos = bar_height + 0.003 # Position the text at the top of the bar
    ax.text(x_pos, y_pos, f'{bar_height:.2f}', ha='right', va='bottom', color='black', fontsize=10)

multiplier += 1
offset = width * multiplier
rects = ax.bar(x + offset, np.maximum(-(df_both["BioCons Δ"].to_numpy()),0), width, color="red", label="Decrease in Bio Conservation score")
#ax.bar_label(rects, padding=3, fmt='%.2f')
for bar in rects:
    bar_height = bar.get_height()
    # Shift text left by reducing the x coordinate slightly (e.g., 0.1 units)
    # The x position is the center of the bar by default, so a small offset works.
    x_pos = bar.get_x() + 0.01
    y_pos = bar_height + 0.003 # Position the text at the top of the bar
    ax.text(x_pos, y_pos, f'{bar_height:.2f}', ha='left', va='bottom', color='black', fontsize=10)   

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Δ Score from Baseline Method', fontsize=16)
#ax.set_title('')
names = []
colors = []
n_scaled = 0
for name in list(df_both["Embedding"]):
    tokens = name.split("_")
    baseline, approach, metric = name.split("_")[0:3]
    if baseline == "Harm":
        baseline = "Harmony"
    elif baseline == "Scan":
        baseline = "Scanorama"
    if approach == "select":
        approach = "filter"
    elif approach == "scale":
        n_scaled += 1
        approach = "scal"
    if metric == "PCRComp":
        metric = "PCR Comp."
    elif metric == "ShilBatch" or metric == "SilBatch": #I have a typo in some places
        metric = "Sil. Batch"
    
    names.append(f"{baseline} {approach}ed w/ {metric}")
    colors.append(darken_color(color_map.get(baseline, 'gray')))
    
ax.set_xticks(x + width/2, names, rotation=75, ha='right', fontsize=16)
for xtick, color in zip(ax.get_xticklabels(), colors):
    xtick.set_color(color)
fig.legend(loc='lower center', ncols=2, fontsize=12)
ax.set_ylim(0, 0.4)
ax.set_xlim(-2*width, len(names)-width)
ax.tick_params(axis='y', which='major', labelsize=12)
ax.vlines(x=n_scaled-width*1.5, ymin=0, ymax=0.4, colors='gray', linestyles='--', linewidth=2)




plt.savefig('plot_prelim_bars.svg', format='svg', bbox_inches='tight')
plt.close(fig)


