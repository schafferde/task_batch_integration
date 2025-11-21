import argparse
import numpy as np
import pandas as pd

#Adapted from scib-metrics to run on a pre-made DF
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from plottable import ColumnDefinition, Table
from plottable.cmap import normed_cmap
from plottable.plots import bar

_METRIC_TYPE = "Metric Type"
_AGGREGATE_SCORE = "Aggregate score"


def plot_results_table(df, show: bool = False, save_path: str | None = None, batch: bool = True, biocons: bool = True):
    """Plot the benchmarking results.

    Parameters
    ----------
    min_max_scale
        Whether to min max scale the results.
    show
        Whether to show the plot.
    save_dir
        The directory to save the plot to. If `None`, the plot is not saved.
    """
    num_embeds = len(df) - 1
    cmap_fn = lambda col_data: normed_cmap(col_data, cmap=mpl.cm.PRGn, num_stds=2.5)
    # Do not want to plot what kind of metric it is
    df = df.drop(columns=['Total'])
    plot_df = df.drop(_METRIC_TYPE, axis=0).astype(np.float64)
    ## Sort by total score
    #if batch and biocons:
    #    sort_col = "Total"
    #elif batch is not None:
    #    sort_col = "Batch correction"
    #else:
    #    sort_col = "Bio conservation"
    #plot_df = plot_df.sort_values(by=sort_col, ascending=False).astype(np.float64)
    plot_df["Method"] = plot_df.index

    # Split columns by metric type, using df as it doesn't have the new method col
    score_cols = df.columns[df.loc[_METRIC_TYPE] == _AGGREGATE_SCORE]
    other_cols = df.columns[df.loc[_METRIC_TYPE] != _AGGREGATE_SCORE]
    column_definitions = [
        ColumnDefinition("Method", width=4.5, textprops={"ha": "left", "weight": "bold"}), #Increased width
    ]
    # Circles for the metric values
    column_definitions += [
        ColumnDefinition(
            col,
            title=col.replace(" ", "\n", 1).replace("connectivity", "connect.").replace("comparison", "compar."),
            width=1,
            textprops={
                "ha": "center",
#                "bbox": {"boxstyle": "circle", "pad": 0.25},
            },
#            cmap=cmap_fn(plot_df[col]),
            group=df.loc[_METRIC_TYPE, col],
            formatter="{:.2f}",
        )
        for i, col in enumerate(other_cols)
    ]
    # Bars for the aggregate scores
    column_definitions += [
        ColumnDefinition(
            col,
            width=1.5,
            title=col.replace(" ", "\n", 1),
            plot_fn=bar,
            plot_kw={
                "cmap": mpl.cm.YlGnBu,
                "plot_bg_bar": False,
                "annotate": True,
                "height": 0.9,
                "formatter": "{:.2f}",
            },
            group=df.loc[_METRIC_TYPE, col],
            border="left" if i == 0 else None,
        )
        for i, col in enumerate(score_cols[:2])
    ]
    column_definitions += [
        ColumnDefinition(
            col,
            width=1,
            title=col.replace(" ", "\n", 1),
            textprops={"ha": "center", "fontsize": 12},
            group=df.loc[_METRIC_TYPE, col],
            border="left" if i == 0 else None,
            formatter="{:.2f}",
            cmap=cmap_fn(plot_df[col]),

        )
        for i, col in enumerate(score_cols[2:])
    ]    
    # Allow to manipulate text post-hoc (in illustrator)
    with mpl.rc_context({"svg.fonttype": "none"}):
        fig, ax = plt.subplots(figsize=(len(df.columns) * 1.25, 3 + 0.3 * num_embeds))
        tab = Table(
            plot_df,
            cell_kw={
                "linewidth": 0,
                "edgecolor": "k",
            },
            column_definitions=column_definitions,
            ax=ax,
            row_dividers=True,
            footer_divider=True,
            textprops={"fontsize": 10, "ha": "center"},
            row_divider_kw={"linewidth": 1, "linestyle": (0, (1, 5))},
            col_label_divider_kw={"linewidth": 1, "linestyle": "-"},
            column_border_kw={"linewidth": 1, "linestyle": "-"},
            index_col="Method",
        ).autoset_fontcolors(colnames=plot_df.columns)
    if show:
        plt.show()
    if save_path is not None:
        fig.savefig(save_path, facecolor=ax.get_facecolor(), dpi=300)

    return tab


def create_parser():
    parser = argparse.ArgumentParser(
        description='A brief description of your script and what it does.'
    )
    parser.add_argument(
        '--biocons', '-b',
        action='store_true',
        help='Use bioconservation metrics instead of batch'
    )
    parser.add_argument(
        '--select', '-s',
        action='store_true',
        help='Evaluate selection (default - scaling)'
    )
    parser.add_argument(
        '--select_scale', '-ss',
        action='store_true',
        help='Optional: Select and scale the selected columns'
    )

    parser.add_argument(
        '--dims', '-d',
        type=int,
        default = 100,
        help='Number of dimensions produced (default: 100)'
    )





    return parser

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()

    

    
    if args.select:
        approach = "select"
    elif args.select_scale:
        approach = "select+scale"
    else:
        approach = "scale"


    methods = ["Harm", "PCA", "SCA", "Scan", "Seurat"]
    total_dims = args.dims
    dfs = []
    for method in methods:
        output_dir = f'{method}_{total_dims}_{approach}_{"biocons" if args.biocons else "batch"}'
        cur_df = pd.read_pickle((f'{output_dir}/scores.pkl'))
        type_row = cur_df.loc[[_METRIC_TYPE]]
        type_row[['Batch Δ', 'BioCons Δ']] = type_row.iloc[:, -2:]
        cur_df = cur_df.drop(_METRIC_TYPE)
        sort_col = "Batch correction"
        cur_df = cur_df.sort_values(by=sort_col, ascending=False).astype(np.float64)
        baseline_key = method + "100"
        if approach == "scale" and not args.biocons:
            #Baseline data are present
            baseline_row = cur_df.loc[[baseline_key]]
            cur_df = cur_df.drop(baseline_key)
        else:
            #Get baseline data
            base_out_dir = f'{method}_100_scale_batch'
            base_df = pd.read_pickle((f'{base_out_dir}/scores.pkl'))
            baseline_row = base_df.loc[[baseline_key]]
        if method == "Harm":
            method_name = "Harmony"
        elif method == "Scan":
            method_name = "Scanorama"
        else:
            method_name = method
        baseline_row = baseline_row.rename(index={baseline_key: method_name + " (Baseline)"})
        cur_df = pd.concat([cur_df, baseline_row])
        batch_del = cur_df["Batch correction"] - baseline_row["Batch correction"].item()
        cur_df["Batch Δ"] = batch_del
        biocons_del = cur_df["Bio conservation"] - baseline_row["Bio conservation"].item()
        cur_df["BioCons Δ"] = biocons_del        
        dfs.append(cur_df)
    final_df = pd.concat(dfs + [type_row])
   
    # Fix some typos
    def rename(s):
        return s.replace("select", "filter").replace("Shil", "Sil").replace("CILISI","cLISI").replace("GraphCons","GraphCon")

    # Apply the function to the index
    final_df.index = final_df.index.to_series().apply(rename)

    out_name = f'tables/Combined_{total_dims}_{approach}_{"biocons" if args.biocons else "batch"}_scores_v1'
    final_df.to_csv(out_name + ".csv")
    
    plot_results_table(final_df, save_path=out_name + ".svg")

