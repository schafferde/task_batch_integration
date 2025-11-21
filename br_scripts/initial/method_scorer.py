import argparse
import numpy as np
import scanpy as sc
import pandas as pd
from scib_metrics.benchmark import Benchmarker, BioConservation, BatchCorrection
import os

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
    plot_df = df.drop(_METRIC_TYPE, axis=0)
    # Sort by total score
    if batch and biocons:
        sort_col = "Total"
    elif batch is not None:
        sort_col = "Batch correction"
    else:
        sort_col = "Bio conservation"
    plot_df = plot_df.sort_values(by=sort_col, ascending=False).astype(np.float64)
    plot_df["Method"] = plot_df.index

    # Split columns by metric type, using df as it doesn't have the new method col
    score_cols = df.columns[df.loc[_METRIC_TYPE] == _AGGREGATE_SCORE]
    other_cols = df.columns[df.loc[_METRIC_TYPE] != _AGGREGATE_SCORE]
    column_definitions = [
        ColumnDefinition("Method", width=1.5, textprops={"ha": "left", "weight": "bold"}),
    ]
    # Circles for the metric values
    column_definitions += [
        ColumnDefinition(
            col,
            title=col.replace(" ", "\n", 1),
            width=1,
            textprops={
                "ha": "center",
                "bbox": {"boxstyle": "circle", "pad": 0.25},
            },
            cmap=cmap_fn(plot_df[col]),
            group=df.loc[_METRIC_TYPE, col],
            formatter="{:.2f}",
        )
        for i, col in enumerate(other_cols)
    ]
    # Bars for the aggregate scores
    column_definitions += [
        ColumnDefinition(
            col,
            width=1,
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
        for i, col in enumerate(score_cols)
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
        fig.savefig(os.path.join(save_path, "scib_results.svg"), facecolor=ax.get_facecolor(), dpi=300)

    return tab


def create_parser():
    parser = argparse.ArgumentParser(
        description='A brief description of your script and what it does.'
    )

    parser.add_argument(
        '--method', '-m',
        type=str,
        required=True,
        help='Name of the method to be evaluated'
    )

    parser.add_argument(
        '--scores', '-s',
        type=str,
        required=True,
        help='Name of the dataframe with per-dimension scores'
    )

    parser.add_argument(
        '--anndata', '-a',
        type=str,
        required=True,
        help='Path to the input AnnData file, with embedding(s) & original data.'
    )

    parser.add_argument(
        '--obsm_key', '-o',
        type=str,
        required=True,
        help='The key of the method in the AnnData\'s obsm'
    )

    parser.add_argument(
        '--biocons', '-b',
        action='store_true',
        help='Use bioconservation metrics instead of batch'
    )

    parser.add_argument(
        '--batch2', '-2',
        action='store_true',
        help='Use only 2 batch metrics (PCR Selection & iLISI)'
    )

    parser.add_argument(
        '--select',
        action='store_true',
        help='Evaluate selection (default - scaling)'
    )

    parser.add_argument(
        '--first_n', '-n',
        type=int,
        default=None,
        help='Optional: Act on only the first n input dimensions'
    )

    parser.add_argument(
        '--baseline', '-e',
        action='store_true',
        help='Optional: Include scoring of baseline'
    )

    parser.add_argument(
        '--add_baseline',
        action='store_true',
        help='Optional: Add baseline to existing scores'
    )

    parser.add_argument(
        '--select_scale', '-ss',
        action='store_true',
        help='Optional: Select and scale the selected columns'
    )

    parser.add_argument(
        '--reduced_dims', '-r',
        type=int,
        default = 100,
        help='Number of dimensions to produce (default: 100)'
    )





    return parser

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()

    adata = sc.read_h5ad(args.anndata)
    batch_key = "batch"
    label_key = "cell_types"



    reduced_dims = args.reduced_dims #100
    orig_emb = adata.obsm[args.obsm_key]
    total_dims = orig_emb.shape[1]
    if args.first_n and args.first_n < total_dims:
        total_dims = args.first_n
        orig_emb = orig_emb[:, :total_dims]

    new_keys = []

    if args.baseline or args.add_baseline:
        new_key = args.method + str(total_dims)
        adata.obsm[new_key] = orig_emb
        print("RUNNING BASELINE", flush=True)
        print(new_key)
        new_keys.append(new_key)
    if not args.add_baseline:
        score_df = pd.read_pickle(args.scores)
        #Last row is strings for metric type
        score_df.drop(score_df.tail(1).index,inplace=True) 

        if args.biocons:
            metrics = {"Bio conservation":"BioCons", "Isolated labels":"IsoLabels", "KMeans NMI":"KMeansNMI", "KMeans ARI":"KMeansARI", "Silhouette label":"ShilLabel", "cLISI":"cLISI"}
        elif args.batch2:
            metrics = {"iLISI":"iLISI", "PCR comparison":"PCRComp"}
        else:
            metrics = {"Batch correction":"Batch", "Silhouette batch":"SilBatch", "iLISI":"iLISI", "KBET":"KBET", "Graph connectivity":"GraphCon", "PCR comparison":"PCRComp"}

        if args.select:
            approach = "select"
        elif args.select_scale:
            approach = "select+scale"
        else:
            approach = "scale"

        for metric, m_key in metrics.items():
            m_scores = score_df[metric].to_numpy(dtype=np.float64)[:total_dims]
            if args.select:
                m_some_cols = np.argpartition(m_scores, -reduced_dims)[-reduced_dims:]
                key = f"{args.method}_{approach}_{m_key}_from{total_dims}"
                adata.obsm[key] = orig_emb[:, m_some_cols]
            else:
                m_scores - m_scores.min()
                m_scores /= m_scores.max()
                if args.select_scale:
                    m_some_cols = np.argpartition(m_scores, -reduced_dims)[-reduced_dims:]
                    key = f"{args.method}_{approach}_{m_key}_from{total_dims}"
                    adata.obsm[key] = orig_emb[:, m_some_cols] * m_scores[m_some_cols]
                else:
                    key = f"{args.method}_{approach}_{m_key}"
                    adata.obsm[key] = orig_emb * m_scores
            new_keys.append(key)

    print("Starting final benchmarking")

    #Final evaluation
    sc_bm = Benchmarker(
        adata,
        batch_key= batch_key,
        label_key= label_key,
        embedding_obsm_keys=new_keys,
        bio_conservation_metrics = BioConservation(),
        batch_correction_metrics = BatchCorrection(),
        n_jobs=24
    )
    sc_bm.benchmark()
    score_df2 = sc_bm.get_results(min_max_scale=False)
    output_dir = f'{args.method}_{total_dims}_{approach}_{"biocons" if args.biocons else "batch"}'

    if args.add_baseline:
        score_df = pd.read_pickle(f'{output_dir}/scores.pkl')
        score_df.drop(score_df.tail(1).index,inplace=True)
        score_df2 = pd.concat([score_df, score_df2])   
    else:
        os.makedirs(output_dir, exist_ok=True)      

    score_df2.to_pickle(f'{output_dir}/scores.pkl')
    
    #sc_bm.plot_results_table(min_max_scale=False, show=False, save_dir=output_dir)
    plot_results_table(score_df2, save_path=output_dir)

