"""Compare t-SNE perplexities on saved training latent means; no model retraining."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import trustworthiness

from mcpp.tsne import plot_tsne


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latents", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--perplexities", nargs="+", type=float,
                        default=[5, 10, 20, 30, 50, 80, 120, 180, 250])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=1500)
    args = parser.parse_args()
    with np.load(args.latents, allow_pickle=False) as saved:
        latents, labels, patient_ids = (saved[k].copy() for k in ("latent_means", "labels", "patient_ids"))
    categories = np.where(labels, "pMI", "healthy")
    output = args.output_dir or args.latents.parent / "perplexity_comparison"
    output.mkdir(parents=True, exist_ok=True)
    variance = PCA().fit(latents).explained_variance_ratio_
    rows, embeddings = [], []
    for perplexity in args.perplexities:
        figure, embedding, fitted = plot_tsne(
            latents, categories, perplexity=perplexity, random_state=args.seed, max_iter=args.max_iter,
            subtitle=f"{len(latents)} training patients | {latents.shape[1]} latent dimensions | "
                     f"perplexity {perplexity:g} | seed {args.seed}")
        stem = f"training_tsne_perplexity_{perplexity:g}"
        for extension in ("png", "pdf"):
            figure.savefig(output / f"{stem}.{extension}", dpi=180, facecolor="white")
        plt.close(figure)
        pd.DataFrame({"patient_index": patient_ids, "MI": categories,
                      "tsne_1": embedding[:, 0], "tsne_2": embedding[:, 1]}).to_csv(output / f"{stem}.csv", index=False)
        row = {"perplexity": perplexity, "trustworthiness_10": float(trustworthiness(latents, embedding, n_neighbors=10)),
               "trustworthiness_30": float(trustworthiness(latents, embedding, n_neighbors=30)),
               "kl_divergence": float(fitted.kl_divergence_), "iterations": int(fitted.n_iter_)}
        rows.append(row)
        embeddings.append(embedding)
        print(row, flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(output / "perplexity_metrics.csv", index=False)
    ncols = min(3, len(rows))
    nrows = (len(rows) + ncols - 1) // ncols
    figure_height = 4.15 * nrows + 1.1
    figure, axes = plt.subplots(nrows, ncols, figsize=(14, figure_height), squeeze=False)
    for axis, row, embedding in zip(axes.flat, rows, embeddings):
        for category, colour in (("healthy", "#0072B2"), ("pMI", "#D55E00")):
            mask = categories == category
            axis.scatter(embedding[mask, 0], embedding[mask, 1], c=colour, s=12, alpha=.78,
                         edgecolors="white", linewidths=.15)
        axis.set_title(f"Perplexity {row['perplexity']:g}", loc="left", fontsize=13, fontweight="bold")
        axis.set(xlabel="t-SNE 1", ylabel="t-SNE 2")
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_aspect("equal", adjustable="datalim")
        axis.tick_params(labelsize=8)
        axis.grid(alpha=.15, linewidth=.5)
    for axis in list(axes.flat)[len(rows):]:
        axis.set_visible(False)
    figure.suptitle("t-SNE perplexity comparison: training latent means", fontsize=19, fontweight="bold",
                   y=1 - .12 / figure_height)
    figure.text(.5, 1 - .58 / figure_height, f"{len(latents)} patients | {latents.shape[1]} latent dimensions | same seed {args.seed} "
                 f"and settings | PC1 explains {variance[0]:.2%} of latent variance", ha="center", fontsize=11, color="#53616e")
    handles = [Line2D([], [], linestyle="", marker="o", color=c, markersize=7,
                      label=f"{category} (n={int((categories == category).sum())})")
               for category, c in (("healthy", "#0072B2"), ("pMI", "#D55E00"))]
    figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, 1 - .8 / figure_height), ncol=2, frameon=False)
    figure.text(.5, .12 / figure_height, "Colours show recorded MI categories; category labels do not enter t-SNE. "
                 "Each panel has its own arbitrary coordinate scale.", ha="center", fontsize=9, color="#53616e")
    figure.subplots_adjust(top=1 - 1.5 / figure_height, bottom=.8 / figure_height,
                           left=.065, right=.98, hspace=.34, wspace=.28)
    for extension in ("png", "pdf"):
        figure.savefig(output / f"perplexity_comparison.{extension}", dpi=190, facecolor="white")
    plt.close(figure)
    (output / "perplexity_comparison.json").write_text(json.dumps({
        "latent_source": str(args.latents.resolve()), "n_patients": len(latents),
        "latent_dim": latents.shape[1], "perplexities": args.perplexities,
        "random_state": args.seed, "max_iter": args.max_iter, "init": "pca", "learning_rate": "auto",
        "metric": "euclidean", "latent_scaling": "original trained scale", "labels_used_for_embedding": False,
        "pca_explained_variance_ratio": variance.tolist(), "results": rows,
        "notes": "Trustworthiness assesses local-neighbour preservation without class labels. "
                 "KL divergences at different perplexities are not a common objective for ranking plots."
    }, indent=2, allow_nan=False) + "\n")
    print(f"Comparison: {output / 'perplexity_comparison.png'}", flush=True)


if __name__ == "__main__":
    main()
