"""Extract a saved VAE's training latent means and plot t-SNE by recorded MI category."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from mcpp.tsne import plot_tsne
from mcpp.vae_data import PreprocessConfig, load_cohort
from mcpp.vae_evaluation import _experiment_identity
from mcpp.vae_experiments import atomic_json_save
from mcpp.vae_training import load_checkpoint, predict_prepared


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/training_data")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--perplexity", type=float, default=30.)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=1500)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    if args.batch_size < 1 or args.cpu_threads < 1:
        parser.error("batch-size and cpu-threads must be positive.")
    torch.set_num_threads(args.cpu_threads)
    checkpoint = args.checkpoint.resolve()
    checksum = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    model, scaler, saved = load_checkpoint(checkpoint, device=args.device)
    identity = _experiment_identity(checkpoint.parent, saved)
    context = saved["metadata"]["context"]
    development = np.asarray(identity["patient_ids"])
    train_ids = development[context["train_indices"]]
    validation_ids = development[context["validation_indices"]]
    if len(np.unique(train_ids)) != len(train_ids) or np.intersect1d(train_ids, validation_ids).size:
        raise ValueError("Saved training membership is invalid or overlaps validation patients.")
    if "fit_patient_ids" in identity and not np.array_equal(train_ids, identity["fit_patient_ids"]):
        raise ValueError("Checkpoint training membership differs from experiment metadata.")
    print(f"Preparing {len(train_ids)} training patients from {identity['experiment_name']!r}.", flush=True)
    xyz, regions, demographics, labels, audit = load_cohort(
        args.data_dir, train_ids, PreprocessConfig(**saved["preprocess_config"]), exclude_unrepairable=False)
    np.testing.assert_array_equal([a["patient_index"] for a in audit], train_ids)
    _, latents = predict_prepared(model, scaler, xyz, regions, demographics, batch_size=args.batch_size)
    categories = np.where(labels, "pMI", "healthy")
    print(f"Extracted {latents.shape[1]} latent means per patient. Running t-SNE...", flush=True)
    epoch = saved["metadata"]["epoch"]
    figure, embedding, tsne = plot_tsne(
        latents, categories, perplexity=args.perplexity, random_state=args.seed, max_iter=args.max_iter,
        subtitle=f"{identity['experiment_name']} | best validation checkpoint, epoch {epoch}\n"
                 f"{len(train_ids)} patients | {latents.shape[1]} latent dimensions | perplexity {args.perplexity:g} | seed {args.seed}")
    output = args.output_dir or ROOT / "results/latent_space" / identity["experiment_name"] / checkpoint.stem
    output.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        figure.savefig(output / f"training_tsne.{extension}", dpi=220, facecolor="white")
    plt.close(figure)
    table = pd.DataFrame({"patient_index": train_ids, "MI": categories,
                          "tsne_1": embedding[:, 0], "tsne_2": embedding[:, 1]})
    for i in range(latents.shape[1]):
        table[f"latent_{i + 1}"] = latents[:, i]
    table.to_csv(output / "training_tsne.csv", index=False)
    np.savez_compressed(output / "training_latents.npz", patient_ids=train_ids, labels=labels, latent_means=latents)
    atomic_json_save({"checkpoint": str(checkpoint), "checkpoint_sha256": checksum, "checkpoint_epoch": epoch,
                      "experiment_name": identity["experiment_name"], "split": "training", "n_patients": len(train_ids),
                      "category_counts": table.MI.value_counts().to_dict(), "latent_dim": latents.shape[1],
                      "latent_representation": "posterior means, evaluation mode, original trained scale",
                      "labels_used_for_embedding": False, "demographics_in_embedding": False,
                      "perplexity": args.perplexity, "random_state": args.seed, "max_iter": args.max_iter,
                      "init": "pca", "learning_rate": "auto", "metric": "euclidean",
                      "kl_divergence": float(tsne.kl_divergence_), "iterations": int(tsne.n_iter_),
                      "device": str(next(model.parameters()).device), "model_config": saved["model_config"],
                      "preprocess_config": saved["preprocess_config"],
                      "sklearn_version": __import__("sklearn").__version__}, output / "training_tsne.json")
    print(f"Plot: {output / 'training_tsne.png'}\nPatient coordinates: {output / 'training_tsne.csv'}", flush=True)
    print(f"Category counts: {table.MI.value_counts().to_dict()}", flush=True)


if __name__ == "__main__":
    main()
