"""Draw the CardiacVAE architecture as editable vector and high-resolution figures."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from mcpp.vae import VAEConfig


ROOT = Path(__file__).resolve().parents[1]
INK = "#192D3B"
MUTED = "#526573"
EDGE = "#D5DEE4"
TEAL = "#087F87"
PURPLE = "#7951B0"
AMBER = "#AE6C17"
ROSE = "#AE4662"


def draw_architecture(config, config_label="VAEConfig"):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "svg.fonttype": "none", "pdf.fonttype": 42,
        "mathtext.fontset": "dejavusans",
    })
    fig, ax = plt.subplots(figsize=(23.4, 11.6))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set(xlim=(0, 23.4), ylim=(0, 11.6))
    ax.axis("off")
    fig.patch.set_facecolor("white")
    texts = []

    def txt(x, y, s, size=11, color=INK, weight="normal", ha="center", **kwargs):
        text = ax.text(x, y, s, fontsize=size, color=color, weight=weight,
                       ha=ha, va="center", linespacing=1.55, **kwargs)
        texts.append(text)
        return text

    def box(x, y, w, h, fill="white", edge=EDGE, lw=1.1, radius=.10):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
            facecolor=fill, edgecolor=edge, linewidth=lw, zorder=1,
        ))

    def arrow(x1, y1, x2, y2, color=MUTED, width=1.5):
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12,
            linewidth=width, color=color, shrinkA=3, shrinkB=3, zorder=3,
        ))

    def card(x, width, title, body, color=TEAL, fill="#F2F9F9", size=10.5):
        box(x, 7.0, width, 2.65, fill=fill, edge=color)
        txt(x + width / 2, 9.24, title, 12.2, color, "bold")
        ax.plot([x + .16, x + width - .16], [8.91, 8.91], color=color, alpha=.22, lw=1)
        txt(x + width / 2, 7.96, body, size)

    c = config
    t, n, s, d, z = c.n_frames, c.n_points, c.spatial_tokens, c.width, c.latent_dim
    txt(.6, 11.00, "Cardiac variational autoencoder", 27, weight="bold", ha="left")
    txt(.6, 10.51, "Spatial–temporal attention encodes a cardiac cycle; the latent mean also supports MI prediction.",
        13, MUTED, ha="left")
    txt(22.8, 11.03, "MODEL ARCHITECTURE", 10, MUTED, "bold", ha="right")

    # The top row follows the actual order of encode(), forward(), and decode().
    for left, right, label, color in [
        (.6, 2.9, "INPUT", MUTED), (3.35, 11.15, "ENCODER", TEAL),
        (11.60, 17.05, "GAUSSIAN LATENT SPACE", PURPLE),
        (17.5, 20.15, "DECODER", AMBER), (20.60, 22.8, "OUTPUT", MUTED),
    ]:
        txt(left, 10.04, label, 10.5, color, "bold", ha="left")
        ax.plot([left, right], [9.83, 9.83], color=color, lw=2)

    card(.6, 2.3, "Cardiac cycle", f"{t} frames\n{n} points per frame\n3 XYZ coordinates\n{c.n_regions} anatomical regions",
         color=MUTED, fill="#F6F8FA", size=11)
    txt(1.75, 6.65, rf"$X:\ {t}\times {n}\times 3$", 12)
    txt(1.75, 6.27, f"Region IDs: {n}", 10.5, MUTED)

    card(3.35, 2.30, "Point embedding",
         f"Region ID → 16 features\nXYZ + embedding: 19\nLinear: 19 → {d}\nGELU · LayerNorm\n+ learned point position", size=10.0)
    txt(4.5, 6.65, rf"${t}\times {n}\times {d}$", 12)

    card(6.1, 2.35, "Spatial pooling",
         f"{s} learned spatial queries\nCross-attend to {n} points\nwithin each frame\nResidual + LayerNorm\n+ learned time position", size=10.0)
    txt(7.275, 6.65, rf"${t}\times {s}\times {d}$", 12)

    card(8.9, 2.25, f"Axial encoder ×{c.encoder_depth}",
         "Spatial attention\nthen temporal attention\n\nLayerNorm + mean pool\nover time and spatial slots", size=9.8)
    txt(10.025, 6.65, rf"$h\in\mathbb{{R}}^{{{d}}}$", 12)

    box(11.60, 7.0, 2.55, 2.65, fill="#F7F3FB", edge=PURPLE)
    txt(12.875, 9.24, "Posterior heads", 12.2, PURPLE, "bold")
    txt(12.875, 8.85, f"Two linear maps: {d} → {z}", 10, MUTED)
    box(11.82, 8.12, 2.11, .48, fill="white", edge="#CDBCE3")
    txt(12.875, 8.36, r"$\ell=\log\sigma^2$", 13, PURPLE)
    box(11.82, 7.35, 2.11, .48, fill="white", edge="#CDBCE3")
    txt(12.875, 7.59, r"$\mu$  (posterior mean)", 11.5, PURPLE)
    txt(13.08, 6.65, rf"$\mu,\ell\in\mathbb{{R}}^{{{z}}}$", 11, PURPLE, ha="left")

    box(14.60, 7.0, 2.45, 2.65, fill="#F7F3FB", edge=PURPLE)
    txt(15.825, 9.24, "Latent sampling", 12.2, PURPLE, "bold")
    txt(15.825, 8.52, r"$\epsilon\sim\mathcal{N}(0,I)$", 14, PURPLE)
    txt(15.825, 7.93, r"$z=\mu+e^{\ell/2}\odot\epsilon$", 13.5)
    txt(15.825, 7.39, "Reparameterization", 10.5, MUTED)
    txt(15.825, 6.65, rf"$z\in\mathbb{{R}}^{{{z}}}$", 12, PURPLE)
    txt(15.825, 6.27, r"Evaluation: $z=\mu$", 10.5, MUTED)

    card(17.50, 2.65, f"Axial decoder ×{c.decoder_depth}",
         f"Linear: {z} → {t*s*d:,}\nReshape: {t} × {s} × {d}\n+ learned time position\nSpatial → temporal attention\nLayerNorm\nLinear: {d} → {n//s*3}",
         color=AMBER, fill="#FFF8EE", size=10.0)
    txt(18.825, 6.65, rf"${t}\times {s}\times {n//s*3}$", 12)
    txt(18.825, 6.27, f"{n//s} points generated per slot", 10, MUTED)

    card(20.60, 2.20, "Reconstruction",
         "Reshape into\nordered XYZ\n\nSame sampled points\nas the input cycle",
         color=MUTED, fill="#F6F8FA", size=10.5)
    txt(21.70, 6.65, rf"$\hat{{X}}:\ {t}\times {n}\times 3$", 12)

    for x1, x2 in [(2.9, 3.35), (5.65, 6.1), (8.45, 8.9), (11.15, 11.60),
                   (17.05, 17.50), (20.15, 20.60)]:
        arrow(x1, 8.32, x2, 8.32)
    arrow(14.15, 8.36, 14.60, 8.36, PURPLE)
    arrow(14.15, 7.59, 14.60, 7.59, PURPLE)

    # Explain the shared axial block without duplicating the full transformer.
    box(.6, 2.63, 6.8, 2.95, fill="#F7FAFB")
    txt(.85, 5.21, "INSIDE EACH AXIAL BLOCK", 10.5, TEAL, "bold", ha="left")
    box(.90, 3.60, 2.75, 1.12, fill="white", edge="#A9CBCD")
    txt(2.275, 4.35, "Spatial transformer", 11.5, TEAL, "bold")
    txt(2.275, 3.92, f"{s} slots within each frame", 10.3)
    arrow(3.65, 4.16, 4.27, 4.16, TEAL)
    box(4.27, 3.60, 2.80, 1.12, fill="white", edge="#A9CBCD")
    txt(5.67, 4.35, "Temporal transformer", 11.5, TEAL, "bold")
    txt(5.67, 3.92, f"{t} frames within each slot", 10.3)
    txt(4.0, 3.15, f"Each: {c.heads} heads · width {d} · feed-forward {c.feedforward} · GELU",
        10.5, MUTED)
    txt(4.0, 2.82, "Pre-norm, residual connections and dropout", 10, MUTED)

    # Only the posterior mean and standardized demographics enter the MI head.
    box(7.85, 2.63, 14.95, 2.95, fill="#FCF6F8", edge="#E5CCD4")
    txt(8.10, 5.21, "MI CLASSIFICATION", 10.5, ROSE, "bold", ha="left")
    box(8.10, 3.27, 3.4, 1.39, fill="white", edge="#D7A8B6")
    txt(9.80, 4.29, "7 demographics", 12, ROSE, "bold")
    txt(9.80, 3.76, "Age, BMI, height, weight,\ndiastolic BP, systolic BP, sex", 9.9)
    txt(9.80, 2.93, "Standardized on training data", 10, MUTED)
    box(12.05, 3.27, 2.80, 1.39, fill="white", edge="#D7A8B6")
    txt(13.45, 4.29, "Concatenate", 12, ROSE, "bold")
    txt(13.45, 3.77, rf"$[\mu,\,d_{{\mathrm{{demo}}}}]\in\mathbb{{R}}^{{{z+c.demographic_dim}}}$", 13)
    arrow(11.5, 3.97, 12.05, 3.97, ROSE)
    # The branch leaves the bottom of the mean head, never the log-variance head.
    arrow(12.875, 7.35, 12.875, 4.66, PURPLE)
    txt(13.05, 5.95, "Mean μ", 10.5, PURPLE, ha="left")
    box(15.40, 3.27, 3.70, 1.39, fill="white", edge="#D7A8B6")
    txt(17.25, 4.29, "Classification MLP", 12, ROSE, "bold")
    txt(17.25, 3.82, f"{z+c.demographic_dim} → 128 → 64 → 1 logit", 12)
    txt(17.25, 2.93, "GELU + dropout after each hidden layer", 10, MUTED)
    arrow(14.85, 3.97, 15.4, 3.97, ROSE)
    box(19.65, 3.27, 2.85, 1.39, fill="white", edge="#D7A8B6")
    txt(21.075, 4.29, "MI probability", 12, ROSE, "bold")
    txt(21.075, 3.77, "Sigmoid → P(pMI)", 12)
    txt(21.075, 2.93, "Healthy = 0  ·  pMI = 1", 10, MUTED)
    arrow(19.1, 3.97, 19.65, 3.97, ROSE)

    # Loss reductions match vae_loss(): all-element XYZ MSE and per-dimension KL.
    ax.plot([.6, 22.8], [2.18, 2.18], color=EDGE, lw=1)
    txt(.6, 1.73, "TRAINING OBJECTIVE", 10.5, MUTED, "bold", ha="left")
    txt(9.9, 1.70,
        r"$\mathcal{L}=\mathrm{MSE}(X,\hat{X})"
        r"+\beta\,\overline{D_{\mathrm{KL}}}(q_\phi(z\mid X,r)\,\|\,\mathcal{N}(0,I))"
        r"+\lambda\,\mathrm{BCEWithLogits}(y,a)$", 16)
    txt(20.75, 1.70, "Reconstruction + KL\n+ supervised classification", 10.5, MUTED)
    txt(.6, 1.05,
        "MSE averages all XYZ values; KL averages patients and latent dimensions.  "
        "Default training: β warms to 0.01 over 10 epochs; λ = 1; BCE uses training-derived class weighting.",
        10.4, MUTED, ha="left")
    txt(.6, .55,
        f"{config_label}: latent dimension {z}, dropout {c.dropout:g}.  "
        "Tensor shapes omit the batch dimension.  Demographics enter the classification head only.",
        10.4, MUTED, ha="left")
    txt(22.8, .55, "Source: src/mcpp/vae.py", 9.5, MUTED, ha="right")

    # Prevent accidental clipping when changing the latent dimension or labels.
    fig.canvas.draw()
    bounds = fig.bbox
    for text in texts:
        extent = text.get_window_extent(fig.canvas.get_renderer())
        if extent.x0 < bounds.x0 or extent.x1 > bounds.x1 or extent.y0 < bounds.y0 or extent.y1 > bounds.y1:
            raise ValueError(f"Text extends outside figure: {text.get_text()}")
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latent-dim", type=int)
    parser.add_argument("--model-report", type=Path, help="JSON report containing model_config")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/figures")
    args = parser.parse_args()
    if args.latent_dim is not None and args.latent_dim < 1:
        parser.error("--latent-dim must be positive")
    config, label = VAEConfig(), "VAEConfig"
    if args.model_report:
        config = VAEConfig(**json.loads(args.model_report.read_text())["model_config"])
        label = "Saved model"
    if args.latent_dim is not None:
        config = replace(config, latent_dim=args.latent_dim)
    fig = draw_architecture(config, label)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg", "pdf"):
        path = args.output_dir / f"vae_architecture.{extension}"
        fig.savefig(path, dpi=240, facecolor="white")
        print(path)
    preview = args.output_dir / "vae_architecture_preview.png"
    fig.savefig(preview, dpi=110, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
