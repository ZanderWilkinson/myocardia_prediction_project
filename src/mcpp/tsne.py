"""Plot a t-SNE projection of patient latent means, coloured by MI category."""

import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, DrawingArea
from matplotlib.patches import Circle, Wedge
import matplotlib.patheffects as path_effects
import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import cdist
from sklearn.manifold import TSNE
from threadpoolctl import threadpool_limits


def plot_tsne(latents, categories, *, perplexity=30., random_state=42,
              max_iter=1500, title="Training patients: VAE latent space", subtitle=""):
    """Return (figure, embedding, fitted_tsne); labels affect colours only.

    Use deterministic posterior means with dropout disabled. Latent coordinates
    retain their trained scale; neither demographics nor MI labels enter t-SNE.
    """
    latents, categories = np.asarray(latents), np.asarray(categories)
    if (latents.ndim != 2 or latents.shape[1] < 2 or len(latents) < 3
            or categories.shape != (len(latents),) or not np.isfinite(latents).all()):
        raise ValueError("Supply finite latent vectors and one category per patient.")
    if not np.isin(categories, ["healthy", "pMI"]).all():
        raise ValueError("MI categories must be healthy or pMI.")
    if not 0 < perplexity < len(latents) or max_iter < 250:
        raise ValueError("Perplexity must be below the patient count; max_iter must be at least 250.")
    tsne = TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto",
                max_iter=max_iter, metric="euclidean", random_state=random_state, n_jobs=4)
    with threadpool_limits(limits=4):
        embedding = tsne.fit_transform(latents)
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False}):
        figure, axis = plt.subplots(figsize=(10, 7.4), layout="constrained")
        figure.suptitle(title, fontsize=18, fontweight="bold", x=.04, ha="left")
        axis.set_title(subtitle, loc="left", fontsize=10, color="#53616e", pad=15)
        for category, colour in (("healthy", "#0072B2"), ("pMI", "#D55E00")):
            mask = categories == category
            axis.scatter(embedding[mask, 0], embedding[mask, 1], s=36, c=colour,
                         alpha=.78, linewidths=.4, edgecolors="white",
                         label=f"{category} (n={int(mask.sum())})", zorder=3)
        axis.set(xlabel="t-SNE 1", ylabel="t-SNE 2")
        axis.grid(alpha=.18, linewidth=.6, zorder=0)
        axis.legend(title="Recorded MI category", loc="upper left", bbox_to_anchor=(1.01, 1),
                    frameon=False, borderaxespad=0)
        axis.margins(.07)
        figure.supxlabel("Each point represents one training patient. t-SNE shows local relationships in a 2D projection.",
                         fontsize=9, color="#53616e")
    return figure, embedding, tsne


def plot_aggregated_tsne(embedding, categories, *, radius_points=3.4, gap_points=.8,
                         title="Training latent space: t-SNE", subtitle=""):
    """Merge visually overlapping markers without changing the saved embedding.

    Marker centres are group means. Areas are proportional to patient counts;
    pie slices show category proportions. Groups use chart geometry only and
    merge until their circles no longer overlap. Faint dots retain all original
    positions. The aggregation depends on figure size, which is kept fixed.
    Returns figure, patient-to-marker assignments, and per-marker records.
    """
    embedding, categories = np.asarray(embedding), np.asarray(categories)
    if (embedding.ndim != 2 or embedding.shape[1] != 2 or not len(embedding)
            or categories.shape != (len(embedding),) or not np.isfinite(embedding).all()
            or not np.isin(categories, ["healthy", "pMI"]).all()
            or radius_points <= 0 or gap_points < 0):
        raise ValueError("Supply finite 2D coordinates, MI categories and valid marker sizes.")
    figure = plt.figure(figsize=(12, 9), facecolor="white")
    axis = figure.add_axes([.075, .14, .69, .68])
    figure.text(.075, .955, title, fontsize=20, fontweight="bold", ha="left")
    figure.text(.075, .911, subtitle, fontsize=11, color="#53616e")
    figure.text(.075, .875, "Marker area shows patient count; coloured slices show MI proportions.",
                fontsize=10.5, color="#53616e")
    axis.scatter(embedding[:, 0], embedding[:, 1], s=5, c="#B6C1CB", alpha=.6, linewidths=0, zorder=1)
    axis.margins(.12)
    axis.set_aspect("equal", adjustable="datalim")
    axis.set(xlabel="t-SNE 1", ylabel="t-SNE 2")
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(alpha=.15, linewidth=.6, zorder=0)
    figure.canvas.draw()
    # Work in physical points, so overlap and bubble radii use the same units.
    screen_points = axis.transData.transform(embedding) * 72 / figure.dpi
    groups = [np.array([i]) for i in range(len(embedding))]
    while True:
        centres = np.array([screen_points[g].mean(axis=0) for g in groups])
        radii = radius_points * np.sqrt([len(g) for g in groups])
        overlaps = cdist(centres, centres) < radii[:, None] + radii[None, :] + gap_points
        n_groups, component = connected_components(overlaps, directed=False)
        if n_groups == len(groups):
            break
        groups = [np.concatenate([g for g, label in zip(groups, component) if label == j])
                  for j in range(n_groups)]
    assignments = np.empty(len(embedding), dtype=int)
    records = []
    for marker_id, group in enumerate(groups, 1):
        assignments[group] = marker_id
        n = len(group)
        healthy = int(np.sum(categories[group] == "healthy"))
        centre = embedding[group].mean(axis=0)
        radius = radius_points * np.sqrt(n)
        diameter = 2 * radius + 2
        drawing = DrawingArea(diameter, diameter, 0, 0)
        origin = (radius + 1, radius + 1)
        angle = 90.
        for count, colour in ((healthy, "#0072B2"), (n - healthy, "#D55E00")):
            if count:
                end = angle + 360 * count / n
                drawing.add_artist(Wedge(origin, radius, angle, end, facecolor=colour,
                                         edgecolor="white", linewidth=.6))
                angle = end
        drawing.add_artist(Circle(origin, radius, fill=False, edgecolor="#425465", linewidth=.45))
        axis.add_artist(AnnotationBbox(drawing, centre, frameon=False, pad=0, zorder=3))
        if n >= 3:
            text = axis.text(*centre, str(n), ha="center", va="center", fontsize=7.5,
                             fontweight="bold", color="#16232D", zorder=4)
            text.set_path_effects([path_effects.withStroke(linewidth=2, foreground="white")])
        records.append({"marker_id": marker_id, "tsne_1": float(centre[0]), "tsne_2": float(centre[1]),
                        "n_patients": n, "n_healthy": healthy, "n_pMI": n - healthy,
                        "radius_points": float(radius)})
    legend = figure.add_axes([.81, .18, .18, .64])
    legend.set(xlim=(0, 1), ylim=(0, 1))
    legend.axis("off")
    legend.text(0, .97, "MI category", fontsize=12, fontweight="bold")
    for y, category, colour in ((.90, "healthy", "#0072B2"), (.83, "pMI", "#D55E00")):
        legend.plot(.05, y, "o", color=colour, markersize=8)
        legend.text(.15, y, f"{category} (n={int((categories == category).sum())})", va="center", fontsize=10)
    legend.text(0, .68, "Patients per marker", fontsize=11, fontweight="bold")
    for y, n in zip((.58, .45, .31, .15), (1, 5, 10, 20)):
        radius = radius_points * np.sqrt(n)
        drawing = DrawingArea(2 * radius + 2, 2 * radius + 2, 0, 0)
        drawing.add_artist(Circle((radius + 1, radius + 1), radius, facecolor="#E7EDF2",
                                  edgecolor="#617282", linewidth=.7))
        legend.add_artist(AnnotationBbox(drawing, (.12, y), xycoords="axes fraction", frameon=False, pad=0))
        legend.text(.35, y, str(n), va="center", fontsize=11)
    figure.text(.075, .067, f"{len(embedding)} patients represented by {len(groups)} markers. "
                 "Numbers inside larger markers give patient counts.", fontsize=10, color="#53616e")
    figure.text(.075, .036, "Faint grey dots retain the original positions. "
                 "Aggregation changes the display only; group membership does not use MI labels.",
                 fontsize=9, color="#53616e")
    return figure, assignments, records
