"""Patient-local frame repair and compact, correspondence-preserving VAE inputs."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits


DEMOGRAPHIC_COLUMNS = (
    "age", "BMI", "height", "weight", "diastolic_BP", "systolic_BP", "sex",
)


@dataclass(frozen=True)
class PreprocessConfig:
    n_frames: int = 10
    n_points: int = 384
    n_regions: int = 3
    robust_z: float = 6.0
    expansion_ratio: float = 1.75
    spatial_coherence_ratio: float = 3.0
    cyclic: bool = False
    seed: int = 42


class UnrepairableCycleError(ValueError):
    """A cycle has too few trustworthy frames; retain its quality audit."""

    def __init__(self, message, audit):
        super().__init__(message)
        self.audit = audit


def repair_frames(data, *, robust_z=6.0, expansion_ratio=1.75,
                  spatial_coherence_ratio=3.0, cyclic=False):
    """Detect gross expansion/nonfinite frames before any subsampling or scaling.

    High log-scale RMS radius, 99th-percentile radius, maximum radius and bounding
    extent must exceed both median + robust_z * 1.4826*MAD and expansion_ratio
    times their within-patient median. This deliberately ignores normal shrinkage.
    A separate robust centroid-distance check detects large frame translations.
    Isolated temporal spikes are checked with pointwise interpolation residuals:
    these must exceed a six-MAD threshold, three times the median residual and
    15% of typical heart radius. The surrounding frames must agree much better
    with one another than with the suspect frame, avoiding flags on the good
    neighbours of an abnormal frame. No population statistics or MI
    labels enter this procedure. Spatial scatter is also checked independently
    of temporal neighbours: the median distance between consecutive same-region
    point indices, divided by median radius, is clustered per patient with
    two-cluster K-means in log space. The rougher cluster is flagged only when
    its geometric-mean score exceeds the smoother cluster by at least three
    times. Median radius avoids distorting the score with a single distant
    point. This uses the dataset's ordered surface templates and catches
    multi-frame corruption even when the median frame is abnormal.

    Isolated frames become the arithmetic mean of adjacent valid frames; runs
    use linear interpolation between nearest valid frames. Endpoints copy the
    closest valid frame unless cyclic=True. Region IDs are never interpolated.
    More than half the frames flagged is ambiguous and raises for review.
    Returns a new array and a JSON-serializable audit record; never edits source.
    """
    data = np.asarray(data)
    if data.ndim != 3 or data.shape[0] < 3 or data.shape[1] < 1 or data.shape[2] != 4:
        raise ValueError("Expected (frames>=3, points>=1, 4) containing XYZ and region IDs.")
    if robust_z <= 0 or expansion_ratio <= 1 or spatial_coherence_ratio <= 1:
        raise ValueError("robust_z must be positive and expansion/coherence ratios must exceed 1.")
    regions = data[..., 3]
    if not np.isfinite(regions).all() or not np.array_equal(
        regions, np.broadcast_to(regions[0], regions.shape)
    ):
        raise ValueError("Region IDs must be finite and point correspondence stable across frames.")
    xyz = np.asarray(data[..., :3], dtype=np.float64)
    finite = np.isfinite(xyz).all(axis=(1, 2))
    valid_indices = np.flatnonzero(finite)
    if len(valid_indices) < 3:
        raise ValueError("At least three finite frames are required for frame quality analysis.")
    clouds = xyz[finite]
    centres = np.median(clouds, axis=1)
    radii = np.linalg.norm(clouds - centres[:, None], axis=-1)
    stats = np.column_stack((
        np.sqrt(np.mean(radii ** 2, axis=1)),
        np.quantile(radii, .99, axis=1), radii.max(axis=1),
        np.linalg.norm(np.ptp(clouds, axis=1), axis=1),
    ))
    log_stats = np.log(np.maximum(stats, 1e-12))
    med = np.median(log_stats, axis=0)
    mad = 1.4826 * np.median(np.abs(log_stats - med), axis=0)
    thresholds = med + np.maximum(robust_z * mad, np.log(expansion_ratio))
    expanded = (log_stats > thresholds).any(axis=1)
    typical_radius = max(float(np.median(stats[:, 0])), 1e-12)
    centre_distance = np.linalg.norm(centres - np.median(centres, axis=0), axis=-1)
    centre_median = np.median(centre_distance)
    centre_mad = 1.4826 * np.median(np.abs(centre_distance - centre_median))
    translated = centre_distance > max(
        centre_median + robust_z * centre_mad, expansion_ratio * typical_radius,
    )
    bad = ~finite
    bad[valid_indices] = expanded | translated
    adjacent = regions[0, 1:] == regions[0, :-1]
    spatial_scores = np.full(len(xyz), np.nan)
    spatial_threshold = None
    spatial_bad = np.zeros(len(xyz), dtype=bool)
    spatial_cluster_labels = np.full(len(xyz), -1, dtype=int)
    spatial_cluster_centres = []
    spatial_cluster_ratio = 1.
    if adjacent.any():
        neighbour_distance = np.linalg.norm(np.diff(clouds, axis=1)[:, adjacent], axis=-1)
        spatial_scores[finite] = np.median(neighbour_distance, axis=1) / np.maximum(np.median(radii, axis=1), 1e-12)
        log_scores = np.log(np.maximum(spatial_scores[finite], 1e-12)).reshape(-1, 1)
        spatial_cluster_labels[finite] = 0
        if np.ptp(log_scores) > 1e-6:
            # Ten frames are far too small to benefit from an OpenMP thread pool.
            with threadpool_limits(limits=1, user_api="openmp"):
                clusters = KMeans(n_clusters=2, n_init=10, random_state=42).fit(log_scores)
            order = np.argsort(clusters.cluster_centers_[:, 0])
            spatial_cluster_centres = np.exp(clusters.cluster_centers_[order, 0]).tolist()
            spatial_cluster_ratio = float(spatial_cluster_centres[1] / spatial_cluster_centres[0])
            spatial_cluster_labels[finite] = (clusters.labels_ == order[1]).astype(int)
            if spatial_cluster_ratio >= spatial_coherence_ratio:
                spatial_threshold = float(np.sqrt(np.prod(spatial_cluster_centres)))
                spatial_bad = spatial_cluster_labels == 1
        bad |= spatial_bad
    # Some exploding clouds retain almost normal global size: individual points
    # scatter in random directions. Test point correspondence, not only radius.
    residual = np.full(len(xyz), np.nan)
    for frame in range(len(xyz)):
        if cyclic:
            left, right = (frame - 1) % len(xyz), (frame + 1) % len(xyz)
            prediction = .5 * (xyz[left] + xyz[right])
            usable = finite[[left, frame, right]].all()
        elif frame == 0:
            prediction = 2 * xyz[1] - xyz[2]
            usable = finite[:3].all()
        elif frame == len(xyz) - 1:
            prediction = 2 * xyz[-2] - xyz[-3]
            usable = finite[-3:].all()
        else:
            prediction = .5 * (xyz[frame - 1] + xyz[frame + 1])
            usable = finite[frame - 1:frame + 2].all()
        if usable:
            residual[frame] = np.sqrt(np.mean((xyz[frame] - prediction) ** 2))
    usable_residual = residual[np.isfinite(residual)]
    residual_threshold = None
    temporal_bad = np.zeros(len(xyz), dtype=bool)
    if len(usable_residual) >= 3:
        residual_median = np.median(usable_residual)
        residual_mad = 1.4826 * np.median(np.abs(usable_residual - residual_median))
        residual_threshold = float(max(residual_median + robust_z * residual_mad,
                                       3 * residual_median, .15 * typical_radius))
        for frame in np.flatnonzero(residual > residual_threshold):
            if not cyclic and frame in (0, len(xyz) - 1):
                adjacent, further = (1, 2) if frame == 0 else (len(xyz) - 2, len(xyz) - 3)
                suspect_step = np.sqrt(np.mean((xyz[frame] - xyz[adjacent]) ** 2))
                good_step = np.sqrt(np.mean((xyz[adjacent] - xyz[further]) ** 2))
                temporal_bad[frame] = suspect_step > max(3 * good_step, .5 * residual_threshold)
            else:
                left, right = (frame - 1) % len(xyz), (frame + 1) % len(xyz)
                bypass = np.sqrt(np.mean((xyz[left] - xyz[right]) ** 2))
                temporal_bad[frame] = residual[frame] > 1.5 * bypass
    bad |= temporal_bad
    audit = {
        "flagged_frames": np.flatnonzero(bad).tolist(), "replacements": [],
        "finite_frames": finite.tolist(),
        "stat_names": ["rms_radius", "q99_radius", "max_radius", "bounding_extent"],
        "finite_frame_statistics": stats.tolist(),
        "upper_thresholds": np.exp(thresholds).tolist(),
        "spatial_scatter_frames": np.flatnonzero(spatial_bad).tolist(),
        "spatial_coherence_scores": [float(v) if np.isfinite(v) else None for v in spatial_scores],
        "spatial_coherence_threshold": spatial_threshold,
        "spatial_cluster_labels": spatial_cluster_labels.tolist(),
        "spatial_cluster_centres": spatial_cluster_centres,
        "spatial_cluster_ratio": spatial_cluster_ratio,
        "spatial_detection_method": "patient-local KMeans(k=2) on log spatial coherence; separated rough cluster only",
        "temporal_spike_frames": np.flatnonzero(temporal_bad).tolist(),
        "temporal_residuals": [float(value) if np.isfinite(value) else None for value in residual],
        "temporal_residual_threshold": residual_threshold,
        "excluded": False,
    }
    if bad.sum() > len(bad) // 2:
        audit.update(excluded=True, exclusion_reason="More than half the frames are suspect; review this patient manually.")
        raise UnrepairableCycleError(audit["exclusion_reason"], audit)
    good = np.flatnonzero(~bad)
    repaired = np.array(data, dtype=np.float32, copy=True)
    replacements = []
    for frame in np.flatnonzero(bad):
        if cyclic:
            left = int(good[np.argmin((frame - good) % len(bad))])
            right = int(good[np.argmin((good - frame) % len(bad))])
            distance_left = (frame - left) % len(bad)
            distance_right = (right - frame) % len(bad)
        else:
            left = int(good[good < frame][-1]) if (good < frame).any() else int(good[0])
            right = int(good[good > frame][0]) if (good > frame).any() else int(good[-1])
            distance_left, distance_right = abs(frame - left), abs(right - frame)
        weight = distance_left / (distance_left + distance_right) if left != right else 0.
        repaired[frame, :, :3] = (1 - weight) * xyz[left] + weight * xyz[right]
        replacements.append({"frame": int(frame), "left": left, "right": right,
                             "right_weight": float(weight)})
    audit["replacements"] = replacements
    return repaired, audit


def prepare_patient(data, config=PreprocessConfig()):
    """Repair all points, sample equally per region, then centre the entire cycle.

    A fixed seeded sample of point indices per region is shared by all frames.
    Indices within each region retain their original order. Correspondence across
    frames (and ordered templates across patients for the XYZ loss) is assumed.
    One centre is removed per patient, preserving contraction and temporal motion.
    """
    if config.n_points < 1 or config.n_regions < 1 or config.n_points % config.n_regions:
        raise ValueError("n_points must be positive and divisible by n_regions.")
    if np.shape(data)[0] != config.n_frames:
        raise ValueError(f"Expected {config.n_frames} frames.")
    repaired, audit = repair_frames(
        data, robust_z=config.robust_z, expansion_ratio=config.expansion_ratio,
        spatial_coherence_ratio=config.spatial_coherence_ratio,
        cyclic=config.cyclic,
    )
    labels = repaired[0, :, 3]
    if not np.isin(labels, np.arange(config.n_regions)).all():
        raise ValueError("Unexpected anatomical region IDs.")
    rng = np.random.default_rng(config.seed)
    per_region = config.n_points // config.n_regions
    selected = []
    for region in range(config.n_regions):
        candidates = np.flatnonzero(labels == region)
        if len(candidates) < per_region:
            raise ValueError(f"Region {region} has fewer than {per_region} points.")
        selected.extend(np.sort(rng.choice(candidates, per_region, replace=False)))
    selected = np.asarray(selected, dtype=np.int64)
    xyz = repaired[:, selected, :3].copy()
    centre = repaired[..., :3].mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    xyz -= centre
    audit.update({"selected_indices": selected.tolist(), "centre": centre.tolist()})
    return xyz, labels[selected].astype(np.int64), audit


def prepare_arrays(arrays, config=PreprocessConfig()):
    prepared = [prepare_patient(array, config) for array in arrays]
    if not prepared:
        raise ValueError("At least one patient is required.")
    return (np.stack([p[0] for p in prepared]), np.stack([p[1] for p in prepared]),
            [p[2] for p in prepared])


def load_cohort(directory, indices=None, config=PreprocessConfig(), *, exclude_unrepairable=True,
                metadata_filename="mixed_demographics.csv", samples_subdir="mixed_samples",
                labelled_filenames=True):
    """Read only selected patients, one file at a time, without the eager loader.

    Defaults to training_data/mixed_samples naming. For the external final set,
    use test.csv, test_samples and labelled_filenames=False (0.npy, 1.npy, ...).
    Cycles with a majority of suspect frames are excluded by default, with their
    full audit and reason retained. Pass exclude_unrepairable=False to raise.
    Audit includes excluded patients; arrays contain only retained patients in
    the same relative order. Use the non-excluded audit records for patient IDs.
    """
    directory = Path(directory)
    metadata = pd.read_csv(directory / metadata_filename)
    indices = np.arange(len(metadata)) if indices is None else np.asarray(indices, dtype=int)
    rows = metadata.iloc[indices]
    if not rows.MI.isin(["healthy", "pMI"]).all():
        raise ValueError("MI must be healthy or pMI.")
    demographics = rows.loc[:, DEMOGRAPHIC_COLUMNS].copy()
    # CSV bool values can also arrive as strings; never map 'False' to True.
    demographics["sex"] = demographics.sex.map(
        {True: 1., False: 0., "True": 1., "False": 0.}
    )
    demographics = demographics.to_numpy(dtype=np.float32)
    if not np.isfinite(demographics).all():
        raise ValueError("Demographics must be numeric and complete, including a boolean sex value.")
    prepared, kept, audit = [], [], []
    for position, (index, label) in enumerate(zip(indices, rows.MI)):
        path = directory / samples_subdir / f"{index}{label if labelled_filenames else ''}.npy"
        try:
            xyz, regions, record = prepare_patient(np.load(path, mmap_mode="r", allow_pickle=False), config)
        except UnrepairableCycleError as exc:
            if not exclude_unrepairable:
                raise
            record = exc.audit
        else:
            prepared.append((xyz, regions))
            kept.append(position)
        audit.append({"patient_index": int(index), **record})
    if not prepared:
        raise ValueError("No repairable patient cycles remain.")
    if len(kept) < len(indices):
        print(f"Excluded {len(indices) - len(kept)} cycles with a majority of suspect frames; see the quality audit.", flush=True)
    labels = (rows.MI == "pMI").to_numpy(np.int64)
    return (np.stack([p[0] for p in prepared]), np.stack([p[1] for p in prepared]),
            demographics[kept], labels[kept], audit)
