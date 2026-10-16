"""Aggregate overlapping points in an existing t-SNE plot into sized category pies."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mcpp.tsne import plot_aggregated_tsne


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coordinates", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--radius-points", type=float, default=3.4)
    parser.add_argument("--gap-points", type=float, default=.8)
    args = parser.parse_args()
    patients = pd.read_csv(args.coordinates)
    if patients.patient_index.isna().any() or not patients.patient_index.is_unique:
        raise ValueError("Supply one unique patient ID per input point.")
    output = args.output_dir or args.coordinates.parent / (args.coordinates.stem + "_aggregated")
    output.mkdir(parents=True, exist_ok=True)
    figure, assignment, records = plot_aggregated_tsne(
        patients[["tsne_1", "tsne_2"]].to_numpy(), patients.MI.to_numpy(),
        radius_points=args.radius_points, gap_points=args.gap_points,
        subtitle="mi-vae-z15 | best validation checkpoint, epoch 683 | perplexity 5 | seed 42")
    patients["marker_id"] = assignment
    markers = pd.DataFrame(records)
    assert markers.n_patients.sum() == len(patients)
    assert markers.n_healthy.sum() == (patients.MI == "healthy").sum()
    assert markers.n_pMI.sum() == (patients.MI == "pMI").sum()
    # Check that every marker's labelled composition agrees with its members.
    for row in markers.itertuples():
        members = patients[patients.marker_id == row.marker_id]
        assert len(members) == row.n_patients
        assert (members.MI == "healthy").sum() == row.n_healthy
        np.testing.assert_allclose(members[["tsne_1", "tsne_2"]].mean(), [row.tsne_1, row.tsne_2])
    for extension in ("png", "pdf"):
        figure.savefig(output / f"training_tsne_perplexity_5_aggregated.{extension}", dpi=240, facecolor="white")
    plt.close(figure)
    patients.to_csv(output / "patients_by_marker.csv", index=False)
    markers.to_csv(output / "aggregated_markers.csv", index=False)
    (output / "aggregation.json").write_text(json.dumps({
        "source_coordinates": str(args.coordinates.resolve()), "n_patients": len(patients),
        "n_markers": len(markers), "largest_marker_patient_count": int(markers.n_patients.max()),
        "category_counts": patients.MI.value_counts().to_dict(), "perplexity": 5,
        "single_patient_radius_points": args.radius_points, "gap_points": args.gap_points,
        "figure_size_inches": [12, 9], "grouping_uses_labels": False,
        "method": "Iteratively merge overlapping display circles; radius scales as sqrt(patient count). "
                  "Centres are group mean coordinates. Pie angles encode MI category proportions. "
                  "Original t-SNE coordinates and all patients are retained in patients_by_marker.csv."
    }, indent=2, allow_nan=False) + "\n")
    print(f"Patients: {len(patients)}; markers: {len(markers)}; largest group: {markers.n_patients.max()}")
    print(f"Plot: {output / 'training_tsne_perplexity_5_aggregated.png'}")


if __name__ == "__main__":
    main()
