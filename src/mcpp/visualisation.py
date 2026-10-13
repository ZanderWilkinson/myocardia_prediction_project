
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from mcpp.data_loader import Patient

# # Locate project root using the location of this script
# project_dir = Path(__file__).resolve().parent.parent

# # Data directories
# data_dir = project_dir / "data"
# mixed_dir = data_dir / "mixed_samples"
# test_dir = data_dir / "test_samples" / "test_samples"

# # List .npy files
# mixed_files = sorted(mixed_dir.glob("*.npy"))
# test_files = sorted(test_dir.glob("*.npy"))

# """
# print("Mixed directory exists:", mixed_dir.exists())
# print("Test directory exists:", test_dir.exists())

# print("Mixed samples:", len(mixed_files))
# print("Test samples:", len(test_files))
# print("First five files:", [f.name for f in mixed_files[:5]])
# """
# # Inspect one point cloud
# if mixed_files:
#     cloud = np.load(mixed_files[0], allow_pickle=False)

#     print("\nFile:", mixed_files[0].name)
#     print("Array shape:", cloud.shape)
#     print("Data type:", cloud.dtype)
#     print("First five entries:\n", cloud[:5])
# else:
#     print("\nNo point cloud files found!")


# STEP 2: Visualise one point cloud

# Select the first frame of the first patient

def visualise_cloud(patient: Patient, frame_number):
    frame = patient.data[frame_number]

    # Separate coordinates and anatomical labels
    xyz = frame[:, :3]
    labels = frame[:, 3]

    # Inspect labels
    unique_labels, counts = np.unique(
        labels, return_counts=True
    )

    print("\nAnatomical labels:")
    for label, count in zip(unique_labels, counts):
        print(f"Label {label}: {count} points")

    # Create 3D plot
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")

    for label in unique_labels:
        points = xyz[labels == label]

        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            s=0.3,
            alpha=0.5,
            label=f"Structure {label}"
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.set_title("Patient 0 — Cardiac Anatomy (Frame 0)")
    ax.legend()

    # Keep spatial axes at the same scale
    axis_range = np.ptp(xyz, axis=0)
    ax.set_box_aspect(np.maximum(axis_range, 1e-6))

    plt.tight_layout()
    plt.show()


# STEP 4: Quantify anatomical changes across frames

def visualise_heartbeat(patient: Patient):
    cloud = patient.data

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    structure_names = {
        0: "LV Inner Surface",
        1: "LV Outer Surface",
        2: "Right Ventricle"
    }

    for label, ax in zip(range(3), axes):

        dimensions = []

        for frame_num in range(10):

            frame = cloud[frame_num]

            # Select points belonging to this structure
            points = frame[frame[:, 3] == label, :3]

            # Calculate X, Y, Z spatial extent
            extent = np.ptp(points, axis=0)

            dimensions.append(extent)

        dimensions = np.array(dimensions)

        for axis_num, axis_name in enumerate(["X", "Y", "Z"]):
            ax.plot(
                range(10),
                dimensions[:, axis_num],
                marker="o",
                label=axis_name
            )

        ax.set_title(structure_names[label])
        ax.set_xlabel("Frame")
        ax.set_ylabel("Spatial extent (coordinate units)")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.suptitle("Anatomical Changes Across Cardiac Frames")
    plt.tight_layout()
    plt.show()