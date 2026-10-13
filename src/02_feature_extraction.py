
import numpy as np
import pandas as pd
from pathlib import Path

project_dir = Path(__file__).resolve().parent.parent
mixed_dir = project_dir / "data" / "mixed_samples"
results_dir = project_dir / "results"

results_dir.mkdir(exist_ok=True)

records = []

for file in mixed_dir.glob("*.npy"):

    cloud = np.load(file, allow_pickle=False)

    # Extract patient ID and MI label
    patient_id = int(file.stem.split("healthy")[0]) if "healthy" in file.stem else int(file.stem.split("pMI")[0])

    label = "healthy" if "healthy" in file.stem else "pMI"

    patient = {
        "patient_id": patient_id,
        "MI": label
    }

    for structure in [0, 1, 2]:

        extents = []

        for frame in cloud:

            points = frame[frame[:, 3] == structure, :3]

            extent = np.ptp(points, axis=0)

            extents.append(extent)

        extents = np.array(extents)

        for axis_idx, axis_name in enumerate(["X", "Y", "Z"]):

            values = extents[:, axis_idx]

            patient[f"S{structure}_{axis_name}_mean"] = values.mean()

            patient[f"S{structure}_{axis_name}_change"] = (
                values.max() - values.min()
            )

            patient[f"S{structure}_{axis_name}_relative_change"] = (
                (values.max() - values.min()) / values.max()
                if values.max() > 0 else np.nan
            )

    records.append(patient)

# Create one row per patient
features = pd.DataFrame(records)

# Sort by patient ID
features = features.sort_values("patient_id").reset_index(drop=True)

print("Feature dataset shape:", features.shape)
print(features.head())

# Check class distribution
print("\nMI distribution:")
print(features["MI"].value_counts())

# Save features
features.to_csv(
    results_dir / "anatomical_features.csv",
    index=False
)

print("\nSaved features to results/anatomical_features.csv")
