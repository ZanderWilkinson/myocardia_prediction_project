
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# Load extracted features
project_dir = Path(__file__).resolve().parent.parent

df = pd.read_csv(
    project_dir / "results" / "anatomical_features.csv"
)
"""
print("Dataset shape:", df.shape)

print("\nMI distribution:")
print(df["MI"].value_counts())

# Select LV inner surface features
features = [
    "S0_X_relative_change",
    "S0_Y_relative_change",
    "S0_Z_relative_change"
]

# Convert from wide to long format
plot_data = df.melt(
    id_vars="MI",
    value_vars=features,
    var_name="Feature",
    value_name="Relative Change"
)

# Make names easier to read
plot_data["Feature"] = plot_data["Feature"].replace({
    "S0_X_relative_change": "LV X",
    "S0_Y_relative_change": "LV Y",
    "S0_Z_relative_change": "LV Z"
})

# Visualise distributions
plt.figure(figsize=(11, 6))

sns.violinplot(
    data=plot_data,
    x="Feature",
    y="Relative Change",
    hue="MI",
    split=True,
    inner="quart"
)

plt.title("LV Inner Surface: Healthy vs pMI")
plt.xlabel("Anatomical Dimension")
plt.ylabel("Relative Change Across Frames")
plt.tight_layout()
plt.show()
"""

import numpy as np

# First patient
cloud = np.load(mixed_files[0])

# Compare corresponding point positions
frame_0 = cloud[0, :, :3]
frame_1 = cloud[1, :, :3]

# Distance between same-index points
displacement = np.linalg.norm(
    frame_1 - frame_0,
    axis=1
)

print("Mean same-index displacement:", displacement.mean())
print("Median displacement:", np.median(displacement))
print("Maximum displacement:", displacement.max())

# Compare against randomly paired points
rng = np.random.default_rng(42)
random_order = rng.permutation(len(frame_1))

random_distance = np.linalg.norm(
    frame_1[random_order] - frame_0,
    axis=1
)

print("\nMean random-pair distance:", random_distance.mean())
print("Median random-pair distance:", np.median(random_distance))
