import numpy as np
from sklearn import decomposition
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def pca_datasubset(datasubset, n_components=None, model=None, *, motion_only=None):
    """Return PCA scores and a model that can transform other patient subsets.

    ``motion_only=True`` uses XYZ coordinates from every frame, excluding both
    demographics and the anatomical-label channel. Coordinates are absolute;
    this does not calculate displacements or normalize individual frames.

    When fitting, None defaults to False, preserving the original combined
    demographics + complete-array behavior. When transforming, None inherits
    the fitted model's mode. An explicit conflicting mode raises ValueError.
    """
    if model is not None:
        fitted_mode = getattr(model, "motion_only_", False)
        if motion_only is not None and motion_only != fitted_mode:
            raise ValueError("motion_only must match the mode used to fit PCA.")
        motion_only = fitted_mode
    elif motion_only is None:
        motion_only = False

    if not datasubset.patients:
        raise ValueError("PCA requires at least one patient.")

    if motion_only:
        shape = datasubset.patients[0].data.shape
        if len(shape) != 3 or shape[-1] < 3:
            raise ValueError("Expected patient arrays shaped (frames, points, channels>=3).")
        # Allocate once: avoid stacking copies of every full-resolution patient.
        features = np.empty((len(datasubset.patients), shape[0] * shape[1] * 3), dtype=np.float32)
        for row, patient in enumerate(datasubset.patients):
            if patient.data.shape != shape:
                raise ValueError("All patients must have the same array shape.")
            features[row] = patient.data[..., :3].reshape(-1)
    else:
        csv = datasubset.data.drop(columns=[
            col for col in datasubset.data.columns
            if col == "MI" or col.lower().startswith("unnamed")
        ])
        if model is not None and hasattr(model, "demographic_columns_"):
            csv = csv.loc[:, model.demographic_columns_]
        samples = np.stack([patient.data.ravel() for patient in datasubset.patients])
        features = np.column_stack((csv.to_numpy(dtype=float), samples))

    if model is None:
        model = make_pipeline(StandardScaler(), decomposition.PCA(
            n_components=n_components, random_state=42))
        scores = model.fit_transform(features)
        model.motion_only_ = bool(motion_only)
        model.demographic_columns_ = [] if motion_only else csv.columns.tolist()
    else:
        scores = model.transform(features)
    return scores, model
