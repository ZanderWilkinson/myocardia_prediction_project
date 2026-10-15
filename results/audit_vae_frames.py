"""Audit development-patient frame repair without loading the held-out test set."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from mcpp.vae_data import PreprocessConfig, UnrepairableCycleError, repair_frames


def main():
    root = Path(__file__).resolve().parents[1]
    directory = root / "data/training_data"
    metadata = pd.read_csv(directory / "mixed_demographics.csv")
    development, held_out = train_test_split(
        np.arange(len(metadata)), test_size=.2, stratify=metadata.MI, random_state=42,
    )
    config = PreprocessConfig()
    flagged, errors = [], []
    for index in development:
        path = directory / "mixed_samples" / f"{index}{metadata.MI.iloc[index]}.npy"
        try:
            _, audit = repair_frames(
                np.load(path, mmap_mode="r", allow_pickle=False), robust_z=config.robust_z,
                expansion_ratio=config.expansion_ratio, cyclic=config.cyclic,
                spatial_coherence_ratio=config.spatial_coherence_ratio,
            )
            if audit["flagged_frames"]:
                flagged.append({"patient_index": int(index), **audit})
        except UnrepairableCycleError as exc:
            flagged.append({"patient_index": int(index), **exc.audit})
        except ValueError as exc:
            errors.append({"patient_index": int(index), "error": str(exc)})
    report = {"patients_checked": len(development), "held_out_patients_not_opened": len(held_out),
              "test_patients_excluded": held_out.tolist(),
              "robust_z": config.robust_z, "expansion_ratio": config.expansion_ratio,
              "spatial_coherence_ratio": config.spatial_coherence_ratio,
              "patients_flagged": len(flagged),
              "frames_flagged": sum(len(a["flagged_frames"]) for a in flagged),
              "patients_repaired": sum(not a["excluded"] for a in flagged),
              "frames_repaired": sum(len(a["flagged_frames"]) for a in flagged if not a["excluded"]),
              "patients_excluded": sum(a["excluded"] for a in flagged),
              "flagged": flagged, "errors": errors}
    output = root / "results/vae_frame_quality_audit.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ("test_patients_excluded", "flagged")}, indent=2))


if __name__ == "__main__":
    main()
