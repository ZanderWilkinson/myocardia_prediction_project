"""Fresh-fold VAE fitting, train-only scaling, checkpoints and raw-array inference."""

from dataclasses import asdict, dataclass
from pathlib import Path
import random
import time

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset

from .vae import CardiacVAE, VAEConfig, vae_loss
from .vae_data import DEMOGRAPHIC_COLUMNS, PreprocessConfig, prepare_arrays


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 60
    batch_size: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 1e-3
    beta: float = .01
    kl_warmup_epochs: int = 10
    classification_weight: float = 1.
    patience: int = 10
    inner_validation_fraction: float = .15
    balance_classes: bool = True
    seed: int = 42
    device: str = "auto"
    cpu_threads: int = 4


@dataclass
class FoldScaler:
    """One physical XYZ scale; one mean/std per head-only demographic feature."""
    xyz_scale: float
    demographic_mean: np.ndarray
    demographic_std: np.ndarray

    @classmethod
    def fit(cls, xyz, demographics):
        scale = float(np.sqrt(np.mean(np.square(xyz, dtype=np.float64))))
        if not np.isfinite(scale) or scale <= 1e-8:
            raise ValueError("Training coordinates have zero or invalid scale.")
        return cls(scale, demographics.mean(axis=0), np.maximum(demographics.std(axis=0), 1e-6))

    def transform(self, xyz, demographics):
        return (np.asarray(xyz / self.xyz_scale, dtype=np.float32),
                np.asarray((demographics - self.demographic_mean) / self.demographic_std,
                           dtype=np.float32))

    def state(self):
        return {"xyz_scale": self.xyz_scale, "demographic_mean": self.demographic_mean.tolist(),
                "demographic_std": self.demographic_std.tolist()}


def select_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _loader(xyz, regions, demographics, labels, scaler, batch_size, *, seed=42, shuffle=False):
    xyz, demographics = scaler.transform(xyz, demographics)
    tensors = [torch.from_numpy(np.ascontiguousarray(array)) for array in
               (xyz, np.asarray(regions, dtype=np.int64), demographics,
                np.asarray(labels, dtype=np.int64))]
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle,
                      generator=torch.Generator().manual_seed(seed), num_workers=0)


def _validate_arrays(xyz, regions, demographics, labels, model_config):
    n = len(xyz)
    if (xyz.shape != (n, model_config.n_frames, model_config.n_points, 3)
            or regions.shape != (n, model_config.n_points)
            or demographics.shape != (n, model_config.demographic_dim)
            or labels.shape != (n,)):
        raise ValueError("Patient array shapes do not match the model configuration.")
    if not all(np.isfinite(a).all() for a in (xyz, regions, demographics, labels)):
        raise ValueError("All model inputs must be finite after preprocessing.")
    if (not np.isin(labels, [0, 1]).all() or not np.equal(regions, regions.astype(int)).all()
            or (regions < 0).any() or (regions >= model_config.n_regions).any()):
        raise ValueError("Expected binary MI labels and integer anatomical IDs in range.")


@torch.inference_mode()
def predict_prepared(model, scaler, xyz, regions, demographics, *, batch_size=8, device=None):
    """Use posterior means, without decoding or stochastic latent sampling."""
    model.eval()
    device = next(model.parameters()).device if device is None else torch.device(device)
    loader = _loader(xyz, regions, demographics, np.zeros(len(xyz), np.int64), scaler, batch_size)
    probabilities, latents = [], []
    for coordinates, anatomy, demographic, _ in loader:
        mu, _ = model.encode(coordinates.to(device), anatomy.to(device))
        probabilities.append(model.classify(mu, demographic.to(device)).sigmoid().cpu().numpy())
        latents.append(mu.cpu().numpy())
    return np.concatenate(probabilities), np.concatenate(latents)


def _classification_scores(labels, prediction):
    """Pool patient predictions before scoring; never average minibatch F1s."""
    return {"accuracy": float(accuracy_score(labels, prediction)),
            "macro_f1": float(f1_score(labels, prediction, average="macro", zero_division=0)),
            "f1_pMI": float(f1_score(labels, prediction, zero_division=0))}


def classification_metrics(labels, probabilities):
    prediction = probabilities >= .5
    return {"roc_auc": float(roc_auc_score(labels, probabilities)),
            "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
            **_classification_scores(labels, prediction)}


def fit_vae(xyz, regions, demographics, labels, train_indices, validation_indices, *,
            model_config=VAEConfig(), train_config=TrainConfig(), verbose=True):
    """Fit using only train_indices; validation selects the best MI BCE epoch.

    A fresh model, optimizer, class weight and scaler are created on every call.
    Validation loss uses the same train-derived class weight. Returned model is
    restored to its best epoch. Never pass an outer scoring fold as validation.
    Each epoch logs pooled patient accuracy, pMI F1 and macro F1 at cutoff 0.5.
    Training metrics use predictions collected during the training pass (dropout
    enabled); validation metrics use the epoch-end model with dropout disabled.
    """
    _validate_arrays(xyz, regions, demographics, labels, model_config)
    c = train_config
    if (c.epochs < 1 or c.batch_size < 1 or c.patience < 1 or c.learning_rate <= 0
            or c.weight_decay < 0 or c.beta < 0 or c.classification_weight <= 0
            or c.kl_warmup_epochs < 0 or c.cpu_threads < 1):
        raise ValueError("Invalid training settings.")
    train_indices, validation_indices = np.asarray(train_indices), np.asarray(validation_indices)
    for indices in (train_indices, validation_indices):
        if indices.ndim != 1 or not len(indices) or len(np.unique(indices)) != len(indices):
            raise ValueError("Train/validation indices must be nonempty, unique 1D arrays.")
        if not np.issubdtype(indices.dtype, np.integer) or indices.min() < 0 or indices.max() >= len(xyz):
            raise ValueError("Patient indices must be integers within the cohort.")
    if np.intersect1d(train_indices, validation_indices).size:
        raise ValueError("Train and validation patients must not overlap.")
    if np.unique(labels[train_indices]).size != 2 or np.unique(labels[validation_indices]).size != 2:
        raise ValueError("Both MI classes are required in train and early-stopping validation.")
    random.seed(c.seed)
    np.random.seed(c.seed)
    torch.manual_seed(c.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(c.seed)
    torch.set_num_threads(c.cpu_threads)
    device = select_device(c.device)
    scaler = FoldScaler.fit(xyz[train_indices], demographics[train_indices])
    def subset_loader(indices, shuffle=False):
        return _loader(xyz[indices], regions[indices], demographics[indices], labels[indices],
                       scaler, c.batch_size, seed=c.seed, shuffle=shuffle)
    train_loader, validation_loader = subset_loader(train_indices, True), subset_loader(validation_indices)
    model = CardiacVAE(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=c.learning_rate, weight_decay=c.weight_decay)
    counts = np.bincount(labels[train_indices].astype(int), minlength=2)
    pos_weight = torch.tensor(counts[0] / counts[1] if c.balance_classes else 1.,
                              device=device, dtype=torch.float32)
    best_loss, best_state, best_epoch, stale = float("inf"), None, 0, 0
    history = []
    for epoch in range(1, c.epochs + 1):
        started = time.monotonic()
        beta = c.beta * min(1., epoch / max(1, c.kl_warmup_epochs))
        model.train()
        totals = dict.fromkeys(("loss", "reconstruction", "kl", "classification"), 0.)
        train_targets, train_predictions = [], []
        for coordinates, anatomy, demographic, target in train_loader:
            coordinates, anatomy, demographic, target = [
                tensor.to(device) for tensor in (coordinates, anatomy, demographic, target)]
            optimizer.zero_grad(set_to_none=True)
            output = model(coordinates, anatomy, demographic)
            train_targets.append(target.detach().cpu().numpy())
            train_predictions.append((output["logits"].detach() >= 0).cpu().numpy())
            losses = vae_loss(output, coordinates, target, beta=beta,
                              classification_weight=c.classification_weight, pos_weight=pos_weight)
            if not torch.isfinite(losses["loss"]):
                raise FloatingPointError("Nonfinite VAE loss; inspect inputs and learning rate.")
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            for key in totals:
                totals[key] += float(losses[key].detach()) * len(target)
        model.eval()
        validation_bce = 0.
        validation_targets, validation_predictions = [], []
        with torch.inference_mode():
            for coordinates, anatomy, demographic, target in validation_loader:
                mu, _ = model.encode(coordinates.to(device), anatomy.to(device))
                logits = model.classify(mu, demographic.to(device))
                validation_targets.append(target.numpy())
                validation_predictions.append((logits >= 0).cpu().numpy())
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, target.to(device).float(), pos_weight=pos_weight)
                validation_bce += float(loss) * len(target)
        validation_bce /= len(validation_indices)
        if not np.isfinite(validation_bce):
            raise FloatingPointError("Nonfinite validation loss.")
        train_scores = _classification_scores(np.concatenate(train_targets), np.concatenate(train_predictions))
        validation_scores = _classification_scores(
            np.concatenate(validation_targets), np.concatenate(validation_predictions))
        record = {"epoch": epoch, "beta": beta,
                  **{key: value / len(train_indices) for key, value in totals.items()},
                  **{f"train_{key}": value for key, value in train_scores.items()},
                  **{f"validation_{key}": value for key, value in validation_scores.items()},
                  "validation_bce": validation_bce, "seconds": time.monotonic() - started}
        history.append(record)
        if verbose:
            print(f"epoch {epoch:03d}: loss={record['loss']:.4f}, "
                  f"XYZ={record['reconstruction']:.4f}, KL={record['kl']:.4f}, "
                  f"MI={record['classification']:.4f}, val MI={validation_bce:.4f} "
                  f"| train acc={train_scores['accuracy']:.4f}, F1(pMI)={train_scores['f1_pMI']:.4f}, "
                  f"macro F1={train_scores['macro_f1']:.4f} "
                  f"| val acc={validation_scores['accuracy']:.4f}, F1(pMI)={validation_scores['f1_pMI']:.4f}, "
                  f"macro F1={validation_scores['macro_f1']:.4f} "
                  f"({record['seconds']:.1f}s)", flush=True)
        if validation_bce < best_loss - 1e-5:
            best_loss, best_epoch, stale = validation_bce, epoch, 0
            best_state = {key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()}
        else:
            stale += 1
        if stale >= c.patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    return model, scaler, {"best_epoch": best_epoch, "history": history,
                           "device": str(device), "parameter_count": model.parameter_count(),
                           "positive_class_weight": float(pos_weight)}


def save_checkpoint(path, model, scaler, *, preprocess_config=PreprocessConfig(), metadata=None):
    """Save inference weights together with every transform and ordered schema."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_config": asdict(model.config), "scaler": scaler.state(),
                "preprocess_config": asdict(preprocess_config),
                "demographic_columns": list(DEMOGRAPHIC_COLUMNS),
                "class_mapping": {"healthy": 0, "pMI": 1}, "metadata": metadata or {},
                "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()}}, path)


def load_checkpoint(path, *, device="auto"):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    model = CardiacVAE(VAEConfig(**saved["model_config"]))
    model.load_state_dict(saved["state_dict"])
    model.to(select_device(device)).eval()
    scaling = saved["scaler"]
    scaler = FoldScaler(scaling["xyz_scale"], np.asarray(scaling["demographic_mean"], np.float32),
                        np.asarray(scaling["demographic_std"], np.float32))
    return model, scaler, saved


def predict_patients(checkpoint, arrays, demographics, *, device="auto", batch_size=8):
    """Predict raw cycles. Demographics must follow checkpoint demographic_columns.

    Returns MI probabilities, latent means and repair audit for each patient.
    """
    model, scaler, saved = load_checkpoint(checkpoint, device=device)
    xyz, regions, audit = prepare_arrays(arrays, PreprocessConfig(**saved["preprocess_config"]))
    demographics = np.asarray(demographics, dtype=np.float32)
    _validate_arrays(xyz, regions, demographics, np.zeros(len(xyz), np.int64), model.config)
    probability, latent = predict_prepared(model, scaler, xyz, regions, demographics, batch_size=batch_size)
    return probability, latent, audit


def cross_validate_vae(xyz, regions, demographics, labels, *, n_splits=5,
                       model_config=VAEConfig(), train_config=TrainConfig(),
                       preprocess_config=PreprocessConfig(), output_dir=None, patient_ids=None,
                       verbose=True):
    """Stratified patient CV; inner holdout stops training, outer fold scores once.

    All supplied patients are development patients; the caller must withhold
    final test patients. Geometry repair/subsampling is patient-local. XYZ and
    demographic scaling are fitted afresh on each fold's inner training set.
    """
    _validate_arrays(xyz, regions, demographics, labels, model_config)
    if n_splits < 2 or np.unique(labels).size != 2 or np.bincount(labels).min() < n_splits:
        raise ValueError("Each of the two classes needs at least n_splits patients (n_splits>=2).")
    if not 0 < train_config.inner_validation_fraction < 1:
        raise ValueError("inner_validation_fraction must lie between zero and one.")
    patient_ids = np.arange(len(labels)) if patient_ids is None else np.asarray(patient_ids)
    if len(patient_ids) != len(labels) or len(np.unique(patient_ids)) != len(labels):
        raise ValueError("Supply exactly one unique patient ID per cycle.")
    oof = np.full(len(labels), np.nan)
    latent = np.zeros((len(labels), model_config.latent_dim), dtype=np.float32)
    folds = []
    splitter = StratifiedKFold(n_splits, shuffle=True, random_state=train_config.seed)
    for fold, (development, score) in enumerate(splitter.split(xyz, labels), 1):
        fit, stop = train_test_split(
            development, test_size=train_config.inner_validation_fraction,
            stratify=labels[development], random_state=train_config.seed + fold,
        )
        if verbose:
            print(f"Fold {fold}/{n_splits}: fit {len(fit)}, stop {len(stop)}, score {len(score)}", flush=True)
        fold_config = TrainConfig(**{**asdict(train_config), "seed": train_config.seed + fold})
        model, scaler, training = fit_vae(
            xyz, regions, demographics, labels, fit, stop,
            model_config=model_config, train_config=fold_config, verbose=verbose,
        )
        oof[score], latent[score] = predict_prepared(
            model, scaler, xyz[score], regions[score], demographics[score],
            batch_size=train_config.batch_size,
        )
        result = {"fold": fold, "metrics": classification_metrics(labels[score], oof[score]),
                  "fit_patient_ids": patient_ids[fit].tolist(),
                  "early_stop_patient_ids": patient_ids[stop].tolist(),
                  "score_patient_ids": patient_ids[score].tolist(), **training}
        if output_dir is not None:
            save_checkpoint(Path(output_dir) / f"fold_{fold}.pt", model, scaler,
                            preprocess_config=preprocess_config, metadata=result)
        folds.append(result)
        del model
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    names = folds[0]["metrics"]
    return {"folds": folds, "oof_probabilities": oof, "oof_latent_means": latent,
            "oof_metrics": classification_metrics(labels, oof),
            "mean_fold_metrics": {key: float(np.mean([f["metrics"][key] for f in folds])) for key in names},
            "std_fold_metrics": {key: float(np.std([f["metrics"][key] for f in folds], ddof=1)) for key in names}}
