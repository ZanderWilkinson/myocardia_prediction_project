"""Resumable patient-fold VAE fitting, train-only scaling and raw-array inference."""

from dataclasses import asdict, dataclass
from pathlib import Path
import random
import time

import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from filelock import Timeout
import torch
from torch.utils.data import DataLoader, TensorDataset

from .vae import CardiacVAE, VAEConfig, vae_loss
from .vae_data import DEMOGRAPHIC_COLUMNS, PreprocessConfig, prepare_arrays
from .vae_experiments import (array_signature, atomic_json_save, atomic_torch_save,
                              begin_invocation, capture_rng, experiment_lock, training_output_log,
                              restore_rng, training_identity, canonical_identity)


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 1
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
    amp: bool = True  # CUDA FP16 autocast + gradient scaling; CPU/MPS stay FP32.
    checkpoint_every: int = 100
    early_stopping: bool = False


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

    @classmethod
    def from_state(cls, state):
        return cls(state["xyz_scale"], np.asarray(state["demographic_mean"], np.float32),
                   np.asarray(state["demographic_std"], np.float32))


def select_device(name="auto"):
    """Prefer CUDA, then MPS, then CPU; fail clearly for unavailable requests."""
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    try:
        device = torch.device(name)
    except (RuntimeError, ValueError) as exc:
        raise ValueError("Use auto, cpu, mps, cuda or a CUDA device index such as cuda:1.") from exc
    if device.type not in {"cpu", "mps", "cuda"}:
        raise ValueError("Supported training devices are cpu, mps and cuda.")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but PyTorch cannot access an NVIDIA GPU. "
                               "Install a CUDA-enabled PyTorch build and check the NVIDIA driver.")
        index = torch.cuda.current_device() if device.index is None else device.index
        if index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device {index} does not exist; detected {torch.cuda.device_count()} GPU(s).")
        device = torch.device("cuda", index)
    elif device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested, but the Apple GPU is unavailable to this process.")
    return device


def _loader(xyz, regions, demographics, labels, scaler, batch_size, *, seed=42,
            shuffle=False, pin_memory=False):
    xyz, demographics = scaler.transform(xyz, demographics)
    tensors = [torch.from_numpy(np.ascontiguousarray(array)) for array in
               (xyz, np.asarray(regions, dtype=np.int64), demographics,
                np.asarray(labels, dtype=np.int64))]
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle,
                      generator=torch.Generator().manual_seed(seed), num_workers=0,
                      pin_memory=pin_memory)


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
    cuda = device.type == "cuda"
    loader = _loader(xyz, regions, demographics, np.zeros(len(xyz), np.int64), scaler,
                     batch_size, pin_memory=cuda)
    probabilities, latents = [], []
    for coordinates, anatomy, demographic, _ in loader:
        mu, _ = model.encode(coordinates.to(device, non_blocking=cuda), anatomy.to(device, non_blocking=cuda))
        probabilities.append(model.classify(mu, demographic.to(device, non_blocking=cuda)).sigmoid().cpu().numpy())
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


def mi_classification_report(labels, probabilities):
    """Report the patient MI diagnosis at cutoff .5, separately from VAE losses."""
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    if (labels.ndim != 1 or labels.shape != probabilities.shape or not len(labels)
            or not np.isfinite(probabilities).all() or not np.isin(labels, [0, 1]).all()
            or (probabilities < 0).any() or (probabilities > 1).any()):
        raise ValueError("Supply one binary MI label and finite probability in [0, 1] per patient.")
    prediction = probabilities >= .5
    return {"n_patients": len(labels), "classification_threshold": .5,
            "class_mapping": {"healthy": 0, "pMI": 1},
            "metrics": classification_metrics(labels, probabilities),
            "per_class": classification_report(labels, prediction, labels=[0, 1],
                                                 target_names=["healthy", "pMI"],
                                                 output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(labels, prediction, labels=[0, 1]).tolist(),
            "confusion_matrix_order": ["healthy", "pMI"]}


def fit_vae(xyz, regions, demographics, labels, train_indices, validation_indices, *,
            model_config=VAEConfig(), train_config=TrainConfig(), verbose=True,
            checkpoint_path=None, checkpoint_context=None, validation_checkpoint_path=None,
            preprocess_config=PreprocessConfig()):
    """Fit using only train_indices; validation selects the best MI BCE epoch.

    New folds initialize their own model, optimizer, class weight and scaler;
    matching checkpoints restore the fold's saved training state.
    Validation loss uses the same train-derived class weight. Returned model is
    restored to its best epoch. Never pass an outer scoring fold as validation.
    Each epoch logs pooled patient accuracy, pMI F1 and macro F1 at cutoff 0.5.
    Training metrics use predictions collected during the training pass (dropout
    enabled); validation metrics use the epoch-end model with dropout disabled.
    CUDA uses FP16 automatic mixed precision and gradient scaling by default;
    the VAE loss is evaluated in FP32. Disable with TrainConfig(amp=False).
    If checkpoint_path exists, resume full training state to the absolute epoch
    target train_config.epochs. Replace that rolling checkpoint every 100 epochs
    by default, and at invocation end or early stopping. Return the best model;
    the rolling checkpoint retains the latest weights for continued training.
    Separately, validation_checkpoint_path retains the highest validation macro
    F1 weights immediately on each strict improvement, including across resumes.
    """
    _validate_arrays(xyz, regions, demographics, labels, model_config)
    c = train_config
    if (c.epochs < 1 or c.batch_size < 1 or c.patience < 1 or c.learning_rate <= 0
            or c.weight_decay < 0 or c.beta < 0 or c.classification_weight <= 0
            or c.kl_warmup_epochs < 0 or c.cpu_threads < 1 or c.checkpoint_every < 1):
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
    cuda = device.type == "cuda"
    amp_enabled = cuda and c.amp
    device_name = torch.cuda.get_device_name(device) if cuda else "Apple GPU (MPS)" if device.type == "mps" else "CPU"
    if verbose:
        print(f"Training on {device_name} ({device}); "
              f"precision={'FP16 mixed' if amp_enabled else 'FP32'}", flush=True)
    checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
    validation_checkpoint_path = Path(validation_checkpoint_path) if validation_checkpoint_path is not None else None
    if (checkpoint_path is not None and validation_checkpoint_path is not None
            and checkpoint_path.resolve() == validation_checkpoint_path.resolve()):
        raise ValueError("Resume and validation-best checkpoints need different paths.")
    context, saved = None, None
    if checkpoint_path is not None or validation_checkpoint_path is not None:
        context = {"model_config": asdict(model_config), "training": training_identity(c),
                   "inputs": array_signature(xyz, regions, demographics, labels),
                   "train_indices": train_indices.tolist(), "validation_indices": validation_indices.tolist(),
                   "extra": checkpoint_context}
        if checkpoint_path is not None and checkpoint_path.exists():
            saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            if saved.get("checkpoint_version") != 1 or canonical_identity(saved.get("context")) != canonical_identity(context):
                raise ValueError("Checkpoint does not match the model, training settings, patient data or split.")
            if saved["epoch"] > c.epochs:
                raise ValueError("Requested epoch target is earlier than the saved checkpoint.")
    scaler = FoldScaler.from_state(saved["scaler"]) if saved else FoldScaler.fit(xyz[train_indices], demographics[train_indices])
    def subset_loader(indices, shuffle=False):
        return _loader(xyz[indices], regions[indices], demographics[indices], labels[indices],
                       scaler, c.batch_size, seed=c.seed, shuffle=shuffle, pin_memory=cuda)
    train_loader, validation_loader = subset_loader(train_indices, True), subset_loader(validation_indices)
    model = CardiacVAE(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=c.learning_rate, weight_decay=c.weight_decay)
    gradient_scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    counts = np.bincount(labels[train_indices].astype(int), minlength=2)
    pos_weight = torch.tensor(counts[0] / counts[1] if c.balance_classes else 1.,
                              device=device, dtype=torch.float32)
    best_loss, best_state, best_epoch, stale = float("inf"), None, 0, 0
    history, last_epoch, start_epoch = [], 0, 0
    if saved is not None:
        model.load_state_dict(saved["state_dict"])
        optimizer.load_state_dict(saved["optimizer"])
        if amp_enabled and saved["gradient_scaler"]:
            gradient_scaler.load_state_dict(saved["gradient_scaler"])
        best_loss, best_state = saved["best_loss"], saved["best_state_dict"]
        best_epoch, stale = saved["best_epoch"], saved["stale"]
        history, last_epoch = saved["history"], saved["epoch"]
        restore_rng(saved["rng"], device, train_loader, validation_loader)
        start_epoch = last_epoch
        if verbose:
            print(f"Resuming saved epoch {last_epoch}; target {c.epochs}; best epoch {best_epoch}.", flush=True)
    best_macro_f1, best_macro_f1_epoch = -1., 0
    if history:
        previous_best = max(history, key=lambda r: r["validation_macro_f1"])
        best_macro_f1, best_macro_f1_epoch = previous_best["validation_macro_f1"], previous_best["epoch"]

    def save_validation_best(record, state_dict=None):
        save_checkpoint(validation_checkpoint_path, model, scaler,
                        preprocess_config=preprocess_config, state_dict=state_dict,
                        metadata={"selection_metric": "validation_macro_f1", "epoch": record["epoch"],
                                  "validation_macro_f1": record["validation_macro_f1"],
                                  "validation_accuracy": record["validation_accuracy"],
                                  "validation_f1_pMI": record["validation_f1_pMI"],
                                  "validation_bce": record["validation_bce"], "context": context})

    if validation_checkpoint_path is not None:
        if validation_checkpoint_path.exists():
            selected = torch.load(validation_checkpoint_path, map_location="cpu", weights_only=True)
            metadata = selected["metadata"]
            if (canonical_identity(metadata.get("context")) != canonical_identity(context)
                    or metadata.get("selection_metric") != "validation_macro_f1"):
                raise ValueError("Validation-best checkpoint does not match this experiment and split.")
            best_macro_f1, best_macro_f1_epoch = metadata["validation_macro_f1"], metadata["epoch"]
            del selected
        elif saved is not None:
            # Older runs only retained the BCE-best weights. Seed from those;
            # earlier unsaved epoch weights cannot be reconstructed from history.
            candidate = next((r for r in history if r["epoch"] == best_epoch), None)
            if candidate is not None:
                save_validation_best(candidate, best_state)
                best_macro_f1, best_macro_f1_epoch = candidate["validation_macro_f1"], best_epoch
    if verbose:
        print(f"Early stopping: {'enabled (patience ' + str(c.patience) + ')' if c.early_stopping else 'disabled'}; "
              f"saved epoch {last_epoch}, requested target {c.epochs}.", flush=True)
    target_epoch = last_epoch if c.early_stopping and stale >= c.patience else c.epochs
    for epoch in range(last_epoch + 1, target_epoch + 1):
        started = time.monotonic()
        beta = c.beta * min(1., epoch / max(1, c.kl_warmup_epochs))
        model.train()
        totals = dict.fromkeys(("loss", "reconstruction", "kl", "classification"), 0.)
        train_targets, train_predictions = [], []
        for coordinates, anatomy, demographic, target in train_loader:
            coordinates, anatomy, demographic, target = [
                tensor.to(device, non_blocking=cuda) for tensor in (coordinates, anatomy, demographic, target)]
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp_enabled):
                output = model(coordinates, anatomy, demographic)
            train_targets.append(target.detach().cpu().numpy())
            train_predictions.append((output["logits"].detach() >= 0).cpu().numpy())
            losses = vae_loss(output, coordinates, target, beta=beta,
                              classification_weight=c.classification_weight, pos_weight=pos_weight)
            if not torch.isfinite(losses["loss"]):
                raise FloatingPointError("Nonfinite VAE loss; inspect inputs and learning rate.")
            gradient_scaler.scale(losses["loss"]).backward()
            gradient_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=not amp_enabled)
            # GradScaler skips an overflowing FP16 step and adjusts its scale.
            gradient_scaler.step(optimizer)
            gradient_scaler.update()
            for key in totals:
                totals[key] += float(losses[key].detach()) * len(target)
        model.eval()
        validation_bce = 0.
        validation_targets, validation_predictions = [], []
        with torch.inference_mode():
            for coordinates, anatomy, demographic, target in validation_loader:
                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp_enabled):
                    mu, _ = model.encode(coordinates.to(device, non_blocking=cuda),
                                         anatomy.to(device, non_blocking=cuda))
                    logits = model.classify(mu, demographic.to(device, non_blocking=cuda))
                validation_targets.append(target.numpy())
                validation_predictions.append((logits >= 0).cpu().numpy())
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits.float(), target.to(device, non_blocking=cuda).float(), pos_weight=pos_weight)
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
        if validation_scores["macro_f1"] > best_macro_f1:
            if validation_checkpoint_path is not None:
                save_validation_best(record)
                if verbose:
                    print(f"Saved {validation_checkpoint_path.name}: validation macro F1 "
                          f"{validation_scores['macro_f1']:.4f} at epoch {epoch}.", flush=True)
            best_macro_f1, best_macro_f1_epoch = validation_scores["macro_f1"], epoch
        last_epoch = epoch
        if checkpoint_path is not None and (
            epoch % c.checkpoint_every == 0 or epoch == target_epoch or (c.early_stopping and stale >= c.patience)
        ):
            atomic_torch_save({
                "checkpoint_version": 1, "context": context, "model_config": asdict(model_config),
                "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                "gradient_scaler": gradient_scaler.state_dict(), "scaler": scaler.state(),
                "epoch": epoch, "history": history, "best_loss": best_loss,
                "best_state_dict": best_state, "best_epoch": best_epoch, "stale": stale,
                "rng": capture_rng(device, train_loader, validation_loader),
            }, checkpoint_path)
        if c.early_stopping and stale >= c.patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    early_stopped = c.early_stopping and stale >= c.patience
    if verbose:
        print(f"Stopped at epoch {last_epoch}: "
              f"{'validation loss did not improve within patience' if early_stopped else 'requested epoch target reached'}; "
              f"best validation epoch {best_epoch}.", flush=True)
    return model, scaler, {"best_epoch": best_epoch, "history": history,
                           "best_validation_macro_f1": best_macro_f1,
                           "best_validation_macro_f1_epoch": best_macro_f1_epoch,
                           "validation_best_checkpoint": str(validation_checkpoint_path) if validation_checkpoint_path else None,
                           "device": str(device), "parameter_count": model.parameter_count(),
                           "device_name": device_name, "amp_enabled": amp_enabled,
                           "completed_epochs": last_epoch, "epochs_this_invocation": last_epoch - start_epoch,
                           "early_stopped": early_stopped,
                           "stop_reason": "early_stopping" if early_stopped else "epoch_target",
                           "positive_class_weight": float(pos_weight)}


def save_checkpoint(path, model, scaler, *, preprocess_config=PreprocessConfig(), metadata=None, state_dict=None):
    """Save inference weights together with every transform and ordered schema."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save({"model_config": asdict(model.config), "scaler": scaler.state(),
                "preprocess_config": asdict(preprocess_config),
                "demographic_columns": list(DEMOGRAPHIC_COLUMNS),
                "class_mapping": {"healthy": 0, "pMI": 1}, "metadata": metadata or {},
                "state_dict": {key: value.detach().cpu() for key, value in
                               (model.state_dict() if state_dict is None else state_dict).items()}}, path)


def load_checkpoint(path, *, device="auto"):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    model = CardiacVAE(VAEConfig(**saved["model_config"]))
    model.load_state_dict(saved["state_dict"])
    model.to(select_device(device)).eval()
    scaler = FoldScaler.from_state(saved["scaler"])
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


def cross_validate_vae(xyz, regions, demographics, labels, *, output_dir=None,
                       experiment_name="default", run_lock=None, **kwargs):
    """Run/resume named CV. train_config.epochs adds epochs to each saved fold.

    Matching data/configuration resumes automatically; an interrupted invocation
    finishes its existing targets before another invocation can add epochs.
    With output_dir=None, train a fresh in-memory experiment.
    """
    if output_dir is None:
        return _cross_validate_vae(xyz, regions, demographics, labels, **kwargs)
    lock = run_lock if run_lock is not None else experiment_lock(output_dir)
    try:
        with lock:
            with training_output_log(Path(output_dir) / "training.log"):
                return _cross_validate_vae(xyz, regions, demographics, labels,
                                           output_dir=Path(output_dir), experiment_name=experiment_name, **kwargs)
    except Timeout as exc:
        raise RuntimeError("This experiment is already being trained by another process.") from exc


def train_holdout_vae(xyz, regions, demographics, labels, *, train_samples=None,
                      validation_fraction=.25, train_indices=None, validation_indices=None, **kwargs):
    """One model, using the development cohort's validation holdout for scoring.

    By default 75% of development patients fit the model (60:20:20 overall when
    the final 20% is held out). The CLI supplies fixed indices to preserve the
    original split before quality exclusions. Validation never fits the scalers.
    These validation scores are used for model selection and are not OOF scores.
    """
    result = cross_validate_vae(xyz, regions, demographics, labels, n_splits=1,
                                _holdout=True, train_samples=train_samples,
                                _holdout_fraction=validation_fraction,
                                _holdout_train=train_indices, _holdout_validation=validation_indices, **kwargs)
    result["validation_probabilities"] = result.pop("oof_probabilities")
    result["validation_latent_means"] = result.pop("oof_latent_means")
    result["validation_metrics"] = result.pop("oof_metrics")
    return result


def _cross_validate_vae(xyz, regions, demographics, labels, *, n_splits=5,
                       model_config=VAEConfig(), train_config=TrainConfig(),
                       preprocess_config=PreprocessConfig(), output_dir=None, patient_ids=None,
                       verbose=True, experiment_name="default", _holdout=False, train_samples=None,
                       _holdout_fraction=.25, _holdout_train=None, _holdout_validation=None):
    """Stratified patient CV; inner holdout stops training, outer fold scores once.

    All supplied patients are development patients; the caller must withhold
    final test patients. Geometry repair/subsampling is patient-local. XYZ and
    demographic scaling are fitted afresh on each fold's inner training set.
    """
    _validate_arrays(xyz, regions, demographics, labels, model_config)
    if ((_holdout and n_splits != 1) or (not _holdout and n_splits < 2)
            or np.unique(labels).size != 2 or np.bincount(labels).min() < max(2, n_splits)):
        raise ValueError("Each of the two classes needs at least n_splits patients (n_splits>=2).")
    if not 0 < train_config.inner_validation_fraction < 1:
        raise ValueError("inner_validation_fraction must lie between zero and one.")
    if train_config.epochs < 1 or train_config.checkpoint_every < 1:
        raise ValueError("epochs and checkpoint_every must be positive.")
    patient_ids = np.arange(len(labels)) if patient_ids is None else np.asarray(patient_ids)
    if len(patient_ids) != len(labels) or len(np.unique(patient_ids)) != len(labels):
        raise ValueError("Supply exactly one unique patient ID per cycle.")
    if _holdout:
        if not 0 < _holdout_fraction < 1:
            raise ValueError("validation_fraction must be between zero and one.")
        if train_samples is not None and (train_samples < 2 or train_samples > len(labels) - 2):
            raise ValueError("train_samples must leave at least two validation patients after quality exclusions.")
        if _holdout_train is not None or _holdout_validation is not None:
            if _holdout_train is None or _holdout_validation is None or train_samples is not None:
                raise ValueError("Supply both fixed split indices, without train_samples.")
            fit, score = np.asarray(_holdout_train), np.asarray(_holdout_validation)
            combined = np.concatenate((fit, score))
            if (fit.ndim != 1 or score.ndim != 1 or not len(fit) or not len(score)
                    or not np.issubdtype(combined.dtype, np.integer)
                    or not np.array_equal(np.sort(combined), np.arange(len(labels)))):
                raise ValueError("Fixed holdout indices must partition all usable patients exactly once.")
        else:
            fit, score = train_test_split(
                np.arange(len(labels)), train_size=train_samples,
                test_size=None if train_samples is not None else _holdout_fraction,
                stratify=labels, random_state=train_config.seed,
            )
        plan = [(fit, score, score)]
    else:
        splitter = StratifiedKFold(n_splits, shuffle=True, random_state=train_config.seed)
        plan = []
        for development, score in splitter.split(xyz, labels):
            fit, stop = train_test_split(
                development, test_size=train_config.inner_validation_fraction,
                stratify=labels[development], random_state=train_config.seed + len(plan) + 1,
            )
            plan.append((fit, stop, score))
    manifest = None
    if output_dir is not None:
        identity = {"experiment_name": experiment_name, "model_config": asdict(model_config),
                    "preprocess_config": asdict(preprocess_config), "training": training_identity(train_config),
                    "input_signature": array_signature(xyz, regions, demographics, labels),
                    "patient_ids": patient_ids.tolist(), "n_splits": n_splits,
                    "demographic_columns": list(DEMOGRAPHIC_COLUMNS)}
        if _holdout:
            identity.update(evaluation_mode="validation_holdout", train_samples=train_samples,
                            validation_fraction=_holdout_fraction,
                            fit_patient_ids=patient_ids[plan[0][0]].tolist(),
                            validation_patient_ids=patient_ids[plan[0][1]].tolist())
        manifest = begin_invocation(output_dir, identity, epochs=train_config.epochs, n_splits=n_splits)
        if verbose:
            print(f"Experiment {experiment_name!r}, invocation {manifest['invocation']['number']}: "
                  f"fold epoch targets {manifest['invocation']['target_epochs']}", flush=True)
    oof = np.full(len(labels), np.nan)
    latent = np.zeros((len(labels), model_config.latent_dim), dtype=np.float32)
    folds = []
    for fold, (fit, stop, score) in enumerate(plan, 1):
        if verbose:
            if _holdout:
                print(f"Single fold: train {len(fit)}, validation {len(score)}; "
                      "validation scores select the best model.", flush=True)
            else:
                print(f"Fold {fold}/{n_splits}: fit {len(fit)}, stop {len(stop)}, score {len(score)}", flush=True)
        target_epochs = manifest["invocation"]["target_epochs"][fold - 1] if manifest else train_config.epochs
        fold_config = TrainConfig(**{**asdict(train_config), "seed": train_config.seed + fold,
                                     "epochs": target_epochs})
        model, scaler, training = fit_vae(
            xyz, regions, demographics, labels, fit, stop,
            model_config=model_config, train_config=fold_config, verbose=verbose,
            checkpoint_path=output_dir / f"fold_{fold}.last.pt" if output_dir is not None else None,
            checkpoint_context={"experiment_name": experiment_name, "preprocess_config": asdict(preprocess_config)},
            validation_checkpoint_path=(output_dir / ("val_best.pt" if _holdout else f"fold_{fold}.val_best.pt")
                                        if output_dir is not None else None),
            preprocess_config=preprocess_config,
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
        if manifest is not None:
            manifest["completed_epochs"][fold - 1] = training["completed_epochs"]
            atomic_json_save(manifest, output_dir / "experiment.json")
        del model
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    names = folds[0]["metrics"]
    if manifest is not None:
        manifest["invocation"]["status"] = "complete"
        atomic_json_save(manifest, output_dir / "experiment.json")
    scored = np.flatnonzero(np.isfinite(oof))
    return {"folds": folds, "oof_probabilities": oof[scored], "oof_latent_means": latent[scored],
            "scored_indices": scored.tolist(),
            "evaluation": "validation_holdout" if _holdout else "out_of_fold",
            "experiment_name": experiment_name, "invocation": manifest["invocation"] if manifest else None,
            "oof_metrics": classification_metrics(labels[scored], oof[scored]),
            "mean_fold_metrics": {key: float(np.mean([f["metrics"][key] for f in folds])) for key in names},
            "std_fold_metrics": {key: float(np.std([f["metrics"][key] for f in folds], ddof=1))
                                 if len(folds) > 1 else 0. for key in names}}
