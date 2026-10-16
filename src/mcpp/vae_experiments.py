"""Atomic rolling checkpoints and identities for resumable VAE experiments."""

from dataclasses import asdict
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from contextvars import ContextVar
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import traceback
from uuid import uuid4

from filelock import FileLock
import numpy as np
import torch


_active_log = ContextVar("vae_training_log", default=None)


class _TeeStream:
    def __init__(self, terminal, log):
        self.terminal, self.log = terminal, log

    def write(self, text):
        self.terminal.write(text)
        self.log.write(text)
        self.log.flush()
        return len(text)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def __getattr__(self, name):
        return getattr(self.terminal, name)


@contextmanager
def training_output_log(path):
    """Append terminal output immediately, including errors and interrupted runs."""
    path = Path(path).resolve()
    if _active_log.get() == path:
        yield
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", buffering=1) as log:
        token = _active_log.set(path)
        try:
            with redirect_stdout(_TeeStream(sys.stdout, log)), redirect_stderr(_TeeStream(sys.stderr, log)):
                print(f"\n=== Training session {datetime.now(timezone.utc).isoformat()} ===", flush=True)
                try:
                    yield
                except BaseException:
                    traceback.print_exc(file=log)
                    raise
        finally:
            _active_log.reset(token)
            log.flush()


def experiment_lock(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # OS-backed locks release on process exit, including interrupted jobs.
    return FileLock(directory / ".experiment.lock", timeout=0)


def _atomic_write(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_torch_save(value, path):
    """Replace the previous checkpoint only after its successor is fully written."""
    _atomic_write(path, lambda stream: torch.save(value, stream))


def atomic_json_save(value, path):
    payload = json.dumps(value, indent=2, allow_nan=False).encode()
    _atomic_write(path, lambda stream: stream.write(payload))


def training_identity(config):
    # Runtime placement/precision and checkpoint cadence can change on resume.
    ignored = {"epochs", "device", "cpu_threads", "amp", "checkpoint_every", "patience", "early_stopping"}
    return {key: value for key, value in asdict(config).items() if key not in ignored}


def canonical_identity(value):
    """Allow runtime stopping controls to change, including in older checkpoints."""
    if not isinstance(value, dict):
        return value
    result = {key: canonical_identity(item) for key, item in value.items()}
    if isinstance(result.get("training"), dict):
        result["training"] = {k: v for k, v in result["training"].items()
                              if k not in {"patience", "early_stopping"}}
    return result


def array_signature(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        array = np.ascontiguousarray(array)
        digest.update(f"{array.dtype}:{array.shape}:".encode())
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def capture_rng(device, train_loader, validation_loader):
    numpy_state = np.random.get_state()
    device_rng = None
    if device.type == "cuda":
        device_rng = torch.cuda.get_rng_state(device)
    elif device.type == "mps":
        device_rng = torch.mps.get_rng_state()
    return {"python": random.getstate(),
            "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
            "torch": torch.get_rng_state(), "device_type": device.type,
            "device_rng": device_rng,
            "train_loader": train_loader.generator.get_state(),
            "validation_loader": validation_loader.generator.get_state()}


def restore_rng(state, device, train_loader, validation_loader):
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))
    torch.set_rng_state(state["torch"])
    if state["device_type"] == device.type and state["device_rng"] is not None:
        if device.type == "cuda":
            torch.cuda.set_rng_state(state["device_rng"], device)
        elif device.type == "mps":
            torch.mps.set_rng_state(state["device_rng"])
    train_loader.generator.set_state(state["train_loader"])
    validation_loader.generator.set_state(state["validation_loader"])


def begin_invocation(directory, identity, *, epochs, n_splits):
    """Continue interrupted targets, or schedule additional epochs after completion.

    Caller holds the experiment lock for the entire invocation. An interrupted
    invocation retains its original per-fold targets so completed folds cannot
    receive extra epochs just because the command was restarted.
    """
    path = Path(directory) / "experiment.json"
    if path.exists():
        manifest = json.loads(path.read_text())
        if manifest.get("version") != 1 or canonical_identity(manifest.get("identity")) != canonical_identity(identity):
            raise ValueError("Experiment settings or patient data differ from the saved experiment. "
                             "Use a new experiment name for different data, splits or hyperparameters.")
        for fold, completed in enumerate(manifest["completed_epochs"], 1):
            if completed and not (Path(directory) / f"fold_{fold}.last.pt").exists():
                raise ValueError(f"Training checkpoint for fold {fold} is missing; refusing to restart "
                                 "a completed fold silently. Restore its checkpoint or use a new experiment name.")
        if manifest["invocation"]["status"] == "running":
            if manifest["invocation"]["epochs_to_add"] != epochs:
                invocation = manifest["invocation"]
                targets = [target - invocation["epochs_to_add"] + epochs
                           for target in invocation["target_epochs"]]
                for fold, target in enumerate(targets, 1):
                    checkpoint = Path(directory) / f"fold_{fold}.last.pt"
                    completed = manifest["completed_epochs"][fold - 1]
                    if checkpoint.exists():
                        completed = max(completed, torch.load(checkpoint, map_location="cpu",
                                                              weights_only=True, mmap=True)["epoch"])
                    if target < completed:
                        raise ValueError(f"The new epoch budget would put fold {fold}'s target before "
                                         f"its saved epoch {completed}. Increase --epochs instead.")
                invocation.update(epochs_to_add=epochs, target_epochs=targets)
                atomic_json_save(manifest, path)
            return manifest
        number = manifest["invocation"]["number"] + 1
        completed = manifest["completed_epochs"]
    else:
        existing = {p.name for p in Path(directory).iterdir()}
        if existing - {".experiment.lock", "frame_quality.json", "training.log"}:
            raise ValueError("This directory contains older outputs without resumable experiment state. "
                             "Use a new experiment name or output directory.")
        number, completed = 1, [0] * n_splits
    manifest = {"version": 1, "identity": identity, "completed_epochs": completed,
                "invocation": {"number": number, "status": "running", "epochs_to_add": epochs,
                               "target_epochs": [epoch + epochs for epoch in completed]}}
    atomic_json_save(manifest, path)
    return manifest
