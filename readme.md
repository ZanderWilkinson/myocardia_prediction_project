# MCPP — Myocardia Prediction Project

Place Python code in the `src/mcpp/` package.

## Setup

Use Python 3.12 or newer. All Python runtime and notebook dependencies, including
PyTorch and the frame cleaner's thread-pool control, are declared in
`pyproject.toml`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e .
python3 -m ipykernel install --user --name mcpp --display-name "Python (MCPP)"
```

In VS Code, install the Python and Jupyter extensions, open a notebook,
and select **Python (MCPP)** from the kernel picker. This kernel uses the
repository's `.venv` environment. Restart the notebook kernel after installing
the package so it picks up the editable installation.

Import the package with `import mcpp`.

Store datasets in `data/`; this directory is ignored by Git.

include data files in /data/ as
labeled_test_data
test_samples
training_data

## XGBoost experiments

Open `notebooks/xgboosting.ipynb` with the Python (MCPP) kernel. Run the data
and PCA preparation cells once, then edit `XGB_PARAMS` and rerun the training
cells. Prepared features are cached locally and verified against their source
data and patient splits, so they can be reused after a kernel restart. Set
`REFIT_PCA=True` to rebuild them; changing the PCA settings also triggers a refit.

`pca_datasubset(..., motion_only=True)` uses all frames' XYZ coordinates without
demographics or anatomical labels. It does not compute displacements. The
default behavior of existing calls is unchanged. The new notebook optionally
adds demographics after PCA and keeps final test evaluation disabled while tuning.

XGBoost is included in the project dependencies. On macOS, install its OpenMP
runtime with `brew install libomp` if importing XGBoost reports a missing library.
The patient-level PCA score cache is ignored by Git.

## Attention VAE with an MI classification head

Open `notebooks/vae.ipynb` with the Python (MCPP) kernel, or run:

```bash
python -m pip install -e .
python results/train_vae.py
```

The default command trains **one model for one epoch**, preserving the original
stratified **60:20:20 split before frame-quality filtering** (seed 42). It starts
with 540 training, 180 validation and 180 final-test patients. Excluding 19
training and 6 validation cycles leaves **521 training, 174 validation and 180
held-out test patients**. Exclusions never move patients between those splits.

For the NVIDIA GPU machine and a longer single-fold experiment:

```bash
python results/train_vae.py --experiment mi-vae-single --folds 1 --epochs 100000 --device cuda
```

Use a new name when changing an existing two-fold experiment into a single-fold
split. Its old training and validation memberships cannot be safely reused.
`--epochs` adds the requested epochs per model to a completed experiment.
Early stopping is **disabled by default**, including when resuming an older
checkpoint that previously stopped early. Enable it explicitly with
`--early-stopping --patience 10`. The logs print the saved epoch, target epoch,
early-stopping setting and actual reason for stopping.

Validation **pMI F1 and accuracy**, macro F1, balanced accuracy and AUC are
reported on validation patients only. This holdout selects the best model, so
its scores are validation scores, not independent test or out-of-fold scores.
For cross-validation, set `--folds 2` or higher: the original nested holdout and
pooled out-of-fold reporting remain available.

The default experiment is named `default` and lives at
`results/vae_training/default/`. Matching names resume automatically:

```bash
python results/train_vae.py --experiment cardiac-vae --epochs 1
# Continue the same experiment for one more epoch per fold:
python results/train_vae.py --experiment cardiac-vae --epochs 1
# Continue for another 100 epochs per fold:
python results/train_vae.py --experiment cardiac-vae --epochs 100
```

Each fold keeps one rolling `fold_N.last.pt` training checkpoint. It replaces
the previous checkpoint **every 100 epochs**, plus at the end of an invocation
or early stopping, so even one-epoch experiments can resume. Use
`--checkpoint-every` to change the interval. Previous periodic copies are not
retained. `fold_N.pt` holds the lowest-validation-BCE model for inference and
continues to supply the final classification report.

A separate **`val_best.pt`** saves the highest **validation macro F1** model in
single-fold mode; CV keeps `fold_N.val_best.pt` per fold. It is replaced
atomically **immediately whenever macro F1 strictly improves**, independent of
the 100-epoch resume cadence. Ties keep the earlier model. Its metadata records
the epoch, macro F1, accuracy and pMI F1; preprocessing and train-fitted scaling
are included, so it works with `load_checkpoint` and `predict_patients`.
The best score persists across resumed runs, including when this file is newer
than the last rolling checkpoint. On resuming an older experiment without a
macro-F1 checkpoint, selection starts with its retained BCE-best weights and
then compares new epochs; earlier unsaved weights cannot be recovered.

**`training.log`** appends the terminal output for every invocation, including
epoch metrics, summaries, warnings and failures. Output is flushed as it is
written, so completed epoch lines survive interruption between checkpoints.
This also applies to notebook/API training with an output directory.

Resume restores the latest weights, AdamW state, mixed-precision gradient scaler,
train-fitted input scaling, RNG and shuffle states, epoch history, KL warmup
position, best model and early-stopping patience. A resumed fold continues from
the latest optimization step, while scoring uses its best validation model.
Checkpoints are replaced atomically; failed writes leave the previous copy
intact. Abrupt interruption can lose progress since the last checkpoint (up to
99 epochs with the default cadence).

If an invocation is interrupted, repeat the same command: its original per-fold
targets are finished, and folds already at their target are not retrained.
Increasing `--epochs` while it is unfinished extends that invocation's targets;
it does not silently ignore the new value. After completion, the next call adds
more epochs. Early-stopping history is restored, but only stops training when
`--early-stopping` is enabled. Data, patient membership, split settings,
model architecture, batch size and optimization settings must match; use a new
experiment name for different settings. Device, precision, CPU thread count,
early-stopping controls and checkpoint cadence may change, but bitwise reproducibility across different
hardware is not guaranteed. Concurrent writers to one experiment are blocked.
Older inference-only output directories cannot restore optimizer state and are
preserved; start a new named experiment instead.

`experiment.json` records experiment identity, cumulative epoch counts and the
current invocation. `mi_classification_report.json`
contains the overall scores, per-class precision/recall/F1, and confusion matrix;
`mi_predictions.csv` contains each scored patient's true MI diagnosis, predicted
diagnosis, pMI probability, and scoring fold. In single-fold mode it contains
validation patients only; `holdout_results.json` records both memberships and
`validation_predictions.npz` saves their probabilities and latents. CV mode
uses `cv_results.json` and `oof_predictions.npz` instead.

The default model has **9,641,225 trainable parameters**, an **8-dimensional
Gaussian latent space**, eight attention heads, and width 384.
Changing an existing 20-dimensional experiment to 8 requires a new experiment
name. Use `--latent-dim 20` to continue its original architecture. Each cycle starts
as `(10, 18000, 4)` with XYZ and categorical anatomical region IDs. We retain
128 points per region (384 per frame), using the same sampled indices across
all frames. Sixteen learned spatial queries attend to the points in each frame;
two encoder blocks each apply spatial attention followed by temporal attention
across the ten frames. The latent-only decoder uses one such block and
reconstructs `(10, 384, 3)` XYZ coordinates. It reconstructs the sampled cloud,
not all 18,000 original points. Attention uses PyTorch's
[TransformerEncoderLayer](https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html)
and [MultiheadAttention](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html).

**Demographics enter only the classifier:** age, BMI, height, weight, diastolic
BP, systolic BP and sex are standardized using the current training fold, then
concatenated with the latent mean and passed through 128- and 64-unit hidden
layers to predict healthy=0 versus pMI=1. Demographics never enter the encoder
or decoder. Anatomical region IDs 0/1/2 are embedded categorical inputs;
they are distinct from the patient MI label. Predictions use the latent mean
for deterministic inference. Training samples the latent for reconstruction.

Before sampling or normalization, `repair_frames` checks the full-resolution
cycle using RMS radius, 99th-percentile radius, maximum radius, bounding extent
and centroid displacement. Expansion must exceed both a conservative six-MAD
threshold and a 1.75-fold within-patient size threshold. A separate pointwise
temporal check detects scattered points even when global heart size stays nearly
normal: interpolation residuals must exceed six MADs, three times their median,
and 15% of typical radius, and neighbouring frames must agree with one another.
Nonfinite XYZ frames are also flagged. An isolated flagged frame becomes the average of its valid
neighbours; consecutive flags are linearly interpolated between the nearest
valid frames. Endpoints copy their nearest valid frame by default;
`PreprocessConfig(cyclic=True)` interpolates across the cycle boundary instead.
Anatomical IDs remain unchanged. All repairs and selected point indices are
recorded in `frame_quality.json`; source files are never overwritten. These
heuristics target gross artifacts and should be checked against visual examples
before changing thresholds. A spatial coherence check also measures the median
distance between consecutive same-region template points relative to median
radius. Per-patient K-means (`k=2`) clusters the log scores; the rougher cluster
is marked defective only if its centre is at least three times the smoother
cluster's centre. This catches consecutive defects and cycles whose median
frame is noisy, without forcing normal cycles into a defect category. The
audit stores cluster membership, centres and their separation ratio.
This assumes the original ordered surface points; shuffling point order destroys
this measure. Cycles with a majority of suspect frames raise an auditable error
in `repair_frames`. The training loader excludes these cycles, prints the count,
and saves their patient IDs, affected frames and exclusion reasons in the quality
audit. Set `load_cohort(..., exclude_unrepairable=False)` for strict failure.
Cycles with no clean reference at all may still need manual inspection. Run
`python results/audit_vae_frames.py` to reproduce the development-set repair audit.

The current audit identifies **303 suspect frames in 62 of 720 development
patients**: 37 patients receive 109 frame replacements, while 25 patients with
194 suspect frames are excluded because most of their cycle is corrupted.
These counts come from the measured spatial discontinuity, without enforcing
a target defect prevalence. Held-out test point arrays are not opened.

We assume point indices track the same point across frames and ordered templates
are comparable across patients. Frame region consistency is checked explicitly.
The ordered XYZ reconstruction loss depends on this assumption; for unordered
clouds, a set reconstruction objective would be needed. One centre is removed
for the entire patient cycle, preserving inter-frame motion. A shared scalar
XYZ scale is learned on each fold's fitting patients; per-frame normalization
would erase contraction and is deliberately avoided.

Training minimizes mean XYZ MSE + `beta * mean latent KL` + binary classification
loss. `beta` warms to 0.01 over ten epochs; AdamW, gradient clipping, training-only
class weights and best-model selection on validation classification loss are
enabled. Defaults are one model, one epoch, batch size 8, learning rate 0.0003,
and no early stopping. Optional early stopping defaults to patience 10.
GPU (CUDA), Apple GPU (MPS), or CPU is selected automatically when available.
Use `--device mps` to request the Mac GPU explicitly. MPS requires GPU access;
restricted processes can report it unavailable even on a supported Mac.
On an NVIDIA machine, install a CUDA-enabled PyTorch build using the
[official installation selector](https://pytorch.org/get-started/locally/), then run:

```bash
python results/train_vae.py --device cuda
# Select another visible GPU, or disable mixed precision:
python results/train_vae.py --device cuda:1
python results/train_vae.py --device cuda --no-amp
```

CUDA training enables FP16 automatic mixed precision, gradient scaling, pinned
CPU memory and nonblocking transfers. The VAE loss remains FP32 to protect its
KL and squared-error calculations; clipping is applied after gradient unscaling.
See [PyTorch AMP](https://docs.pytorch.org/docs/stable/amp.html).
Device selection prefers CUDA, then MPS, then CPU; explicit unavailable GPU
requests fail before data preparation. The selected GPU name and precision are
printed and saved. CUDA checkpoints can also be loaded on CPU or MPS. CUDA
hardware tests run automatically when an NVIDIA GPU is present; this Mac cannot
execute those tests. MPS and CPU use FP32 even when `amp=True`.
Each epoch prints training and inner-validation accuracy, pMI F1 and macro F1
at a fixed probability cutoff of 0.5, and saves them in the fold's `history`.
Scores pool all patients in each split instead of averaging minibatch scores.
Training scores come from the training pass; validation scores use the model
at the end of the epoch with dropout disabled.

The runner excludes the same 20% patient test split used by `DataSet` (seed 42).
With `--folds 2` or more, stratified outer folds use the remaining usable
patients; each fold has its own inner
15% holdout for early stopping. Model weights, optimizer, geometry scaling and
demographic scaling are fitted afresh per fold. Outer fold patients are used
only for scoring. The runner never loads final test point clouds. Using these
CV scores repeatedly to select architectures is model selection; retain the
held-out test for the eventual final evaluation.

For a short pipeline check using the full-size model:

```bash
python results/train_vae.py --folds 2 --limit-patients 40 --batch-size 4 --output-dir results/vae_smoke_local
python -m unittest discover -s tests -p 'test_vae.py'
python -m unittest discover -s tests -p 'test_pca_xgboost.py'
```

Run these test suites in separate processes on this Mac: loading and exercising
both XGBoost and PyTorch in one test process stalled during verification. Both
suites pass independently; the VAE workflow does not import XGBoost.

Checkpoints, per-epoch histories, fold membership, patient probabilities, latent
means and CV metrics are saved inside the experiment directory (ignored by Git
under `results/vae_training/`). `--output-dir` can specify an exact directory;
matching experiment state resumes there, and incompatible contents are protected.
Latent coordinates from independently trained folds are not aligned and should
not be pooled as one common embedding. `predict_patients` restores both model
and preprocessing from a checkpoint:

```python
from mcpp.vae_training import predict_patients

# demographics is numeric (patients, 7), in the column order listed above;
# encode sex as 0/1 using the source dataset's False/True convention.
probabilities, latent_means, repair_audit = predict_patients(
    "results/vae_training/cardiac-vae/fold_1.pt", raw_patient_arrays, demographics,
)
```
