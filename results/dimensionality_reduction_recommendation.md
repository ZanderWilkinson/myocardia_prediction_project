# Dimensionality reduction for pMI prediction

## Recommendation

Use **PCA on the coordinate sequences**, followed by a regularized classifier.
Append the seven demographic/clinical variables after coordinate reduction.
Keep a model based on simple shape summaries as a serious comparison: in this
exploratory experiment it performed essentially as well as PCA when demographics
were included. These results support PCA as a practical starting method, not a
claim that it is the globally optimal representation.

## What the local data supports

- The training collection has 900 patients: 452 healthy and 448 pMI.
- All 720 development patients checked have finite arrays of shape
  `(10, 18000, 4)`, with 6,000 points for each of three anatomical labels.
- The entire label sequence is identical across these patients and all frames.
  The blocks occur in the order **2, 0, 1**. Each fourth-channel feature is
  therefore constant across patients. It carries no pMI discrimination and
  becomes zero after the current `StandardScaler` transformation.
- Geometry checks used 24 development patients, comparing one reference with
  the other 23 separately for each structure (69 comparisons). After centering
  each structure and normalizing its overall size for this diagnostic only,
  the median coordinate correlation was **0.9962 between patients** and
  **0.9985 between frames 0 and 3**.
- Across patients, a point retained a median **75.5% of its ten nearest
  neighbors by point index**, compared with approximately **0.17% by chance**.
  Median same-index normalized RMS difference was 0.087, versus 1.414 after
  independently shuffling points within structures. Across frames, neighbor
  retention was 73.2%.

This is strong empirical evidence of shared spatial ordering, consistent with
common-template sampling or correspondence. The earlier warning about arbitrary
point ordering remains generally correct, but the observed data does not look
arbitrarily ordered. The repository contains no mesh-generation or registration
provenance, so these checks do **not** establish exact anatomical or material-point
identity. Frame-phase alignment is also not documented by these checks.

## Exploratory prediction comparison

The existing split logic was reproduced to reserve 180 internal test patients.
The remaining 720 patients were evaluated in five stratified folds. All fitted
preprocessing, PCA, PLS, and classifiers were trained separately inside each
fold. The external 100-patient test collection was not used for fitting or scoring.

For speed, coordinate methods used **200 fixed, randomly selected indices per
structure**, consistently across patients and all ten frames: 18,000 coordinate
features per patient. Shape summaries used all points and comprised centroids,
axis extents, and covariance eigenvalues for each structure and frame (270
features). No per-patient spatial registration or size normalization was applied
to prediction features. All classifiers were logistic regression with fixed
`C=1`, with input features standardized inside the fold.

| Representation | Mean validation ROC-AUC | Fold SD |
| --- | ---: | ---: |
| Demographics only | 0.778 | 0.033 |
| 20 centered-coordinate PCs only | 0.783 | 0.019 |
| 100 centered-coordinate PCs only | 0.835 | 0.030 |
| Shape summaries only | 0.846 | 0.033 |
| 10 PLS components + demographics | 0.870 | 0.021 |
| 100 standardized-coordinate PCs + demographics | 0.879 | 0.035 |
| Shape summaries + demographics | 0.882 | 0.018 |
| 100 centered-coordinate PCs + demographics | 0.882 | 0.035 |

PCA plus demographics and shape summaries plus demographics are effectively
tied at this resolution. The fold SD describes variation among folds, not a
confidence interval. PLS is a useful supervised alternative, but the tested
2-, 5-, and 10-component settings did not improve on the strongest PCA setting.
Other methods and extensively tuned classifiers were not evaluated.

These are exploratory model-selection scores, not final test scores. Selecting
the best setting on these folds introduces selection optimism. Point subsampling,
the fixed classifier settings, and the limited component grid further limit the
comparison. Confirm the chosen pipeline with nested validation or a locked
holdout after model selection; the largest tested PCA dimension, 100, is not a
demonstrated optimum.

## Implications for the current implementation

1. **Extract XYZ only:** use `patient.data[..., :3].reshape(-1)`, retaining
   all ten frames as one patient's feature vector. Do not reorder or resample
   points independently per patient. This gives 540,000 features at full resolution.
2. **Apply PCA only to geometry.** The current `src/mcpp/pca.py` concatenates
   seven demographic/clinical features with all 720,000 array entries before
   PCA. The 180,000 constant label entries contribute nothing after scaling;
   the geometry block overwhelmingly outweighs the seven demographic variables
   when PCA chooses its variance-maximizing directions.
3. **Append demographics to the retained PCA scores**, then standardize the
   combined classifier features and fit regularized logistic regression.
4. **Start with coordinate centering rather than automatically standardizing
   every coordinate feature.** All XYZ values have a common unit, and centered
   PCA was competitive here. The classifier's later feature scaling is a
   separate step. Evaluate spatial registration if pose varies; preserve size
   and contraction information unless validation supports removing it. Do not
   independently normalize every frame's size by default.
5. **Choose component count for pMI prediction, not just reconstruction.**
   Centered PCA retained an average 90.1% of coordinate variance with 20 PCs,
   97.6% with 50, and 99.7% with 100. Their geometry-only AUCs were 0.783,
   0.773, and 0.835 respectively: high variance retention does not establish
   that the useful class signal is retained. Tune component counts such as
   20, 50, 100, 150, and 200 together with classifier regularization.
6. **Fit every learned step inside patient-level training folds.** The ten
   frames are repeated observations of one patient, not ten independent
   training cases. Evaluate final performance once on the reserved patients.

A point-cloud neural encoder would become more relevant if future data lack
consistent ordering, or if simpler models reach a demonstrated ceiling. With
900 independent patients and evidence supporting consistent ordering, the
current results do not justify making a large learned encoder the first choice.

## Reproducibility and sources

- `results/dimensionality_reduction_audit.py`: executable audit and comparison.
- `results/dimensionality_reduction_audit.json`: diagnostic measurements,
  variance spectra, and per-fold scores.
- `results/dimensionality_reduction_comparison.csv`: complete model comparison.

Method references:

- [scikit-learn PCA documentation](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
- [Selecting dimensionality reduction with cross-validation](https://scikit-learn.org/stable/auto_examples/compose/plot_compare_reduction.html)
- [Cross decomposition and supervised PLS](https://scikit-learn.org/stable/modules/cross_decomposition.html)
- [Preventing preprocessing leakage](https://scikit-learn.org/stable/common_pitfalls.html)
- [PointNet: permutation-invariant point-cloud networks](https://arxiv.org/abs/1612.00593)
