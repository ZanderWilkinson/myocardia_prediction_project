# XGBoost PCA dimension search

The notebook now uses 25 PCA components on heart XYZ coordinates from all frames, with seven demographic/clinical columns appended after PCA. PCA is fitted only on the 540 training patients; validation and test patients use that fit. The matching 25-PC cache prevents refitting PCA when tuning XGBoost. The 180-patient test split was not scored.

## Balanced-accuracy cutoff search

The search compares 10, 25, 50, 100, 200, 300, and 540 PCs. It evaluates 240 broad XGBoost configurations per dimension using the same five stratified patient-level folds. Each outer fold has an inner stopping/calibration subset; its cutoff maximizes balanced accuracy. Candidate settings are selected by mean outer-fold balanced accuracy. For the chosen fixed-tree-count model at each dimension, the cutoff is recalibrated with the same training-only inner-fold procedure used by the notebook.

| PCs | CV balanced accuracy | Validation AUC | Validation balanced accuracy | pMI detected at calibrated cutoff | Cutoff |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 0.798 | 0.840 | 0.715 | 48 / 89 | 0.520 |
| 25 | 0.828 | 0.907 | 0.839 | 75 / 89 | 0.342 |
| 50 | 0.839 | 0.891 | 0.794 | 68 / 89 | 0.476 |
| 100 | 0.854 | 0.919 | 0.833 | 69 / 89 | 0.513 |
| 200 | 0.857 | 0.905 | 0.834 | 76 / 89 | 0.412 |
| 300 | 0.850 | 0.905 | 0.806 | 79 / 89 | 0.353 |
| 540 | 0.835 | 0.885 | 0.601 | 19 / 89 | 0.591 |

25 PCs gave the best balanced accuracy on this validation comparison. 100 PCs had the highest validation AUC; 200 PCs led CV balanced accuracy. The validation set helped choose the component count, so these are model-selection results, not an unbiased final performance estimate. The final test set remains unused.

PCA projections were fitted once on all 540 training patients before inner CV. PCA is unsupervised, but fold transformations therefore share the full training feature distribution (transductive CV). A strict CV estimate would fit PCA separately inside each fold.

The notebook recalibrates the threshold from training patients on every model-cell run, so changing hyperparameters cannot leave a stale cutoff. Files:

- `balanced_cv_trials_10pc.csv` through `balanced_cv_trials_540pc.csv`: all configurations and fold metrics.
- `pca_dimension_comparison_balanced.json` and `.csv`: selected settings, thresholds, and comparison metrics.
- `pca_scores_25.npz`: notebook-compatible feature cache (ignored by Git).
- `pca_dimension_comparison_f2_archive.json` and `.csv`: previous recall-weighted F2 search; not used for the current balanced-accuracy policy.
