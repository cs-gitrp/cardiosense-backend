# Heart Disease Data — Initial Observations

## Files used (14-column processed versions only)
- processed.cleveland.data   → 303 rows
- processed.hungarian.data   → 294 rows
- processed.switzerland.data → 123 rows
- processed.va.data          → 200 rows
Total expected: ~920 rows combined

## Column order (no headers in files, must add manually)
age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, target

## Missing value marker
"?" = missing. Must convert "?" to NaN when loading, not treat as text.

## IMPORTANT — missingness is worse than expected
- Cleveland: mostly complete. Missing only in `ca` and `thal` (a few rows).
- Hungarian: missing in `slope`, `ca`, `thal` heavily, some `fbs`.
- Switzerland: `chol` is mostly 0 (not real — treat 0 as missing, not a real reading), heavy missing in `fbs`, `slope`, `ca`, `thal`.
- VA: similar to Switzerland — `chol`=0 in many rows, `slope`/`ca`/`thal` heavily missing.

Action: when cleaning, treat chol=0 and trestbps=0 as missing too (impossible biological values), not just "?".

## Target column differences
- Cleveland/Hungarian/Switzerland/VA all use 0–4 severity scale (0 = no disease, 1-4 = increasing severity) — good, matches our plan to use multi-class, not binary.

## Next decision needed
- Decide imputation strategy per column (mean/median/mode vs. model-based) — to be decided in Step 6, not yet.

Never let the model see a guessed value without also telling it "this was guessed." That's literally why we made the _missing flag columns earlier — they're not optional extras, they're the safety net that protects your project's legitimacy.

## STATUS: Cleaning complete (as of this 25 jun date)
- Both datasets verified clean, 0 missing values, flags intact.
- multiclass_dataset.csv (626 rows) and binary_dataset.csv (920 rows) are the FINAL versions in data/processed/.
- Old uncleaned versions deleted.

## UPDATE: Fixed imputation data leakage
- Original baselines (notebook 03, first run) had KNN imputer fit on full 
  combined dataset before train/test split = data leakage.
- Created 02b_proper_preprocessing.ipynb: imputers now fit on TRAIN only, 
  applied to test via .transform(), not .fit_transform().
- Re-ran 03_baseline_models.ipynb on corrected splits.
- Results shifted modestly (Accuracy -1 to -3%), confirming leak existed 
  but wasn't catastrophic. Random Forest is now best baseline (83.70%, 
  AUC 0.925), overtaking XGBoost (83.15%, AUC 0.912).
- These are now the OFFICIAL baseline numbers for the project.

## UPDATE: Feature selection validated, clinical feature set FROZEN

Notebook 04 selected 11 features via consensus ranking (Chi-Square + LASSO 
+ Random Forest, top-10-in-2-of-3 rule, decided before seeing results).

Notebook 05 validated this selection against Notebook 03's 17-feature 
clean baseline (same train/test split, random_state=42):

| Model               | Δ Accuracy | Δ F1 Score | Result          |
|---------------------|-----------|-----------|------------------|
| Logistic Regression | +2.18%    | +2.44%    | Improved         |
| Random Forest       | +1.08%    | +1.35%    | Improved         |
| SVM                 | -1.09%    | -0.46%    | Passed (<2% rule)|
| XGBoost             | 0.00%     | -0.15%    | Passed (<2% rule)|

All 4 models passed the pre-committed success rule (F1 drop < 2%). Two 
models actually IMPROVED with fewer features — likely because dropped 
columns (trestbps, restecg, slope, chol, ca_missing, thal_missing) were 
contributing more noise than signal.

DECISION: Consensus-selected clinical feature set (11 features, stored in 
results/metrics/04_selected_features.csv) is now FROZEN as the official 
feature set for all downstream notebooks (06 onward). No notebook should 
hardcode a feature list manually — always load from 04_selected_features.csv.

Best baseline so far: Random Forest, 11 features — Accuracy 84.78%, 
F1 0.8679, Recall 90.2% (catches 92/102 actual heart disease patients, 
misses only 10). Recall is the priority metric for this project given 
the clinical cost of false negatives.

Naming convention locked: all notebooks/files use two-digit zero-padded 
numbering (01_, 02_, ... 10_, 11_) to avoid sort-order issues past notebook 9.

## UPDATE: ECG Architecture decided (Notebook 08, 5K dev subset)

Tested CNN-only vs CNN-LSTM (two variants, isolating conv-depth confound):
- CNN Baseline (3 conv blocks):     AUC 0.9162, F1 0.8508  <- WINNER
- CNN-LSTM (3 conv blocks + LSTM):  AUC 0.8893, F1 0.8374
- CNN-LSTM (2 conv blocks + LSTM):  AUC 0.8785, F1 0.8320

DECISION: CNN-only architecture locked for ECG branch. LSTM does not 
improve performance on single-beat ECG morphology classification at 
this data scale — convolutional feature extraction alone is sufficient.
This is a genuine negative result, controlled for conv-depth confound, 
and will be reported as such (not hidden).

Patient-level leakage check: PASSED (0 patient overlap across 
train/val/test, GroupShuffleSplit used throughout).

Next: scale CNN architecture to full PTB-XL (~21,430 records) on Kaggle.

## UPDATE: Notebook 09 finalized (calibration + fusion)

Calibration results (test set):
- Clinical: AUC 0.9262, Brier 0.1130, ECE 0.0653
- ECG:      AUC 0.9423, Brier 0.0943, ECE 0.0340

ECG branch is both the stronger discriminator and better-calibrated 
branch across all three metrics (AUC, Brier, ECE) — consistent story.

Confidence-adaptive fusion (linear + gamma=3 variants) built and tested 
via illustrative grid analysis + 5 case studies (NOT real paired patients 
— UCI clinical and PTB-XL ECG are separate populations, explicitly 
disclosed as a limitation).

Key qualitative finding: "Strong disagreement" case study (clinical=0.15, 
ECG=0.88) produced fused probability of 0.521 — right at the decision 
boundary. This is a clinically meaningful behavior: when branches 
disagree strongly, fusion output is appropriately uncertain rather than 
falsely confident. Suggests a real deployment should flag such cases 
for human review rather than force a binary decision.

Gamma=3 justified via sensitivity sweep (γ=1,2,3,5) — chosen as a 
moderate sharpening factor, avoiding near-binary model selection seen 
at γ=5.

Fusion module saved as reusable function: 09_confidence_adaptive_fusion.pkl