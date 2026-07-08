# Research Datasets

The original datasets used during CardioSense AI development are **not included**
in this repository due to size and licensing constraints.

---

## 1. UCI Heart Disease Dataset (Multi-Site)

**Used in:** Notebooks 01–05, 09, 13

**Sub-datasets used (all four sites):**

| Sub-dataset   | Rows | Key characteristic                        |
|---------------|------|-------------------------------------------|
| Cleveland     | 303  | Most complete; primary baseline           |
| Hungarian     | 294  | Heavy missingness in `slope`, `ca`, `thal`|
| Switzerland   | 123  | `chol` mostly 0 (treated as missing)      |
| VA Long Beach | 200  | Similar missingness pattern to Switzerland|

**Combined:** 920 rows · binary dataset; 626 rows · multiclass severity dataset  
(row counts differ because rows with unusable severity labels were excluded from the multiclass split)

**License:** CC BY 4.0

**Source:**
- https://archive.ics.uci.edu/dataset/45/heart+disease

**Kaggle mirror:**
- https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci

---

## 2. PTB-XL ECG Dataset

**Used in:** Notebooks 06–12

**What was used:** 12-lead ECG waveform recordings at 100 Hz, 10 seconds per record.
Full dataset: ~21,437 records. Development used a 5K subset (Notebook 08 ablation),
then scaled to full dataset for final CNN training (Notebook 08b).

**License:** PhysioNet Credentialed Health Data License 1.5.0  
⚠️ **Note:** The official PhysioNet source requires a credentialed account and
data use agreement. The Kaggle mirror below does not require credentialing but
re-check its terms before commercial use.

**Source (official, requires credentials):**
- https://physionet.org/content/ptb-xl/1.0.3/

**Kaggle mirror (no credentials required):**
- https://www.kaggle.com/datasets/khyeh0719/ptb-xl-dataset

---

## Reproducing the datasets

The preprocessing pipeline in Notebooks 01–07 reproduces every transformation
used during model development, starting from the raw downloaded files.

Key preprocessing decisions documented in `research/PROCESS.md`:
- Missing value treatment (`?` markers + biological zero imputation for `chol`, `trestbps`)
- Patient-grouped train/val/test splits (`GroupShuffleSplit`, `random_state=42`) — prevents identity leakage
- KNN imputation fit on train-only (Notebook 02b fix — corrects data leakage from initial run)
- 11 consensus-selected features frozen in `results/metrics/04_selected_features.csv`
