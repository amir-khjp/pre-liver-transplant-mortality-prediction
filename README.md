# Pre–Liver Transplant Mortality Prediction (10-Fold CV | Classical ML & Ensembles)

This repository contains an end-to-end machine learning workflow for **predicting patient mortality before liver transplantation** using a clinical dataset. The pipeline focuses on:

- **Class imbalance handling** using **SMOTE**
- **Robust evaluation** with **10-fold Stratified Cross-Validation**
- Benchmarking multiple **classical ML models** (RF, XGBoost, Gradient Boosting, MLP, AdaBoost, Logistic Regression, Linear SVM, etc.)
- **Soft voting ensembles** (Top-3 / Top-4 models based on PR-AUC)
- **Feature redundancy / multicollinearity analysis**
  - Numeric: Pearson correlation (Top-30 numeric features)
  - Categorical: Cramér’s V association matrix (Top categorical features)
- **Feature importance** analysis (RandomForest aggregated importance + XGBoost gain importance)

> Status: **In Progress** (research/experimental code, results may change with further tuning and validation).

---

## Repository Structure (suggested)

## Section 1 — Balancing Data with SMOTE

This step converts the original dataset into an encoded feature matrix and balances the classes:

* Before SMOTE: `Expire=0: 633`, `Expire=1: 89`
* After SMOTE:  `Expire=0: 633`, `Expire=1: 633`

Output shapes (example):

* Unbalanced: `(722, 154)` raw features
* Balanced (encoded): `(1266, 340)` after preprocessing + SMOTE

---

## Section 2 — Model Benchmarking (10-Fold CV)

Models evaluated using `StratifiedKFold(n_splits=10)` and the following metrics:

* Accuracy
* Precision
* Recall
* F1
* PR-AUC (Average Precision)
* ROC-AUC

In addition to individual models, the repo builds:

* **Voting-Top3**: Soft-vote of top 3 models by PR-AUC
* **Voting-Top4**: Soft-vote of top 4 models by PR-AUC

Example results (on balanced dataset):

* **Voting-Top3 (RF+XGB+GB)**: PR-AUC ≈ 0.9905
* **Voting-Top4 (RF+XGB+GB+MLP)**: PR-AUC ≈ 0.9948

---

## Section 3 — Feature Importance

Two importance methods are included:

### 3A) RandomForest importance (aggregated)

* Trains RandomForest on balanced encoded data
* Uses `preprocess.get_feature_names_out()`
* Aggregates one-hot encoded features back to their **original column names**
* Saves:

  * `feature_importance_rf_aggregated.csv`
  * `rf_top_features_aggregated.png`
  * `rf_cumulative_importance_aggregated.png`

### 3B) XGBoost gain importance (aggregated)

* Extracts gain from booster features (`f0`, `f1`, ...)
* Maps gain scores to encoded feature names, then aggregates to original feature names
* Saves:

  * `feature_importance_xgb_gain_aggregated.csv`

---

## Section 4 — Numeric Redundancy (Pearson Correlation)

* Computes Pearson correlation on **Top-30 numeric** features
* Produces an **annotated heatmap image**:

  * `corr_matrix_top30_numeric_annotated.png`
* Also exports the correlation matrix as a dataframe when needed.

---

## Section 5 — Categorical Redundancy (Cramér’s V)

* Computes pairwise **Cramér’s V** (bias-corrected) among selected categorical features
* Generates an **annotated heatmap image**:

  * `cramersV_top30_categorical_annotated.png`

---

## Section 6 — Combined Redundancy Report

Creates a compact redundancy summary for both:

* Numeric pairs (Pearson r)
* Categorical pairs (Cramér’s V)

Exports:

* `section6_numeric_pairs_top30.csv`
* `section6_categorical_pairs_top.csv`
* `section6_redundancy_summary.csv`

This is useful for reporting multicollinearity/redundancy before final model selection.

---

## How to Run

1. Place `cleaned_dataset.csv` next to your notebook(s).
2. Run notebooks in order:

* `01_smote_balancing.ipynb`
* `02_cv_model_benchmark.ipynb`
* `03_feature_importance.ipynb`
* `04_numeric_correlation.ipynb`
* `05_categorical_cramersv.ipynb`
* `06_redundancy_report.ipynb`

---

## Notes / Limitations

* SMOTE is applied **after preprocessing** (scaling + one-hot encoding), which is typical but should be reported clearly.
* Performance on a **balanced (synthetic) dataset** may not reflect real-world deployment performance.
* For clinical ML, consider further steps (future work):

  * Nested CV for model selection
  * Calibration (Platt/Isotonic)
  * External validation cohort
  * SHAP for explainability
  * Leakage checks and temporal splits (if applicable)



## Contact

**Amirhossein Khajepour**
Email: [a.khajepour.official@gmail.com](mailto:a.khajepour.official@gmail.com)


