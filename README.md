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
