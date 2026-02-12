

# 🧠 Liver Expire Prediction — Machine Learning Pipeline

A complete end-to-end machine learning pipeline for predicting **Expire (binary outcome)** using clinical and diagnostic features.

This project includes:

* Data preprocessing
* Class imbalance handling (SMOTE)
* Model benchmarking (10+ classifiers)
* Ensemble learning (Soft Voting)
* Feature importance analysis
* Correlation & association analysis (numeric + categorical)
* High-resolution visual exports

---

## 📌 Project Overview

The objective of this project is to build a robust binary classification system to predict:

> **Expire ∈ {0,1}**

The dataset contains a mixture of:

* Clinical features
* Diagnostic variables
* Bone density metrics
* Demographic attributes
* Categorical and numeric variables

The pipeline emphasizes:

* Reproducibility
* Interpretability
* Proper validation
* Statistical feature analysis

---

# ⚙️ Pipeline Architecture

## 1️⃣ Data Preprocessing

* Remove missing target values
* Separate features and target
* Numeric preprocessing:

  * Median imputation
  * Standard scaling
* Categorical preprocessing:

  * Most frequent imputation
  * One-hot encoding

Implemented using:

```python
ColumnTransformer + Pipeline
```

---

## 2️⃣ Class Imbalance Handling

Original distribution:

* Class 0: 633
* Class 1: 89

After SMOTE:

* Class 0: 633
* Class 1: 633
* Final shape: (1266, 340)

Technique used:

```python
SMOTE(k_neighbors=5)
```

This ensures balanced learning and improved minority recall.

---

# 🤖 Model Benchmarking

### Cross-Validation Strategy

* 10-fold StratifiedKFold
* Shuffle enabled
* Multiple evaluation metrics

### Metrics Used

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* PR-AUC (primary ranking metric)

---

## 📊 Top Performing Models

| Model         | PR-AUC | F1     | Accuracy |
| ------------- | ------ | ------ | -------- |
| RandomForest  | 0.9905 | 0.9413 | 0.9416   |
| XGBoost       | 0.9901 | 0.9398 | 0.9400   |
| GradientBoost | 0.9865 | 0.9339 | 0.9345   |
| MLP           | 0.9863 | 0.9535 | 0.9511   |

---

# 🏆 Ensemble Learning

Two soft-voting ensembles were tested:

### Voting-Top3

(RandomForest + XGBoost + GradBoost)

### Voting-Top4

(RandomForest + XGBoost + GradBoost + MLP)

### Final Best Model

**Voting-Top4**

* PR-AUC: **0.9948**
* ROC-AUC: **0.9946**
* Accuracy: **0.9542**
* F1-score: **0.9545**

This model demonstrated the strongest generalization performance.

---

# 🔍 Feature Importance Analysis

Using RandomForest (800 trees):

* Aggregated encoded features back to original names
* Ranked importance
* Generated cumulative importance curve

### Key Findings

* Top 5 features explain ~30% of predictive power
* Top 25 features explain ~56%
* Importance moderately concentrated

Generated files:

* `feature_importance_rf_aggregated.csv`
* `rf_top_features_aggregated.png`
* `rf_cumulative_importance_aggregated.png`

---

# 📈 Correlation & Association Analysis

## Numeric Features

* Pearson correlation matrix
* Annotated heatmap
* Range: [-1, 1]

File:

```
corr_matrix_top30_numeric_annotated.png
```

---

## Categorical Features

* Bias-corrected Cramér’s V
* Full association matrix
* Annotated heatmap
* Range: [0, 1]

File:

```
cramersV_top30_categorical_annotated.png
```

These analyses help detect:

* Multicollinearity
* Redundant predictors
* Structural relationships

---

# 📂 Project Structure

```
├── cleaned_dataset.csv
├── notebook.ipynb
├── feature_importance_rf_aggregated.csv
├── rf_top_features_aggregated.png
├── rf_cumulative_importance_aggregated.png
├── corr_matrix_top30_numeric_annotated.png
├── cramersV_top30_categorical_annotated.png
└── README.md
```

---

# 🛠 Technologies Used

* Python 3.x
* scikit-learn
* imbalanced-learn
* XGBoost
* pandas
* NumPy
* matplotlib

---

# 📌 Key Takeaways

* Tree-based ensembles dominate performance
* Soft voting significantly improves results
* Balanced training is critical
* Feature importance is moderately concentrated
* No major multicollinearity detected among top numeric features
* Categorical associations are measurable but not extreme

---

# 🚀 Future Improvements

* SHAP-based explainability
* Permutation importance validation
* Top-K feature retraining experiment
* External validation dataset
* Hyperparameter optimization (Bayesian search)

---

# 👨‍💻 Author

Developed as a complete end-to-end ML classification pipeline for structured clinical data.




## Contact

**Amirhossein Khajepour**
Email: [a.khajepour.official@gmail.com](mailto:a.khajepour.official@gmail.com)


