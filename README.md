# Customer Personality Analysis — Unsupervised Customer Segmentation

## Overview

This project applies unsupervised machine learning to segment customers based on their demographic profile, purchasing behaviour, and response to marketing campaigns. The goal is not only to identify technically valid clusters, but to translate each segment into actionable business profiles that a company can use to personalise offers, allocate campaign budgets, and improve customer retention.

**Dataset:** [Customer Personality Analysis — Kaggle](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis)  
**Records:** 2,240 customers | **Features:** 29 variables

---

## Objective

Segment customers into meaningful groups using clustering techniques, then profile each segment with:
- Descriptive name and demographic characteristics
- Purchasing behaviour and preferred channel
- Response to discounts and marketing campaigns
- Concrete business recommendation

---

## Methodology

| Step | Description |
|------|-------------|
| Data loading | Downloaded via `kagglehub`; tab-separated CSV |
| Data cleaning | Handle nulls, invalid dates, ambiguous categories, constant columns, and outliers |
| EDA | Distributions, correlations, and statistical tests (ANOVA, Chi-square, etc.) |
| Feature engineering | New features derived from existing variables |
| Preprocessing | Encoding, scaling with `RobustScaler` |
| Modelling | K-Means clustering; optimal K selected via Elbow method and Silhouette score |
| Evaluation | Internal metrics: silhouette score and inertia; visualisation with UMAP or PCA |
| Interpretability | Cluster profiling and business recommendations |

---

## Project Structure

```
├── Customer_personality_analysis.ipynb   # Main notebook
└── README.md
```

---

## Sessions

### Session 1 — Data Loading, Cleaning & Descriptive Statistics ✅
- Loaded dataset with correct tab separator
- Removed 24 rows with null `Income` values
- No duplicates found
- Dropped constant columns `Z_CostContact` and `Z_Revenue` (zero variance)
- Dropped identifier column `ID`
- Converted `Dt_Customer` to year (integer) with correct `%d-%m-%Y` format; removed 3 implausible records
- Cleaned `Marital_Status`: removed `Absurd`, `YOLO`, and `Alone` (uninterpretable or ambiguous values, 7 rows total)
- **Outlier removal with Isolation Forest** (`contamination=0.1`): removed 221 outliers (~10%), chosen over IQR because IQR applied column-by-column caused excessive cumulative data loss. The anomaly score distribution confirmed a clear separation between normal instances and outliers.
- Final dataset: **1,985 rows × 26 columns**

### Session 2 — EDA, Statistical Testing & Feature Engineering 🔲
- Exploratory visualisations
- Pearson and Spearman correlations
- ANOVA / Kruskal-Wallis, Chi-square tests
- Feature engineering

### Session 3 — Preprocessing & Clustering 🔲
- Encoding and scaling (`RobustScaler`)
- Optimal K selection (Elbow + Silhouette)
- K-Means training
- Dimensionality reduction (UMAP or PCA)

### Session 4 — Interpretation, Conclusions & GitHub 🔲
- Cluster profiling
- Business recommendations
- Model limitations and possible improvements
- LinkedIn post

---

## Key Technical Decisions

**Isolation Forest over IQR for outlier detection:** The IQR method applied sequentially across columns caused cumulative data loss of nearly 50%. Isolation Forest evaluates outliers globally across all features simultaneously, removing only genuinely anomalous records while preserving dataset size.

**RobustScaler for preprocessing:** Since some outliers remain in the data after filtering, `RobustScaler` (which uses median and IQR instead of mean and min/max) will be used to minimise their influence on the clustering model.

---

## Requirements

```bash
pip install kagglehub pandas matplotlib seaborn scikit-learn numpy
```
