# Customer Personality Analysis — Unsupervised Customer Segmentation

Unsupervised machine learning project to segment customers based on their demographic profile, purchasing behaviour, and response to marketing campaigns, using a public dataset of 2,240 customers.

- **Problem:** Group customers into meaningful segments without predefined labels, using only their transactional and demographic data
- **Result:** 3 distinct segments identified (Premium, Deal Seeker, Window Shopper) with a silhouette score of 0.24 after PCA dimensionality reduction
- **Value:** Enables personalised marketing strategies per segment — avoiding wasted promotional budget and increasing campaign ROI

> [Proyecto en Español](README_ES.md)

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Business Value](#business-value)
3. [Dataset](#dataset)
4. [Data Challenges and Transformations](#data-challenges-and-transformations)
5. [Exploratory Data Analysis](#exploratory-data-analysis)
6. [Methodology](#methodology)
7. [Preprocessing Pipeline](#preprocessing-pipeline)
8. [Model Selection](#model-selection)
9. [Customer Segments](#customer-segments)
10. [Model Limitations](#model-limitations)
11. [Conclusions](#conclusions)
12. [Possible Improvements](#possible-improvements)
13. [Requirements](#requirements)

---

## Problem Statement

Understanding customers as individuals rather than as a single homogeneous group is key to effective marketing. Companies that can identify distinct customer profiles are better positioned to personalise their offers, allocate campaign budgets efficiently, and improve customer retention.

This project addresses the following question:

> **Given demographic, transactional, and campaign response data, can we identify distinct customer segments that support differentiated marketing strategies?**

This is framed as an **unsupervised clustering** problem — there are no predefined labels. The model must discover structure in the data from scratch.

---

## Business Value

Identifying customer segments has direct commercial value:

- **Personalised campaigns.** Premium customers receive loyalty programmes and exclusive access; Deal Seekers receive targeted promotions; Window Shoppers receive web-based family offers. Each segment gets the message most likely to convert.
- **Budget optimisation.** Discounts directed at Premium customers who buy at full price anyway are wasted spend. Segmentation allows promotional budgets to be allocated where they have the highest ROI.
- **Retention focus.** Knowing which segment a customer belongs to allows the business to design specific retention strategies before churn occurs.

---

## Dataset

- **Source:** [Customer Personality Analysis — Kaggle](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis)
- **Records:** 2,240 customers before cleaning
- **Features:** 29 original variables

| Column | Description |
|---|---|
| `Year_Birth` | Customer year of birth |
| `Education` | Education level |
| `Marital_Status` | Marital status |
| `Income` | Annual household income |
| `Kidhome` / `Teenhome` | Number of children / teenagers at home |
| `Dt_Customer` | Year of enrolment with the company |
| `MntWines`, `MntMeat`, ... | Spend on each product category (last 2 years) |
| `NumWebPurchases`, `NumStorePurchases`, ... | Purchases per channel |
| `AcceptedCmp1`–`AcceptedCmp5` | Whether each campaign was accepted |
| `Response` | Whether the last campaign was accepted |

---

## Data Challenges and Transformations

### 1. Null values and invalid records

- 24 rows with null `Income` removed
- 3 rows with implausible birth years removed
- 7 rows with uninterpretable `Marital_Status` values (`Absurd`, `YOLO`, `Alone`) removed

### 2. Date parsing

`Dt_Customer` was stored as a string in `%d-%m-%Y` format. Parsing without specifying the format caused day-month confusion for dates where the day was ≤ 12, resulting in 1,311 `NaT` values. Fixed by passing `format='%d-%m-%Y'` explicitly and extracting the year.

### 3. Outlier detection — Isolation Forest over IQR

The IQR method applied sequentially across columns caused cumulative data loss of nearly 50% (2,206 → 827 rows). Isolation Forest evaluates outliers globally across all features simultaneously, removing only genuinely anomalous records while preserving dataset size. With `contamination=0.1`, 221 outliers (~10%) were removed, confirmed visually by the anomaly score distribution.

### 4. Feature engineering

Six new features derived from existing variables:

| Feature | Description |
|---|---|
| `TotalSpend` | Sum of spend across all product categories |
| `TotalPurchases` | Sum of web, catalogue, and store purchases |
| `TotalChildren` | Sum of `Kidhome` and `Teenhome` |
| `TotalCampaignsAccepted` | Sum of all accepted campaigns |
| `SpendingPerPurchase` | `TotalSpend` / `TotalPurchases` |
| `WebEngagementRate` | `NumWebVisitsMonth` / `NumWebPurchases` |

**Final dataset: 1,985 rows × 32 columns**

---

## Exploratory Data Analysis

Key findings from the EDA phase:

- **Income** is strongly correlated with spend across all product categories (Pearson r > 0.6 for most categories)
- **TotalChildren** has a strong negative correlation with income and spend — families with children spend significantly less
- **Education** and **Marital_Status** show statistically significant differences in income across groups (Kruskal-Wallis test)
- **Channel preferences** vary across income levels — higher-income customers use catalogues more; lower-income customers visit the web more but purchase less
- **Campaign response** is concentrated in higher-income customers; the majority of customers across all groups rejected all campaigns

---

## Methodology

1. **Data loading and cleaning** — handle nulls, invalid dates, ambiguous categories, and outliers
2. **Exploratory data analysis** — distributions, correlations, and statistical tests
3. **Feature engineering** — derive higher-level behavioural features
4. **Preprocessing** — One-Hot Encoding, RobustScaler, PCA
5. **K selection** — Elbow method and Silhouette score across K=2 to K=10
6. **Model training** — K-Means; GMM and DBSCAN tested and discarded
7. **Evaluation** — silhouette score and UMAP visualisation
8. **Cluster profiling** — `groupby('Cluster').agg()` on original unscaled features

---

## Preprocessing Pipeline

| Step | Transformation | Reason |
|---|---|---|
| Categorical encoding | One-Hot Encoding | K-Means requires numerical input |
| Numerical scaling | `RobustScaler` (median + IQR) | Robust to remaining outliers after Isolation Forest |
| Dimensionality reduction | PCA — 15 components, 90% variance | Addresses curse of dimensionality; improves cluster separation |

All steps fitted exclusively on the full dataset (unsupervised — no train/test split required).

---

## Model Selection

Three clustering algorithms were evaluated:

| Model | Silhouette | Notes |
|---|---|---|
| K-Means K=3 (raw features) | 0.20 | Baseline |
| GMM K=6 (BIC minimum) | 0.04 | Worse than K-Means; discarded |
| DBSCAN (eps=5) | — | Produced 1 cluster + 3 noise points; discarded |
| **K-Means K=3 + PCA** | **0.24** | Best result; chosen as final model |

**K selection:** The silhouette score was nearly flat across K=2–10 even with PCA, offering no clear signal. The elbow method showed the most pronounced inflection at K=3. K=2 produced a higher average silhouette (0.37) but cluster 0 had many misassigned points (negative silhouette coefficients), making it unreliable. K=4 degraded to 0.21 with heavy cluster overlap in UMAP.

---

## Customer Segments

### Cluster 1 — Premium Customer

**Demographics:** Born ~1968 | Married | Income ~€73k | Almost no children (0.26 avg)  
**Spending:** Total spend €1,274 avg | Spend per purchase €69 | Top categories: wines and meat  
**Channels:** Catalogue preferred | Lowest deal purchases (1.42) — buys at full price  
**Campaigns:** Most responsive (0.79 campaigns accepted avg)  
**Recommendation:** VIP loyalty programme with exclusive rewards and early access to new products. Direct personalised campaigns here — this segment responds and has the budget to increase ticket size.

---

### Cluster 0 — Deal Seeker

**Demographics:** Born ~1966 | Married | Income ~€57k | ~1 child  
**Spending:** Total spend €634 avg | Spend per purchase €39  
**Channels:** In-store and web | Highest deal purchases (3.44) — price promotions drive conversion  
**Campaigns:** Moderate response (0.42 campaigns accepted avg)  
**Recommendation:** Segmented discount campaigns and limited-time promotions. This is where promotional budget has the highest ROI — discounts generate incremental purchases that would not happen at full price.

---

### Cluster 2 — Window Shopper

**Demographics:** Born ~1972 (youngest) | Married | Income ~€34k (lowest) | Most children (1.25 avg)  
**Spending:** Total spend €80 avg | Spend per purchase €13  
**Channels:** Highest web visit frequency (6.43/month) and web engagement rate (3.86) — browses but rarely converts  
**Campaigns:** Very low response (0.17 campaigns accepted avg)  
**Recommendation:** Web-based promotions: family bundles, 2-for-1 deals, entry-level offers. High web engagement signals purchase intent — the barrier is budget, not interest.

---

## Model Limitations

- **Moderate silhouette score (0.24):** Customer profiles in this dataset transition gradually — there are no sharp natural boundaries between segments. This is a data characteristic, not a modelling failure.
- **K-Means assumes spherical clusters:** The algorithm uses Euclidean distance and equal-variance cluster shapes, which may not match the actual geometry of customer data.
- **PCA reduces interpretability:** Cluster assignments are computed on principal components. Profiling must be done on the original variables post-hoc.
- **Static snapshot:** The dataset represents a single point in time. Customer behaviour evolves — segments identified today may shift.
- **No ground truth:** Without labelled data there is no way to validate whether the clusters correspond to real-world meaningful groups beyond business interpretation.
- **Isolation Forest contamination is an assumption:** Setting `contamination=0.1` was validated visually via the anomaly score distribution but is not derived from domain knowledge.

---

## Conclusions

This project identifies three customer segments with clearly different income levels, spending patterns, and responses to marketing campaigns. The segmentation is not perfect — a silhouette score of 0.24 reflects the gradual nature of customer profiles in this dataset — but the three segments are interpretable and actionable.

The most important finding is not a metric but a strategic insight: **applying the same campaign to all customers is inefficient**. Premium customers buy without discounts; offering them one wastes budget. Deal Seekers need price incentives to convert; withholding them loses revenue. Window Shoppers are not disengaged — they visit the web frequently — but need a lower-friction entry point.

The three segments carry concrete strategic implications. **Premium customers** account for the highest spend per purchase (€69 avg) and the highest campaign acceptance rate (0.79 avg), yet they buy at full price regardless — channelling discount budgets towards them yields no incremental revenue. **Deal Seekers** show the opposite pattern: the highest deal purchase frequency (3.44 avg) confirms that price incentives are the lever that drives their conversion. **Window Shoppers** visit the web 6.4 times per month on average — the highest of all segments — but spend only €80 in total. That gap between engagement and spend is the business opportunity: targeted web promotions remove the budget barrier for a segment that already shows purchase intent.

The practical implication is that a one-size-fits-all campaign is the worst allocation of marketing budget across all three segments simultaneously. This analysis provides the segmentation layer that makes differentiated strategies possible.

---

## Possible Improvements

- **Hierarchical clustering (Agglomerative):** Does not assume spherical clusters and produces a dendrogram that can help determine K more intuitively.
- **Feature selection before PCA:** Removing low-variance or highly correlated features before dimensionality reduction could improve cluster compactness.
- **Temporal analysis:** Adding customer tenure or purchase frequency over time would enable more nuanced segmentation (e.g., identifying churning Premium customers).
- **Ensemble clustering:** Combining multiple clustering runs with consensus labels produces more stable segments, reducing sensitivity to random initialisation.
- **External validation:** Linking cluster assignments to actual business outcomes (retention rate, revenue per customer, campaign conversion rate) would allow proper evaluation beyond internal metrics.
- **Better DBSCAN tuning:** A more careful grid search over eps and min_samples could yield better results than the single attempt with the k-distance graph.

---

## Requirements

```bash
pip install kagglehub pandas matplotlib seaborn scikit-learn numpy umap-learn
```

---

*Data source: https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis*
