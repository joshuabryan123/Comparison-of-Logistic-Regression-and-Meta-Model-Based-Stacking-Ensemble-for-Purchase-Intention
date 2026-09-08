# 🛒 Comparison of Logistic Regression and Meta-Model Based Stacking Ensemble for Purchase Intention Classification on Imbalanced Data

This repository contains the code and implementation for the study comparing traditional statistical classification methods with machine learning ensemble strategies for predicting customer purchase intention on highly imbalanced e-commerce session data.

---

## 📌 Background

In e-commerce, predicting customer purchase intention is crucial for optimizing conversion rates and targeting marketing strategies. However, consumer behavior data presents two main challenges:
1. **Complex Patterns:** Non-linear interactions between user browsing behavior, time spent, and seasonal factors.
2. **Class Imbalance:** Most visiting sessions do not result in a purchase (85.4% no purchase vs. 14.6% purchase), causing standard models to bias towards the majority class.

This project addresses these challenges by introducing a **Stacking Ensemble architecture using XGBoost as a meta-model** to integrate predictions from multiple base learners, coupled with **Class Weighting** and **Optuna Threshold Optimization** to handle severe class imbalance effectively without data artificial resampling (such as SMOTE).

---

## 📁 Dataset & Preprocessing

We used the **Online Shoppers Purchasing Intention** dataset from the UCI Machine Learning Repository, comprising **12,330 session records** with 17 features (10 numerical and 8 categorical).

### Key Processing Steps:
* **Outlier Handling:** K-Nearest Neighbors (KNN) distance-based detection ($k=5$) with PyOD. Outliers (<10%) were preserved as legitimate rare user behavior.
* **Feature Engineering:** Created derived features including `nPages_Visited`, `total_Duration`, `avg_Administrative_Duration`, `avg_Informational_Duration`, `avg_Product_Related_Duration`, `avg_Duration_per_Page`, and `is_First_Semester`.
* **Encoding & Scaling:** One-Hot Encoding for categorical features and Standard Scaling for numerical features.
* **Data Partition:** 80:20 Train-Test split using Stratified Random Sampling to maintain target class proportions.

---

## 🛠️ Methodology & Modeling Architecture

The workflow incorporates a two-stage modeling approach (Base Models → Meta-Model):
1. **Class Imbalance Mitigation:** Applied Cost-Sensitive Learning (`class_weight` / `scale_pos_weight`) across all algorithms to penalize minority class misclassifications.
2. **Cross-Validation & Tuning:** Hyperparameter optimization via **Optuna** using **Stratified 5-Fold Cross-Validation** optimized against the **Average Precision** metric.
3. **Threshold Tuning:** Grid search on predicted probability thresholds ($0.05 - 0.95$) to maximize F1-score on training folds.

---

## 📊 Results & Comparative Analysis

Evaluation was performed on the held-out test set to test real-world generalizability across key metrics:

### 1. Classification Performance

| Classification Model | Optimal Threshold | Precision | Recall (Sensitivity) | Balanced Accuracy |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression (RL)** | 0.5200 | 0.5930 | **0.7592** | 0.8318 |
| **Random Forest (RF)** | 0.7300 | 0.5915 | 0.7516 | 0.8283 |
| **LightGBM (LGBM)** | 0.6900 | 0.6651 | 0.7225 | 0.8279 |
| **XGBoost (Standalone)** | 0.6500 | **0.6699** | 0.7225 | 0.8286 |
| **Stacking Ensemble (SE)** 🚀 | 0.6185 | 0.6385 | 0.7513 | **0.8332** |

* **Top Performer:** The **Stacking Ensemble (SE)** achieved the highest **Balanced Accuracy (83.32%)**, balancing high recall with reduced false positives.
* **Threshold Behavior:** Logistic Regression required the lowest threshold (52.00%) due to smoothly distributed probabilities, whereas tree-based ensembles required higher thresholds to balance majority class skewness.

---

### 2. Feature Importance & Base Model Contribution

* **Meta-Model Contribution Breakdown:**
  * 🟢 **LightGBM (LGBM):** 52.99% contribution
  * 🔵 **Random Forest (RF):** 42.54% contribution
  * 🟡 **Logistic Regression (RL):** 4.47% contribution

* **Key Predictors:** Across all individual models, **`PageValues`** (the average value of pages visited prior to completing a transaction) emerged as the single most dominant predictor of purchase conversion, followed by seasonal indicators (**`is_Nov`**, **`is_May`**) and page engagement metrics (**`ExitRates`**, **`total_Duration`**).

---

## 💡 Conclusions & Recommendations

### 🎯 Conclusions
1. **Ensemble Superiority:** The XGBoost-based Stacking Ensemble outperforms single models by effectively combining linear patterns from Logistic Regression with complex non-linear decision boundaries from Random Forest and LightGBM.
2. **Effective Imbalance Handling:** Integrating class weights and probability threshold tuning via Optuna successfully addresses class imbalance without losing data integrity through artificial resampling.
3. **Business Drivers:** Page value and seasonal campaign periods (e.g., November shopping season) are the strongest operational indicators for predicting whether an e-commerce visitor will buy.

### 🔮 Recommendations & Future Work
* **Algorithm Diversity:** Experiment with adding alternative base algorithms such as Support Vector Machines (SVM) or Neural Networks into the stacking layer.
* **Dynamic Datasets:** Validate the stacking model on larger, multi-platform, or stream-based real-time e-commerce user tracking datasets to evaluate performance at enterprise scale.

---

## 👥 Authors

* **Joshua Bryan Wijaya**
* **Mhd Haekal Hakim** 
* **Zifa Aura Rahman**
* **Bryant Indervil Deadora**
* **Muhammad Bagas Ramadhan**
* **Cici Suhaeni**

*Department of Statistics, IPB University, Indonesia*
