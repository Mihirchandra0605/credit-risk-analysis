# Credit Risk Analytics & Loan Default Prediction

An end-to-end credit risk analytics and machine learning project using historical LendingClub loan data to analyze borrower characteristics, predict loan default risk, and evaluate model performance through temporal validation and population stability monitoring.

The project covers exploratory data analysis, cohort construction, feature engineering, probabilistic classification, model evaluation, and model monitoring concepts commonly encountered in credit risk management.

---

## 1. Project Overview

Credit risk assessment involves estimating the likelihood that a borrower will fail to meet their financial obligations. This project develops a baseline binary classification model to estimate the probability of an adverse loan outcome using borrower and loan characteristics available at origination.

The project uses historical LendingClub Accepted Loans data spanning 2007–2018 Q4.

**Primary objectives:**

- Analyze historical loan portfolios and identify patterns associated with adverse loan outcomes.
- Construct a sufficiently mature loan cohort to minimize incomplete-outcome observations.
- Perform data quality checks, feature engineering, and leakage prevention.
- Develop a Logistic Regression model to estimate borrower default probabilities.
- Evaluate model discrimination, probability prediction quality, and classification performance.
- Perform out-of-time validation and population stability analysis.
- Explore monitoring considerations relevant to credit risk models.

**Important:** This is an educational credit risk modelling baseline using peer-to-peer lending data. It is not a production-ready banking model or a regulatory Probability of Default (PD) model.

---

## 2. Dataset

**Source:** LendingClub Accepted Loans dataset (2007–2018 Q4).

The original dataset contains approximately:

| Attribute | Value |
|---|---:|
| Total observations | 2,260,701 |
| Original features | 151 |
| Final mature cohort | 673,314 |
| Final cohort default rate | 14.81% |

### Dataset Files

The project expects the original compressed dataset at:

```text
data/raw/accepted_2007_to_2018Q4.csv.gz
```

The processed modelling cohort is saved at:

```text
data/processed/credit_risk_modeling_cohort.csv.gz
```

The raw dataset is not included in this repository because of its size and data redistribution considerations.

### Target Variable

The binary target represents an adverse loan outcome:

| Loan Status | Target |
|---|---:|
| Fully Paid | 0 |
| Charged Off | 1 |
| Default | 1, if present in the eligible cohort |
| Other unresolved statuses | Excluded |

The modelling dataset contains:

| Outcome | Observations |
|---|---:|
| Fully Paid | 573,572 |
| Charged Off | 99,742 |
| **Total** | **673,314** |

The resulting adverse-outcome rate is approximately **14.81%**.

Charged Off is used as a proxy for default. This mapping should not be interpreted as an exact equivalent of a bank's regulatory default definition.

---

## 3. Project Structure

```text
credit-risk-analytics/
│
├── data/
│   ├── raw/
│   │   └── accepted_2007_to_2018Q4.csv.gz
│   │
│   └── processed/
│       └── credit_risk_modeling_cohort.csv.gz
│
├── notebooks/
│   ├── 01_portfolio_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_pd_modelling.ipynb
│   └── 04_model_validation.ipynb
│
├── models/
│   └── logistic_regression_baseline.joblib
│
├── reports/
│
├── sql/
│
├── src/
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 4. Methodology

The project is organized into four major stages.

### Notebook 01 — Portfolio Exploratory Data Analysis

`01_portfolio_eda.ipynb`

Performed an initial investigation of the LendingClub loan portfolio.

Key activities:

- Examined dataset dimensions, data types, and missing values.
- Analyzed loan status distributions and adverse-outcome rates.
- Investigated borrower characteristics and loan attributes.
- Examined default patterns across loan grades and other portfolio segments.
- Analyzed numerical feature distributions and correlations.
- Identified data quality issues, missing values, and potential outliers.
- Identified variables that could introduce target leakage or encode existing lender risk assessments.

Key observations:

- Credit grades showed increasing adverse-outcome rates from Grade A to Grade G.
- Higher FICO scores were associated with lower observed default rates.
- Higher DTI values showed a positive association with adverse outcomes.
- Several post-origination variables were excluded to prevent information leakage.

### Notebook 02 — Cohort Construction & Feature Engineering

`02_feature_engineering.ipynb`

Constructed an outcome-mature modelling population and prepared the dataset for machine learning.

**Maturity filtering**

Because the dataset represents a historical snapshot through December 2018, loans without sufficient time to reach contractual maturity can have incomplete outcomes.

A conservative maturity filter was applied:

- 36-month loans issued on or before December 2015.
- 60-month loans issued on or before December 2013.

Joint applications were excluded from the final cohort.

The resulting eligible cohort contained **673,314 observations**.

**Modelling population**

A 36-month loan subset was selected for the baseline model to improve comparability.

This resulted in 618,345 observations.

**Data quality treatment**

- Non-positive annual income values were treated as missing.
- A DTI value of 999 was treated as a sentinel value and converted to missing.
- Extreme values were investigated rather than automatically removed.
- Unresolved loan statuses were excluded from the binary modelling target.

**Temporal splitting**

Instead of using a random train-test split, observations were divided chronologically:

| Dataset | Origination Period | Observations |
|---|---|---:|
| Training | 2007–2013 | 172,988 |
| Validation | 2014 | 162,570 |
| Testing | 2015 | 282,787 |
| **Total** | | **618,345** |

This approach simulates the practical scenario of developing a model using historical information and evaluating it on loans originated in later periods.

### Notebook 03 — Probability of Default Modelling

`03_pd_modelling.ipynb`

Developed a Logistic Regression baseline for estimating the probability of an adverse loan outcome.

**Predictor variables**

The final model uses 15 borrower and loan characteristics.

Numerical features (11):

- `loan_amnt`
- `annual_inc`
- `dti`
- `delinq_2yrs`
- `inq_last_6mths`
- `open_acc`
- `pub_rec`
- `revol_bal`
- `revol_util`
- `total_acc`
- `fico_range_low`

Categorical features (4):

- `emp_length`
- `home_ownership`
- `verification_status`
- `purpose`

The loan term was excluded because the baseline modelling population contains only 36-month loans.

**Preprocessing pipeline**

A Scikit-learn preprocessing and modelling pipeline was constructed.

Numerical variables:

1. Median imputation for missing observations.
2. StandardScaler for feature standardization.

Categorical variables:

1. Missing values replaced with an `Unknown` category.
2. One-Hot Encoding with the first category dropped.
3. Unseen categories handled using `handle_unknown="ignore"`.

The transformed feature space contains:

- 11 numerical features.
- 30 one-hot encoded categorical features.
- **41 total model input features.**

All preprocessing transformations are fitted on the training data to prevent data leakage.

**Model configuration**

```python
LogisticRegression(
    solver="lbfgs",
    penalty="l2",
    C=1.0,
    max_iter=1000,
    random_state=42
)
```

L2 regularization is used to control large coefficient values. The L-BFGS optimization algorithm minimizes the regularized binary cross-entropy objective during model fitting.

**Threshold selection**

The model estimates probabilities rather than directly producing binary classifications.

A threshold sweep was performed on the validation set, and a threshold of **0.15** was selected based on the best F1 score among the tested thresholds.

The selected threshold was kept fixed during final test evaluation.

This is an experimental threshold selection strategy, not a production lending-policy threshold.

### Notebook 04 — Model Validation & Monitoring

`04_model_validation.ipynb`

Evaluated the model on the out-of-time test population and investigated changes in feature and prediction distributions.

Key activities:

- Out-of-time model evaluation.
- Confusion matrix and classification metrics.
- Probability prediction assessment.
- Feature Population Stability Index (PSI).
- Model score PSI.
- Interpretation of potential distributional changes.
- Discussion of ongoing model monitoring.

---

## 5. Model Performance

### Validation Performance

| Metric | Validation |
|---|---:|
| ROC-AUC | 0.6494 |
| Average Precision | 0.2148 |
| KS Statistic | 0.2198 |
| Gini | 0.2988 |
| Log Loss | 0.3842 |
| Brier Score | 0.1146 |

### Out-of-Time Test Performance

The model was evaluated on 2015-originated loans, which were not used during model fitting or threshold selection.

| Metric | Test |
|---|---:|
| ROC-AUC | **0.6530** |
| Average Precision | 0.2349 |
| KS Statistic | 0.2208 |
| Gini | 0.3060 |
| Log Loss | 0.4046 |
| Brier Score | 0.1225 |

### Classification Performance

At the validation-selected threshold of 0.15:

| Metric | Test |
|---|---:|
| Accuracy | 0.6757 |
| Precision | 0.2293 |
| Recall | 0.4994 |
| F1 Score | 0.3143 |

Confusion matrix:

| | Predicted Non-Default | Predicted Default |
|---|---:|---:|
| Actual Non-Default | 170,048 | 70,650 |
| Actual Default | 21,069 | 21,020 |

Accuracy is not sufficient to assess the model because the target distribution is imbalanced. Therefore, discrimination, recall, precision, F1, and probability-based metrics are also considered.

The model demonstrates modest discriminatory performance and provides a baseline for further improvement.

---

## 6. Model Stability Analysis

Population Stability Index (PSI) was calculated by comparing feature distributions in the training population against validation and test populations.

Selected results:

| Feature | Train vs Validation | Train vs Test |
|---|---:|---:|
| Verification Status | 0.1226 | 0.1970 |
| Purpose | 0.0927 | 0.1063 |
| DTI | 0.0347 | 0.0841 |
| FICO Range Low | 0.0734 | 0.0559 |
| Revolving Utilization | 0.0248 | 0.0367 |
| Annual Income | 0.0050 | 0.0106 |

Model score distribution PSI:

| Comparison | Score PSI |
|---|---:|
| Train vs Validation | 0.0150 |
| Train vs Test | 0.0063 |

Verification status and loan purpose show comparatively larger changes in their distributions.

However, PSI is a distribution-shift indicator rather than a direct measurement of model predictive performance. It does not establish the cause of a change or automatically justify model redevelopment.

The analysis highlights the importance of monitoring:

- Feature distribution changes.
- Model score distribution changes.
- Default prevalence.
- Probability calibration.
- Missing-value rates.
- Unseen categorical values.
- Segment-level model performance.

The average predicted default probability on the test set was approximately 12.88%, compared with an observed adverse-outcome rate of 14.88%. This suggests potential aggregate underestimation that warrants further calibration analysis.

---

## 7. Key Technical Concepts Demonstrated

- Exploratory Data Analysis and portfolio segmentation.
- Missing-value handling and data quality assessment.
- Target construction and maturity filtering.
- Prevention of data leakage.
- Numerical feature standardization.
- Categorical feature encoding using One-Hot Encoding.
- Scikit-learn Pipeline and ColumnTransformer.
- Logistic Regression and probability estimation.
- Binary Cross-Entropy / Log Loss.
- L2 regularization and L-BFGS optimization.
- Temporal train-validation-test splitting.
- Classification threshold optimization.
- ROC-AUC, PR-AUC / Average Precision, KS, Gini, Brier Score and Log Loss.
- Population Stability Index.
- Out-of-time model validation and monitoring considerations.

---

## 8. Limitations and Scope

This project is a learning-oriented modelling exercise and has several important limitations.

1. **Dataset representativeness:** LendingClub is a peer-to-peer lending platform. Its borrower population and lending policies may differ significantly from those of a traditional commercial bank.

2. **Selection bias:** The analysis uses accepted loans rather than the entire applicant population. Therefore, rejected applicants are not represented.

3. **Default definition:** Charged-off loans are used as an adverse-outcome proxy. This is not necessarily identical to a regulatory default definition.

4. **Maturity filtering:** Contractual maturity cutoffs are heuristic measures intended to reduce incomplete outcomes. They are not a complete statistical treatment of censoring.

5. **Model complexity:** Logistic Regression is used as a baseline. More advanced methods, nonlinear relationships, interactions, and challenger models have not been comprehensively evaluated.

6. **Probability calibration:** Further calibration analysis and potentially recalibration would be required before interpreting the probabilities as reliable operational PD estimates.

7. **Regulatory scope:** The project does not implement the complete PD/LGD/EAD framework, CECL methodology, CCAR stress testing, or a regulatory model validation process.

The reported metrics should therefore be interpreted as experimental results on this dataset, not as evidence of suitability for actual lending decisions.

---

## 9. Installation & Execution

### Prerequisites

- Python 3.10 or later.
- Jupyter Notebook or JupyterLab.
- Git.

### Clone the Repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd credit-risk-analytics
```

### Create a Virtual Environment

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Add the Dataset

Place the original LendingClub dataset at:

```text
data/raw/accepted_2007_to_2018Q4.csv.gz
```

The processed cohort is generated by Notebook 02.

### Execute the Notebooks

Run the notebooks in the following order:

1. `01_portfolio_eda.ipynb`
2. `02_feature_engineering.ipynb`
3. `03_pd_modelling.ipynb`
4. `04_model_validation.ipynb`

The notebooks are designed to build on the outputs generated by earlier stages.

The fitted Logistic Regression pipeline can be serialized using Joblib to:

```text
models/logistic_regression_baseline.joblib
```

The model artifact is generated locally and is not required to reproduce the analysis from the notebooks.

---

## 10. Future Improvements

Potential extensions include:

- Benchmarking Logistic Regression against Random Forest, XGBoost, and Gradient Boosting.
- Hyperparameter tuning and systematic feature selection.
- More detailed probability calibration assessment and recalibration.
- Segment-level performance and fairness analysis.
- More comprehensive vintage analysis using Months on Book (MOB).
- Robust treatment of censoring and incomplete loan outcomes.
- Additional model stability and drift monitoring.
- Challenger model development and independent validation.
- SQL-based portfolio analysis and reporting.

---

## 11. Author

**Mihirchandra Loke**

M.Tech — Artificial Intelligence and Data Science  
International Institute of Information Technology, Bangalore

---

*This project is intended for educational, analytical, and portfolio demonstration purposes. It is not an actual lending decision system or a regulatory credit risk model.*