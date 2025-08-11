# Company Valuation Prediction
Predicting startup post-money valuations using venture funding data with 30,000+ records, 60+ features, and high missingness.
This project leverages XGBoost, Random Forest, and Lasso regression to extract insights and provide predictive modeling for investment decision-making.

## 📌 Project Overview
Venture funding data is often messy, high-dimensional, and incomplete, making valuation prediction a challenging problem.This project builds a data preprocessing and modeling pipeline to clean, encode, and analyze deal and company attributes, and train models to predict PostValuation.

The goal is to:
- Explore the relationship between funding rounds, ownership structures, and valuations
- Handle missingness with domain-informed imputation strategies
- Compare tree-based and linear models for predictive accuracy and interpretability
- Provide actionable insights for stakeholders such as investors and analysts

## 📊 Dataset
Size: ~30,000 rows × 60+ columns

Example Features:
- Deal Attributes: DealSize, PreMoneyValuation, VC Round, Deal Type, Deal Class
- Company Info: Business Status, Ownership Status, Financing Type
- Unstructured Attributes: Universe (multi-label indicators of M&A, public listing, debt financing, etc.)
- Timestamps: Funding dates, last updated timestamps
Target Variable:
- PostValuation – Post-money valuation of the company

⚙️ Methodology
1. Data Cleaning & Preprocessing
Missingness Analysis:
- Assessed patterns of missing values using chi-square tests between predictors
- Applied conditional imputation (e.g., imputing DebtRaisedInRound with 0 if Debts is missing)

Feature Engineering:
- Encoded VC rounds as ordinal values (Angel = 0, 1st Round = 1, etc.)
- One-hot encoded categorical and unstructured fields using MultiLabelBinarizer
- Generated temporal features using past funding data

2. Exploratory Data Analysis
- Conducted t-tests, ANOVA, and Kruskal-Wallis tests to assess valuation differences across groups and visualized trends to uncover patterns between different features

3. Model Performance

| Model             | R² Score | RMSE      | Notes                                                         |
|-------------------|----------|-----------|---------------------------------------------------------------|
| XGBoost           | 0.982    | 1.25e6    | Highest predictive accuracy, less interpretable               |
| Random Forest     | 0.975    | 1.32e6    | Handles missing data well, interpretable via feature importance|
| Linear Regression | 0.940    | 1.80e6    | Most interpretable, but limited by linearity assumption       |


