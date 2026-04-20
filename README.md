# 📈 Telecom Customer Churn Intelligence & ML Pipeline

## 🎯 Executive Summary
This project is an end-to-end, production-ready operational intelligence dashboard designed to analyze, visualize, and predict telecom customer churn. 

Moving beyond standard static Exploratory Data Analysis (EDA), this application implements a **Modular Data Engineering Architecture** and features an integrated **Machine Learning "What-If" Engine**. It allows business stakeholders to not only uncover historical churn drivers but also simulate live scenarios to determine the probability of customer attrition in real-time.

---

## 🚀 Key Features

### 1. The AI Churn Predictor ("What-If" Simulator)
An interactive Machine Learning tab powered by a **Random Forest Classifier**. Users can adjust customer parameters (e.g., tenure, monthly charges, contract type, and tech support) via UI sliders to see the live probability of churn on a dynamic gauge chart. This turns passive data into actionable retention strategies.

### 2. Modular Separation of Concerns (SoC) Architecture
The codebase has been strictly decoupled into an industry-standard format:
* **The Data Engine (`src/data_preprocessing.py`):** Handles missing value imputation, complex type casting, and feature engineering (including custom `RiskScore` and `TenureGroup` generation) outside the UI thread.
* **The Visualization Engine (`src/visualizations.py`):** A centralized library of 27+ optimized Plotly functions, ensuring UI consistency and enabling easy reusability across Jupyter Notebooks and web apps.
* **The Controller (`app/app.py`):** A lightweight Dash application that orchestrates routing, layouts, and interactive callbacks without being bloated by data manipulation logic.

### 3. Comprehensive Analytics Suite
* **Financial Dashboards:** Analyzes the relationship between Monthly Charges, Total Charges, and Tenure using advanced visual techniques like Ridgeline Density Plots, 3D Scatter Plots, and Violin plots.
* **Service Impact Deep-Dives:** Faceted segment analysis to isolate high-risk customer profiles (e.g., Month-to-Month Fiber Optic users without Tech Support).
* **Interactive Data Explorer:** A front-end searchable datatable for granular customer auditing.

---

## 📂 Project Architecture

```text
Customer-Churn-EDA-Analysis/
├── .gitignore                 <- Prevents data leakage and environment commits
├── requirements.txt           <- Pinned dependencies for guaranteed reproducibility
├── README.md                  <- Project documentation
├── data/                      
│   └── raw/                   <- Ignored in Git; houses CUSTOMER_ANALYTICS_Telecom_churn.csv
├── notebooks/                 
│   ├── Customer_Churn_1.ipynb <- Initial flat EDA and data profiling
│   └── Customer_Churn_2.ipynb <- Plotly visualization prototyping
├── src/                       <- Core Logic Module
│   ├── __init__.py            
│   ├── data_preprocessing.py  <- Pandas ETL pipeline and Feature Engineering
│   ├── ml_engine.py           <- Scikit-Learn Random Forest training and inference
│   └── visualizations.py      <- Plotly graph generation factory
└── app/                       <- Dashboard Application
    └── app.py                 <- Multi-page Dash application routing and UI
