import pandas as pd
import numpy as np
import re
import warnings

# Suppress specific warnings if desired
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

def load_and_preprocess_data(filepath):
    """
    Loads and preprocesses the telecom customer churn data.
    Contains all original imputation, type casting, and feature engineering.
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Data loaded successfully from {filepath}")
    except FileNotFoundError:
        print(f"Error: Data file not found at {filepath}")
        return pd.DataFrame()

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    missing_charges_mask = df['TotalCharges'].isnull()
    df['tenure'] = pd.to_numeric(df['tenure'], errors='coerce').fillna(0)
    df['MonthlyCharges'] = pd.to_numeric(df['MonthlyCharges'], errors='coerce').fillna(0)
    df.loc[missing_charges_mask, 'TotalCharges'] = df.loc[missing_charges_mask, 'tenure'] * df.loc[missing_charges_mask, 'MonthlyCharges']
    df['TotalCharges'].fillna(0, inplace=True)
    df['SeniorCitizen'] = df['SeniorCitizen'].map({0: 'No', 1: 'Yes'}).astype('category')

    for col in ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
                'PaperlessBilling', 'PaymentMethod', 'Churn']:
        if col in df.columns:
             if df[col].dtype == 'object': df[col] = df[col].astype(str)
             try: df[col] = df[col].astype('category')
             except TypeError as e: print(f"Warning: Could not convert column '{col}' to category. Error: {e}")

    if 'Churn' in df.columns:
        df['Churn_numeric'] = df['Churn'].map({'Yes': 1, 'No': 0})
    else:
        df['Churn_numeric'] = 0
        
    if 'tenure' in df.columns:
        bins_tenure = [-1, 12, 24, 36, 48, 60, 73]; labels_tenure = ['0-1 Year', '1-2 Years', '2-3 Years', '3-4 Years', '4-5 Years', '5+ Years']
        df['TenureGroup'] = pd.cut(df['tenure'], bins=bins_tenure, labels=labels_tenure, right=True)
        df['TenureGroup'] = pd.Categorical(df['TenureGroup'], categories=labels_tenure, ordered=True)
    else:
        df['TenureGroup'] = pd.NA
        
    if 'MonthlyCharges' in df.columns and df['MonthlyCharges'].notna().any():
        try: df['MonthlyChargeGroup'] = pd.qcut(df['MonthlyCharges'], q=4, labels=['Low', 'Medium', 'High', 'Very High'], duplicates='drop')
        except ValueError:
            print("Warning: Quantile binning for Monthly Charges failed, using fixed bins.")
            charge_bins = [0, 35, 70, 95, df['MonthlyCharges'].max() + 1]; charge_labels = ['Low (<$35)', 'Medium ($35-$70)', 'High ($70-$95)', 'Very High (>$95)']
            df['MonthlyChargeGroup'] = pd.cut(df['MonthlyCharges'], bins=charge_bins, labels=charge_labels, right=False, include_lowest=True)
        if not isinstance(df['MonthlyChargeGroup'].dtype, pd.CategoricalDtype): df['MonthlyChargeGroup'] = df['MonthlyChargeGroup'].astype('category')
        if not df['MonthlyChargeGroup'].cat.ordered:
             try:
                 def get_sort_key(label): label_str = str(label); numbers = [float(s) for s in re.findall(r'-?\d+\.?\d*', label_str)]; return min(numbers) if numbers else 0
                 ordered_labels = sorted(df['MonthlyChargeGroup'].dropna().unique(), key=get_sort_key)
                 df['MonthlyChargeGroup'] = df['MonthlyChargeGroup'].cat.set_categories(ordered_labels, ordered=True)
             except: print("Warning: Could not automatically order MonthlyChargeGroup labels."); df['MonthlyChargeGroup'] = df['MonthlyChargeGroup'].cat.as_ordered()
    else:
        df['MonthlyChargeGroup'] = pd.NA
        
    optional_services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    valid_optional_services = [s for s in optional_services if s in df.columns]
    if valid_optional_services: df['NumOptionalServices'] = df[valid_optional_services].apply(lambda row: sum(1 for service in row if service == 'Yes'), axis=1)
    else: df['NumOptionalServices'] = 0
    
    protective_services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport']
    valid_protective_services = [s for s in protective_services if s in df.columns]
    df['NumProtectiveServices'] = 0
    if 'InternetService' in df.columns and valid_protective_services:
        has_internet_mask = df['InternetService'] != 'No'
        df.loc[has_internet_mask, 'NumProtectiveServices'] = df.loc[has_internet_mask, valid_protective_services].apply(
            lambda row: sum(1 for service in row if service == 'Yes'), axis=1)
            
    df['RiskScore'] = 0
    if 'Contract' in df.columns: df.loc[df['Contract'] == 'Month-to-month', 'RiskScore'] += 3
    if 'InternetService' in df.columns: df.loc[df['InternetService'] == 'Fiber optic', 'RiskScore'] += 2
    if 'PaymentMethod' in df.columns: df.loc[df['PaymentMethod'] == 'Electronic check', 'RiskScore'] += 2
    if 'OnlineSecurity' in df.columns: df.loc[df['OnlineSecurity'] == 'No', 'RiskScore'] += 1
    if 'TechSupport' in df.columns: df.loc[df['TechSupport'] == 'No', 'RiskScore'] += 1
    if 'Dependents' in df.columns: df.loc[df['Dependents'] == 'No', 'RiskScore'] += 1
    if 'Partner' in df.columns: df.loc[df['Partner'] == 'No', 'RiskScore'] += 1
    if 'tenure' in df.columns:
        df.loc[df['tenure'] <= 6, 'RiskScore'] += 2
        df.loc[(df['tenure'] > 6) & (df['tenure'] <= 12), 'RiskScore'] += 1
        
    print("Data preprocessing completed.")
    display_columns = ['customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'TenureGroup', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'MonthlyChargeGroup', 'TotalCharges', 'NumOptionalServices', 'NumProtectiveServices', 'RiskScore', 'Churn']
    final_columns = [col for col in display_columns if col in df.columns]
    
    return df[final_columns]