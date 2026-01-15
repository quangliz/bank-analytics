"""
Feature Engineering Module for Bank Churn Prediction

This module contains functions to transform raw customer data into
model-ready features with business-driven engineering.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler
import joblib


# Feature configuration
ID_COLS = ['RowNumber', 'CustomerId', 'Surname']
TARGET = 'Exited'

NUMERICAL_COLS = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
CATEGORICAL_COLS = ['Geography', 'Gender']
BINARY_COLS = ['HasCrCard', 'IsActiveMember']

NOMINAL_CATS = ['Geography', 'Gender']
ORDINAL_CATS = ['age_group', 'credit_tier', 'balance_tier', 'tenure_group']

ORDINAL_MAPPINGS = {
    'age_group': {'young': 0, 'adult': 1, 'middle': 2, 'senior': 3, 'elderly': 4},
    'credit_tier': {'poor': 0, 'fair': 1, 'good': 2, 'very_good': 3, 'excellent': 4},
    'balance_tier': {'zero': 0, 'low': 1, 'medium': 2, 'high': 3},
    'tenure_group': {'new': 0, 'developing': 1, 'established': 2, 'loyal': 3}
}


def engineer_features(
    df: pd.DataFrame,
    return_balance_median: bool = False
) -> Tuple[pd.DataFrame, float] | pd.DataFrame:
    """Create engineered features for bank churn prediction."""
    df = df.copy()
    balance_median = df['Balance'].median()
    
    # RATIO FEATURES (Financial Health Indicators)
    
    df['balance_salary_ratio'] = df['Balance'] / (df['EstimatedSalary'] + 1)
    df['products_per_tenure'] = df['NumOfProducts'] / (df['Tenure'] + 1)
    df['tenure_age_ratio'] = df['Tenure'] / (df['Age'] + 1)
    df['credit_age_ratio'] = df['CreditScore'] / (df['Age'] + 1)
    
    # COMPOSITE SCORES
    
    df['wealth_score'] = (df['Balance'] * df['CreditScore']) / 100000
    
    df['engagement_score'] = (
        df['HasCrCard'] + 
        df['IsActiveMember'] + 
        (df['NumOfProducts'] > 1).astype(int)
    )
    
    df['stability_score'] = (
        (df['CreditScore'] / 850) * 0.4 +
        (df['Tenure'] / 10) * 0.3 +
        df['IsActiveMember'] * 0.3
    )
    
    # BINNED FEATURES (Categorical from Numerical)
    
    df['age_group'] = pd.cut(
        df['Age'],
        bins=[0, 30, 40, 50, 60, 100],
        labels=['young', 'adult', 'middle', 'senior', 'elderly']
    ).astype(str)
    
    df['credit_tier'] = pd.cut(
        df['CreditScore'],
        bins=[0, 580, 670, 740, 800, 900],
        labels=['poor', 'fair', 'good', 'very_good', 'excellent']
    ).astype(str)
    
    df['balance_tier'] = pd.cut(
        df['Balance'],
        bins=[-1, 0, balance_median, balance_median * 2, float('inf')],
        labels=['zero', 'low', 'medium', 'high']
    ).astype(str)
    
    df['tenure_group'] = pd.cut(
        df['Tenure'],
        bins=[-1, 2, 5, 8, 11],
        labels=['new', 'developing', 'established', 'loyal']
    ).astype(str)
    
    # BINARY FLAGS (Risk Indicators)
    
    df['is_high_balance'] = (df['Balance'] > balance_median).astype(int)
    df['is_zero_balance'] = (df['Balance'] == 0).astype(int)
    df['is_senior'] = (df['Age'] >= 50).astype(int)
    df['is_new_customer'] = (df['Tenure'] <= 2).astype(int)
    df['is_germany'] = (df['Geography'] == 'Germany').astype(int)
    df['is_single_product'] = (df['NumOfProducts'] == 1).astype(int)
    df['is_multi_product'] = (df['NumOfProducts'] > 2).astype(int)
    df['is_low_credit'] = (df['CreditScore'] < 600).astype(int)
    
    # INTERACTION FEATURES
    
    df['senior_inactive'] = df['is_senior'] * (1 - df['IsActiveMember'])
    df['germany_inactive'] = df['is_germany'] * (1 - df['IsActiveMember'])
    df['zero_balance_inactive'] = df['is_zero_balance'] * (1 - df['IsActiveMember'])
    df['new_single_product'] = df['is_new_customer'] * df['is_single_product']
    
    return (df, balance_median) if return_balance_median else df


def encode_categoricals(
    df: pd.DataFrame, 
    nominal_cols: List[str] = NOMINAL_CATS,
    ordinal_cols: List[str] = ORDINAL_CATS,
    ordinal_mappings: Dict = ORDINAL_MAPPINGS
) -> pd.DataFrame:
    """Encode categorical variables."""
    df = df.copy()
    
    # One-Hot Encoding for nominal categories
    for col in nominal_cols:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
            df = pd.concat([df, dummies], axis=1)
            df = df.drop(columns=[col])
    
    # Ordinal Encoding for ordinal categories
    for col in ordinal_cols:
        if col in df.columns:
            df[f'{col}_encoded'] = df[col].map(ordinal_mappings[col])
            df = df.drop(columns=[col])
    
    return df


def preprocess_data(
    df: pd.DataFrame,
    drop_ids: bool = True,
) -> Tuple[pd.DataFrame, float]:
    """Full preprocessing pipeline: drop IDs, engineer features, encode."""
    df = df.copy()
    
    # Drop ID columns
    if drop_ids:
        cols_to_drop = [c for c in ID_COLS if c in df.columns]
        df = df.drop(columns=cols_to_drop)
    
    # Engineer features
    df, balance_median = engineer_features(df, return_balance_median=True)
    
    # Encode categoricals
    df = encode_categoricals(df)
    
    return df, balance_median


def prepare_for_prediction(
    customer_data: Dict,
    scaler: StandardScaler = None,
    feature_config: Dict = None,
) -> pd.DataFrame:
    """Prepare a single customer record for model prediction."""
    # Create dataframe from customer data
    df = pd.DataFrame([customer_data])
    
    # Apply preprocessing
    df, _ = preprocess_data(df, drop_ids=True)
    
    # Drop target if present
    if TARGET in df.columns:
        df = df.drop(columns=[TARGET])
    
    # Apply scaling if scaler provided
    if scaler is not None and feature_config is not None:
        cols_to_scale = feature_config.get('cols_to_scale', [])
        cols_to_scale = [c for c in cols_to_scale if c in df.columns]
        if cols_to_scale:
            df[cols_to_scale] = scaler.transform(df[cols_to_scale])
    
    return df


if __name__ == "__main__":
    print("Feature Engineering Module")
    print("=" * 40)
    print(f"ID columns: {ID_COLS}")
    print(f"Numerical columns: {NUMERICAL_COLS}")
    print(f"Categorical columns: {CATEGORICAL_COLS}")
    print(f"Target: {TARGET}")
