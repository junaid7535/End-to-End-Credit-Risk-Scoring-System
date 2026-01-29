import os
from pathlib import Path
from datetime import datetime

class Config:
    # Paths
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    MODELS_DIR = BASE_DIR / "models"
    REPORTS_DIR = BASE_DIR / "reports"
    LOGS_DIR = BASE_DIR / "logs"
    
    # Create directories
    for dir_path in [DATA_DIR, MODELS_DIR, REPORTS_DIR, LOGS_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Data files
    RAW_DATA_PATH = DATA_DIR / "credit_data.csv"
    PROCESSED_DATA_PATH = DATA_DIR / "processed_credit_data.csv"
    
    # Model files
    PIPELINE_PATH = MODELS_DIR / "credit_pipeline.pkl"
    FINAL_MODEL_PATH = MODELS_DIR / "credit_model.pkl"
    THRESHOLD_PATH = MODELS_DIR / "threshold.json"
    
    # ML settings
    TEST_SIZE = 0.2
    VAL_SIZE = 0.2
    RANDOM_STATE = 42
    CV_FOLDS = 5
    
    # Model parameters
    XGB_PARAMS = {
        'n_estimators': 300,
        'max_depth': 5,
        'learning_rate': 0.01,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': RANDOM_STATE,
        'n_jobs': -1
    }
    
    LGB_PARAMS = {
        'n_estimators': 300,
        'max_depth': 7,
        'learning_rate': 0.01,
        'num_leaves': 31,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': RANDOM_STATE,
        'n_jobs': -1
    }
    
    # Feature settings
    NUMERICAL_FEATURES = [
        'age', 'income', 'credit_amount', 'loan_duration',
        'credit_utilization', 'debt_to_income', 'existing_loans',
        'credit_inquiries', 'total_credit_lines', 'months_employed',
        'residence_months', 'dependents', 'housing_cost',
        'property_value', 'savings_balance', 'checking_balance'
    ]
    
    CATEGORICAL_FEATURES = [
        'employment_type', 'education', 'marital_status',
        'home_ownership', 'credit_history', 'purpose'
    ]
    
    TARGET = 'default'
