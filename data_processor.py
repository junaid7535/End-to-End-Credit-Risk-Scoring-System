import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib
import json
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class DataGenerator:
    """Generate synthetic credit data for demonstration"""
    
    @staticmethod
    def generate_data(n_samples=10000, seed=42):
        np.random.seed(seed)
        
        data = {
            # Customer demographics
            'age': np.random.randint(18, 70, n_samples),
            'income': np.random.exponential(50000, n_samples) + 20000,
            'education': np.random.choice(['high_school', 'bachelors', 'masters', 'phd'], 
                                        n_samples, p=[0.3, 0.4, 0.2, 0.1]),
            'employment_type': np.random.choice(['employed', 'self_employed', 'unemployed', 'retired'], 
                                              n_samples, p=[0.6, 0.2, 0.15, 0.05]),
            'months_employed': np.random.exponential(60, n_samples),
            
            # Credit history
            'credit_score': np.random.normal(650, 100, n_samples),
            'credit_history': np.random.choice(['good', 'fair', 'poor', 'none'], 
                                             n_samples, p=[0.5, 0.3, 0.15, 0.05]),
            'existing_loans': np.random.poisson(2, n_samples),
            'credit_inquiries': np.random.poisson(1.5, n_samples),
            'total_credit_lines': np.random.poisson(5, n_samples),
            'credit_utilization': np.random.beta(2, 5, n_samples),
            
            # Loan details
            'credit_amount': np.random.exponential(20000, n_samples),
            'loan_duration': np.random.choice([12, 24, 36, 48, 60], n_samples),
            'purpose': np.random.choice(['car', 'home', 'education', 'business', 'debt_consolidation'], 
                                      n_samples, p=[0.2, 0.3, 0.1, 0.15, 0.25]),
            
            # Financial situation
            'debt_to_income': np.random.beta(2, 3, n_samples) * 100,
            'savings_balance': np.random.exponential(10000, n_samples),
            'checking_balance': np.random.exponential(5000, n_samples),
            'home_ownership': np.random.choice(['own', 'mortgage', 'rent', 'other'], 
                                             n_samples, p=[0.3, 0.4, 0.25, 0.05]),
            'property_value': np.random.exponential(300000, n_samples),
            
            # Personal details
            'marital_status': np.random.choice(['single', 'married', 'divorced', 'widowed'], 
                                             n_samples, p=[0.4, 0.4, 0.15, 0.05]),
            'dependents': np.random.poisson(1, n_samples),
            'residence_months': np.random.exponential(48, n_samples),
            'housing_cost': np.random.exponential(1500, n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Create target variable with business logic
        default_prob = (
            (df['credit_utilization'] > 0.8) * 0.3 +
            (df['debt_to_income'] > 50) * 0.25 +
            (df['income'] < 30000) * 0.2 +
            (df['credit_score'] < 600) * 0.15 +
            (df['months_employed'] < 12) * 0.1 +
            np.random.normal(0, 0.1, n_samples)
        )
        
        df['default'] = (default_prob > 0.5).astype(int)
        
        # Add some missing values
        for col in ['credit_utilization', 'debt_to_income', 'savings_balance']:
            mask = np.random.random(n_samples) < 0.05
            df.loc[mask, col] = np.nan
        
        return df

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Custom feature engineering transformer"""
    
    def __init__(self):
        self.features_to_create = [
            'income_to_loan_ratio',
            'credit_age_ratio',
            'utilization_times_score',
            'employment_stability',
            'financial_stability_score',
            'credit_burden'
        ]
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # 1. Income and loan related features
        X['income_to_loan_ratio'] = X['income'] / (X['credit_amount'] + 1)
        X['monthly_payment'] = X['credit_amount'] / X['loan_duration']
        X['payment_to_income'] = X['monthly_payment'] / (X['income'] / 12 + 1)
        
        # 2. Credit history features
        X['credit_age_ratio'] = X['credit_score'] / (X['age'] + 1)
        X['utilization_times_score'] = X['credit_utilization'] * X['credit_score']
        X['credit_behavior'] = X['existing_loans'] / (X['months_employed'] / 12 + 1)
        
        # 3. Employment stability
        X['employment_stability'] = X['months_employed'] / (X['age'] * 12 + 1)
        X['job_stability_score'] = np.where(
            X['employment_type'] == 'employed', 1,
            np.where(X['employment_type'] == 'self_employed', 0.7, 0.3)
        )
        
        # 4. Financial stability
        X['savings_to_income'] = X['savings_balance'] / (X['income'] + 1)
        X['total_assets'] = X['property_value'] + X['savings_balance'] + X['checking_balance']
        X['asset_to_debt'] = X['total_assets'] / (X['credit_amount'] + 1)
        X['financial_stability_score'] = (
            (X['savings_to_income'] > 0.2).astype(int) +
            (X['asset_to_debt'] > 2).astype(int) +
            (X['debt_to_income'] < 40).astype(int)
        )
        
        # 5. Credit burden
        X['credit_burden'] = X['existing_loans'] * X['credit_amount']
        X['total_monthly_obligations'] = X['housing_cost'] + X['monthly_payment']
        X['obligation_to_income'] = X['total_monthly_obligations'] / (X['income'] / 12 + 1)
        
        # 6. Risk flags
        X['high_utilization_flag'] = (X['credit_utilization'] > 0.8).astype(int)
        X['low_income_flag'] = (X['income'] < 30000).astype(int)
        X['short_employment_flag'] = (X['months_employed'] < 12).astype(int)
        X['high_inquiry_flag'] = (X['credit_inquiries'] > 3).astype(int)
        
        # 7. Interaction features
        X['risk_score_interaction'] = (
            X['credit_score'] * X['debt_to_income'] / 1000
        )
        
        return X

class DataProcessor:
    """Main data processing pipeline"""
    
    def __init__(self, config):
        self.config = config
        self.preprocessor = None
        
    def create_preprocessor(self):
        """Create preprocessing pipeline"""
        
        # Numerical pipeline
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Categorical pipeline
        cat_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Combine pipelines
        self.preprocessor = ColumnTransformer([
            ('num', num_pipeline, self.config.NUMERICAL_FEATURES),
            ('cat', cat_pipeline, self.config.CATEGORICAL_FEATURES)
        ])
        
        # Full pipeline with feature engineering
        self.full_pipeline = Pipeline([
            ('feature_engineer', FeatureEngineer()),
            ('preprocessor', self.preprocessor)
        ])
        
        return self.full_pipeline
    
    def process_data(self, df, train=True):
        """Process the data through the pipeline"""
        
        if train:
            X = df.drop(columns=[self.config.TARGET])
            y = df[self.config.TARGET]
            
            # Fit and transform
            X_processed = self.full_pipeline.fit_transform(X, y)
            
            # Get feature names
            num_features = self.config.NUMERICAL_FEATURES
            cat_features = self.preprocessor.named_transformers_['cat'].named_steps['encoder'].get_feature_names_out(
                self.config.CATEGORICAL_FEATURES
            )
            all_features = list(num_features) + list(cat_features)
            
            # Add engineered features
            engineered_features = FeatureEngineer().fit_transform(X).columns.difference(
                self.config.NUMERICAL_FEATURES + self.config.CATEGORICAL_FEATURES + [self.config.TARGET]
            )
            all_features.extend(engineered_features)
            
            return X_processed, y, all_features
            
        else:
            X = df.copy()
            if self.config.TARGET in df.columns:
                X = df.drop(columns=[self.config.TARGET])
                y = df[self.config.TARGET]
                X_processed = self.full_pipeline.transform(X)
                return X_processed, y
            else:
                X_processed = self.full_pipeline.transform(X)
                return X_processed
    
    def save_pipeline(self, path=None):
        """Save the preprocessing pipeline"""
        if path is None:
            path = self.config.PIPELINE_PATH
        
        joblib.dump(self.full_pipeline, path)
        print(f"Pipeline saved to {path}")
    
    def load_pipeline(self, path=None):
        """Load the preprocessing pipeline"""
        if path is None:
            path = self.config.PIPELINE_PATH
        
        self.full_pipeline = joblib.load(path)
        return self.full_pipeline
