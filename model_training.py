import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import json
import joblib

class ModelTrainer:
    """Train and evaluate credit risk models"""
    
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_threshold = 0.5
    
    def train_models(self, X_train, y_train, X_val, y_val, feature_names=None):
        """Train multiple models"""
        
        print("Training multiple models...")
        
        # Define models
        self.models = {
            'logistic_regression': LogisticRegression(
                max_iter=1000,
                random_state=self.config.RANDOM_STATE,
                class_weight='balanced'
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=self.config.RANDOM_STATE,
                class_weight='balanced',
                n_jobs=-1
            ),
            'xgboost': xgb.XGBClassifier(
                **self.config.XGB_PARAMS,
                eval_metric='logloss',
                scale_pos_weight=self.calculate_class_weight(y_train)
            ),
            'lightgbm': lgb.LGBMClassifier(
                **self.config.LGB_PARAMS,
                class_weight='balanced'
            ),
            'catboost': cb.CatBoostClassifier(
                iterations=300,
                depth=6,
                learning_rate=0.01,
                random_seed=self.config.RANDOM_STATE,
                verbose=False,
                auto_class_weights='Balanced'
            )
        }
        
        # Train each model
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            # Handle class imbalance with SMOTE
            if name != 'catboost':  # CatBoost has built-in handling
                smote = SMOTE(random_state=self.config.RANDOM_STATE)
                X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
                model.fit(X_train_resampled, y_train_resampled)
            else:
                model.fit(X_train, y_train)
            
            # Evaluate on validation set
            y_pred = model.predict(X_val)
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            
            # Store results
            self.results[name] = {
                'model': model,
                'val_predictions': y_pred,
                'val_probabilities': y_pred_proba,
                'metrics': self.calculate_metrics(y_val, y_pred, y_pred_proba)
            }
            
            print(f"{name} - ROC AUC: {self.results[name]['metrics']['roc_auc']:.4f}")
        
        # Select best model
        self.select_best_model()
        
        # Optimize threshold
        self.optimize_threshold(X_val, y_val)
        
        return self.best_model
    
    def calculate_class_weight(self, y):
        """Calculate class weight for imbalanced data"""
        n_pos = np.sum(y == 1)
        n_neg = np.sum(y == 0)
        return n_neg / n_pos if n_pos > 0 else 1
    
    def calculate_metrics(self, y_true, y_pred, y_pred_proba):
        """Calculate all evaluation metrics"""
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_pred_proba),
            'pr_auc': average_precision_score(y_true, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
        
        # Calculate at different thresholds
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
        for threshold in thresholds:
            y_pred_thresh = (y_pred_proba >= threshold).astype(int)
            metrics[f'precision_{threshold}'] = precision_score(y_true, y_pred_thresh, zero_division=0)
            metrics[f'recall_{threshold}'] = recall_score(y_true, y_pred_thresh, zero_division=0)
            metrics[f'f1_{threshold}'] = f1_score(y_true, y_pred_thresh, zero_division=0)
        
        return metrics
    
    def select_best_model(self):
        """Select the best model based on validation performance"""
        best_score = 0
        best_model_name = None
        
        for name, result in self.results.items():
            # Use weighted score (60% ROC AUC + 40% PR AUC)
            score = 0.6 * result['metrics']['roc_auc'] + 0.4 * result['metrics']['pr_auc']
            
            if score > best_score:
                best_score = score
                best_model_name = name
        
        self.best_model = self.results[best_model_name]['model']
        print(f"\nBest model: {best_model_name} with score: {best_score:.4f}")
        
        return best_model_name
    
    def optimize_threshold(self, X_val, y_val):
        """Optimize decision threshold based on business needs"""
        
        y_pred_proba = self.best_model.predict_proba(X_val)[:, 1]
        
        # Try different thresholds
        thresholds = np.arange(0.1, 0.9, 0.05)
        best_f1 = 0
        best_threshold = 0.5
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            f1 = f1_score(y_val, y_pred, zero_division=0)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        self.best_threshold = best_threshold
        print(f"Optimal threshold: {best_threshold:.3f} (F1: {best_f1:.4f})")
        
        # Save threshold
        threshold_data = {'threshold': float(best_threshold)}
        with open(self.config.THRESHOLD_PATH, 'w') as f:
            json.dump(threshold_data, f)
    
    def cross_validate(self, X, y, model, cv_folds=5):
        """Perform cross-validation"""
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, 
                           random_state=self.config.RANDOM_STATE)
        
        scores = cross_val_score(
            model, X, y,
            cv=cv,
            scoring='roc_auc',
            n_jobs=-1
        )
        
        return scores
    
    def save_model(self, model=None, path=None):
        """Save the trained model"""
        if model is None:
            model = self.best_model
        
        if path is None:
            path = self.config.FINAL_MODEL_PATH
        
        joblib.dump(model, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path=None):
        """Load a trained model"""
        if path is None:
            path = self.config.FINAL_MODEL_PATH
        
        self.best_model = joblib.load(path)
        return self.best_model