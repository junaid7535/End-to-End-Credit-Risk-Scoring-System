import logging
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import json

def setup_logging(log_file: str = None):
    """Setup logging configuration"""
    
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers = [logging.StreamHandler()]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=handlers
    )
    
    return logging.getLogger(__name__)

def save_results(results: Dict[str, Any], filepath: str):
    """Save results to JSON file"""
    
    # Convert numpy types to Python types
    def convert_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict()
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, datetime):
            return obj.isoformat()
        return obj
    
    with open(filepath, 'w') as f:
        json.dump(results, f, default=convert_types, indent=2)
    
    print(f"Results saved to {filepath}")

def load_results(filepath: str) -> Dict[str, Any]:
    """Load results from JSON file"""
    
    with open(filepath, 'r') as f:
        results = json.load(f)
    
    return results

def calculate_business_metrics(y_true, y_pred, y_pred_proba, 
                              interest_rate=0.08, loss_given_default=0.45):
    """
    Calculate business metrics for credit risk model
    
    Parameters:
    -----------
    interest_rate: Average interest rate on loans
    loss_given_default: Percentage of loan lost when default occurs
    """
    
    # Convert to arrays if needed
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred_proba = np.array(y_pred_proba)
    
    # Basic confusion matrix
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    
    # Expected profit calculation (simplified)
    # Assuming average loan amount of $20,000
    avg_loan = 20000
    
    # Revenue from good loans
    good_loan_revenue = tn * avg_loan * interest_rate
    
    # Loss from bad loans (false negatives)
    bad_loan_loss = fn * avg_loan * loss_given_default
    
    # Opportunity cost from rejected good loans (false positives)
    opportunity_cost = fp * avg_loan * interest_rate
    
    # Net profit
    net_profit = good_loan_revenue - bad_loan_loss - opportunity_cost
    
    # Profit per loan
    profit_per_loan = net_profit / len(y_true) if len(y_true) > 0 else 0
    
    metrics = {
        'confusion_matrix': {
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn)
        },
        'profit_analysis': {
            'good_loan_revenue': float(good_loan_revenue),
            'bad_loan_loss': float(bad_loan_loss),
            'opportunity_cost': float(opportunity_cost),
            'net_profit': float(net_profit),
            'profit_per_loan': float(profit_per_loan)
        },
        'business_ratios': {
            'approval_rate': float((tn + fp) / len(y_true)) if len(y_true) > 0 else 0,
            'bad_rate_approved': float(fn / (tn + fn + 1e-10)),
            'good_rate_rejected': float(fp / (tp + fp + 1e-10))
        }
    }
    
    return metrics

def create_scorecard(model, feature_names, scaler=1000, offset=600):
    """
    Create a credit scorecard from model coefficients
    
    Parameters:
    -----------
    scaler: Points to double odds (typically 20-60)
    offset: Base score offset
    """
    
    scorecard = {}
    
    if hasattr(model, 'coef_'):
        # Logistic regression
        coef = model.coef_[0]
        intercept = model.intercept_[0]
        
        for i, (feature, coef_val) in enumerate(zip(feature_names, coef)):
            # Calculate points per feature (simplified)
            points = coef_val * scaler / np.log(2)
            scorecard[feature] = {
                'coefficient': float(coef_val),
                'points': float(points),
                'importance_rank': i + 1
            }
        
        # Base score
        base_score = offset - intercept * scaler / np.log(2)
        scorecard['_base_score'] = float(base_score)
    
    elif hasattr(model, 'feature_importances_'):
        # Tree-based models
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        for rank, idx in enumerate(indices):
            if rank < 20:  # Top 20 features
                scorecard[feature_names[idx]] = {
                    'importance': float(importances[idx]),
                    'importance_rank': rank + 1,
                    'normalized_importance': float(importances[idx] / importances.max())
                }
    
    return scorecard

