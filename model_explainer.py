import shap
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
import json
from datetime import datetime

class ModelExplainer:
    """Explain model predictions and feature importance"""
    
    def __init__(self, model, preprocessor, feature_names):
        self.model = model
        self.preprocessor = preprocessor
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
        
    def calculate_shap_values(self, X, sample_size=1000):
        """Calculate SHAP values for model interpretation"""
        
        print("Calculating SHAP values...")
        
        # Sample data for faster computation
        if len(X) > sample_size:
            sample_idx = np.random.choice(len(X), sample_size, replace=False)
            X_sample = X[sample_idx]
        else:
            X_sample = X
        
        # Create explainer based on model type
        if hasattr(self.model, 'predict_proba'):
            try:
                self.explainer = shap.TreeExplainer(self.model)
                self.shap_values = self.explainer.shap_values(X_sample)
                
                # For binary classification
                if isinstance(self.shap_values, list):
                    self.shap_values = self.shap_values[1]  # Positive class
            except:
                # Fallback to Kernel SHAP
                self.explainer = shap.KernelExplainer(self.model.predict_proba, X_sample[:100])
                self.shap_values = self.explainer.shap_values(X_sample[:200])[1]
        
        return self.shap_values
    
    def plot_feature_importance(self, X, y, save_path=None):
        """Plot feature importance using multiple methods"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Tree-based feature importance
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[-20:]  # Top 20 features
            
            axes[0, 0].barh(range(len(indices)), importances[indices])
            axes[0, 0].set_yticks(range(len(indices)))
            axes[0, 0].set_yticklabels([self.feature_names[i] for i in indices])
            axes[0, 0].set_xlabel('Feature Importance')
            axes[0, 0].set_title('Tree-based Feature Importance')
        
        # 2. SHAP summary plot
        if self.shap_values is not None:
            shap.summary_plot(
                self.shap_values, 
                X,
                feature_names=self.feature_names,
                show=False,
                max_display=20,
                plot_type='dot',
                plot_size=None
            )
            fig_shap, ax_shap = plt.gcf(), plt.gca()
            ax_shap.set_title('SHAP Feature Importance')
            plt.close(fig_shap)
            
            # Get current figure and axes for copying
            current_fig = plt.gcf()
            for i, ax in enumerate(current_fig.axes):
                if i == 0:  # Main plot
                    # We need to recreate it in our subplot
                    axes[0, 1].clear()
                    # This is simplified - in practice you'd need to extract the data
                    axes[0, 1].set_title('SHAP Values (simplified)')
        
        # 3. Permutation importance
        try:
            perm_importance = permutation_importance(
                self.model, X, y,
                n_repeats=10,
                random_state=42,
                n_jobs=-1
            )
            
            sorted_idx = perm_importance.importances_mean.argsort()[-20:]
            axes[1, 0].boxplot(
                perm_importance.importances[sorted_idx].T,
                vert=False,
                labels=[self.feature_names[i] for i in sorted_idx]
            )
            axes[1, 0].set_title('Permutation Importance')
            axes[1, 0].set_xlabel('Decrease in Accuracy')
        
        except Exception as e:
            axes[1, 0].text(0.5, 0.5, f'Error: {str(e)}', 
                          ha='center', va='center')
            axes[1, 0].set_title('Permutation Importance')
        
        # 4. Correlation with target
        try:
            if isinstance(X, np.ndarray):
                X_df = pd.DataFrame(X, columns=self.feature_names)
            else:
                X_df = X.copy()
            
            X_df['target'] = y
            correlations = X_df.corr()['target'].drop('target').abs().sort_values()
            
            axes[1, 1].barh(range(len(correlations[-20:])), correlations[-20:])
            axes[1, 1].set_yticks(range(len(correlations[-20:])))
            axes[1, 1].set_yticklabels(correlations[-20:].index)
            axes[1, 1].set_xlabel('Absolute Correlation')
            axes[1, 1].set_title('Feature Correlation with Target')
        
        except Exception as e:
            axes[1, 1].text(0.5, 0.5, f'Error: {str(e)}', 
                          ha='center', va='center')
            axes[1, 1].set_title('Feature Correlation')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance plot saved to {save_path}")
        
        plt.show()
    
    def explain_prediction(self, X_instance, feature_names=None, top_n=10):
        """Explain individual prediction"""
        
        if feature_names is None:
            feature_names = self.feature_names
        
        # Get prediction probability
        proba = self.model.predict_proba(X_instance.reshape(1, -1))[0]
        prediction = np.argmax(proba)
        
        # Get SHAP values for this instance
        if self.explainer is not None:
            shap_value = self.explainer.shap_values(X_instance.reshape(1, -1))
            if isinstance(shap_value, list):
                shap_value = shap_value[1]  # Positive class
            
            # Sort features by absolute SHAP value
            shap_series = pd.Series(
                shap_value[0],
                index=feature_names
            ).sort_values(key=abs, ascending=False)
            
            # Create explanation
            explanation = {
                'prediction': int(prediction),
                'probability_default': float(proba[1]),
                'probability_no_default': float(proba[0]),
                'top_features': []
            }
            
            for i, (feature, value) in enumerate(shap_series.head(top_n).items()):
                contribution = 'increases' if value > 0 else 'decreases'
                explanation['top_features'].append({
                    'feature': feature,
                    'shap_value': float(value),
                    'contribution': contribution,
                    'impact': 'risk' if (value > 0 and prediction == 1) or (value < 0 and prediction == 0) else 'safety'
                })
            
            return explanation
        
        return None
    
    def generate_risk_report(self, X, y, predictions, save_path=None):
        """Generate comprehensive risk analysis report"""
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'model_performance': {},
            'risk_distribution': {},
            'feature_analysis': {},
            'recommendations': []
        }
        
        # Calculate metrics
        from sklearn.metrics import classification_report, roc_auc_score
        
        report['model_performance'] = classification_report(
            y, predictions, output_dict=True
        )
        report['model_performance']['roc_auc'] = roc_auc_score(y, predictions)
        
        # Risk distribution
        risk_scores = self.model.predict_proba(X)[:, 1]
        report['risk_distribution'] = {
            'low_risk': float(np.mean(risk_scores < 0.3)),
            'medium_risk': float(np.mean((risk_scores >= 0.3) & (risk_scores < 0.7))),
            'high_risk': float(np.mean(risk_scores >= 0.7)),
            'mean_risk_score': float(np.mean(risk_scores)),
            'std_risk_score': float(np.std(risk_scores))
        }
        
        # Feature analysis
        if hasattr(self.model, 'feature_importances_'):
            importances = pd.Series(
                self.model.feature_importances_,
                index=self.feature_names
            ).sort_values(ascending=False)
            
            report['feature_analysis']['top_10_features'] = importances.head(10).to_dict()
        
        # Business recommendations
        high_risk_indices = np.where(risk_scores > 0.7)[0]
        if len(high_risk_indices) > 0:
            report['recommendations'].append(
                f"Found {len(high_risk_indices)} high-risk applications. Recommend manual review."
            )
        
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"Risk report saved to {save_path}")
        
        return report
