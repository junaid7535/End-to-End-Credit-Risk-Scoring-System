## Overview
A comprehensive machine learning system for predicting customer loan default risk. This system includes data processing, model training, evaluation, interpretation, and deployment components.

## Features
- **Data Generation & Processing**: Synthetic data generation with realistic patterns
- **Feature Engineering**: Automated creation of risk indicators and financial ratios
- **Multiple Models**: XGBoost, LightGBM, CatBoost, Random Forest, Logistic Regression
- **Model Interpretation**: SHAP values, feature importance, individual prediction explanations
- **API Deployment**: REST API for real-time predictions
- **Monitoring**: Drift detection and performance monitoring
- **Business Metrics**: Profit analysis and risk-based decision support

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/credit-risk-scoring.git
cd credit-risk-scoring

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt