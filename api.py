from flask import Flask, request, jsonify
from flask_restx import Api, Resource, fields
import numpy as np
import pandas as pd
import joblib
import json
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
api = Api(
    app,
    version='1.0',
    title='Credit Risk Scoring API',
    description='API for predicting credit default risk',
    doc='/docs'
)

# Load configuration
config = Config()

# Load artifacts
try:
    pipeline = joblib.load(config.PIPELINE_PATH)
    model = joblib.load(config.FINAL_MODEL_PATH)
    
    with open(config.THRESHOLD_PATH, 'r') as f:
        threshold_data = json.load(f)
        THRESHOLD = threshold_data['threshold']
    
    with open(config.MODELS_DIR / 'feature_info.json', 'r') as f:
        feature_info = json.load(f)
        FEATURE_NAMES = feature_info['feature_names']
    
    print("Model artifacts loaded successfully")
    
except Exception as e:
    print(f"Error loading artifacts: {e}")
    pipeline = None
    model = None
    THRESHOLD = 0.5
    FEATURE_NAMES = []

# Define request/response models
credit_request = api.model('CreditRequest', {
    'age': fields.Integer(required=True, description='Customer age', example=35),
    'income': fields.Float(required=True, description='Annual income', example=75000),
    'credit_score': fields.Float(required=True, description='Credit score', example=720),
    'credit_amount': fields.Float(required=True, description='Loan amount requested', example=25000),
    'loan_duration': fields.Integer(required=True, description='Loan duration in months', example=36),
    'credit_utilization': fields.Float(required=True, description='Credit utilization ratio', example=0.45),
    'debt_to_income': fields.Float(required=True, description='Debt to income ratio', example=0.35),
    'employment_type': fields.String(required=True, description='Employment type', 
                                   enum=['employed', 'self_employed', 'unemployed', 'retired'],
                                   example='employed'),
    'education': fields.String(required=True, description='Education level',
                             enum=['high_school', 'bachelors', 'masters', 'phd'],
                             example='bachelors'),
    'home_ownership': fields.String(required=True, description='Home ownership status',
                                  enum=['own', 'mortgage', 'rent', 'other'],
                                  example='mortgage')
})

prediction_response = api.model('PredictionResponse', {
    'prediction': fields.Integer(description='0: No default, 1: Default', example=0),
    'probability': fields.Float(description='Probability of default', example=0.12),
    'risk_level': fields.String(description='Risk level category', 
                              enum=['low', 'medium', 'high'],
                              example='low'),
    'threshold_used': fields.Float(description='Decision threshold used', example=0.42),
    'timestamp': fields.String(description='Prediction timestamp'),
    'features_contributing': fields.List(fields.String, description='Top contributing features')
})

batch_response = api.model('BatchResponse', {
    'predictions': fields.List(fields.Nested(prediction_response)),
    'summary': fields.Raw(description='Batch prediction summary')
})

class CreditPredictor:
    """Main predictor class"""
    
    @staticmethod
    def preprocess_input(data):
        """Preprocess input data"""
        # Convert to DataFrame
        df = pd.DataFrame([data])
        
        # Add missing columns with default values
        default_values = {
            'existing_loans': 2,
            'credit_inquiries': 1,
            'total_credit_lines': 5,
            'months_employed': 60,
            'residence_months': 48,
            'dependents': 1,
            'housing_cost': 1500,
            'property_value': 300000,
            'savings_balance': 10000,
            'checking_balance': 5000,
            'marital_status': 'married',
            'credit_history': 'good',
            'purpose': 'debt_consolidation'
        }
        
        for col, default_val in default_values.items():
            if col not in df.columns:
                df[col] = default_val
        
        return df
    
    @staticmethod
    def predict_risk(probability):
        """Determine risk level based on probability"""
        if probability < 0.3:
            return 'low'
        elif probability < 0.7:
            return 'medium'
        else:
            return 'high'
    
    @staticmethod
    def get_top_features(probability, threshold=THRESHOLD):
        """Get features contributing to prediction (simplified)"""
        # In a real implementation, this would use SHAP or LIME
        # For now, return placeholder
        if probability > threshold:
            return ['credit_utilization', 'debt_to_income', 'credit_score']
        else:
            return ['income', 'employment_type', 'credit_history']

@api.route('/predict')
class Predict(Resource):
    @api.expect(credit_request)
    @api.marshal_with(prediction_response)
    def post(self):
        """Make a single prediction"""
        try:
            data = request.json
            
            # Preprocess
            input_df = CreditPredictor.preprocess_input(data)
            
            # Transform
            processed_data = pipeline.transform(input_df)
            
            # Predict
            probability = model.predict_proba(processed_data)[0, 1]
            prediction = 1 if probability >= THRESHOLD else 0
            
            # Prepare response
            response = {
                'prediction': int(prediction),
                'probability': float(probability),
                'risk_level': CreditPredictor.predict_risk(probability),
                'threshold_used': float(THRESHOLD),
                'timestamp': datetime.now().isoformat(),
                'features_contributing': CreditPredictor.get_top_features(probability)
            }
            
            return response, 200
            
        except Exception as e:
            return {'error': str(e)}, 400

@api.route('/predict_batch')
class PredictBatch(Resource):
    @api.expect(api.model('BatchRequest', {
        'applications': fields.List(fields.Nested(credit_request))
    }))
    @api.marshal_with(batch_response)
    def post(self):
        """Make batch predictions"""
        try:
            data = request.json
            applications = data.get('applications', [])
            
            predictions = []
            approved = 0
            rejected = 0
            
            for app in applications:
                # Preprocess
                input_df = CreditPredictor.preprocess_input(app)
                
                # Transform
                processed_data = pipeline.transform(input_df)
                
                # Predict
                probability = model.predict_proba(processed_data)[0, 1]
                prediction = 1 if probability >= THRESHOLD else 0
                
                if prediction == 0:
                    approved += 1
                else:
                    rejected += 1
                
                pred_response = {
                    'prediction': int(prediction),
                    'probability': float(probability),
                    'risk_level': CreditPredictor.predict_risk(probability),
                    'threshold_used': float(THRESHOLD),
                    'timestamp': datetime.now().isoformat(),
                    'features_contributing': CreditPredictor.get_top_features(probability)
                }
                
                predictions.append(pred_response)
            
            summary = {
                'total_applications': len(applications),
                'approved': approved,
                'rejected': rejected,
                'approval_rate': approved / len(applications) if applications else 0,
                'average_risk_score': np.mean([p['probability'] for p in predictions]) if predictions else 0
            }
            
            return {
                'predictions': predictions,
                'summary': summary
            }, 200
            
        except Exception as e:
            return {'error': str(e)}, 400

@api.route('/health')
class Health(Resource):
    def get(self):
        """Health check endpoint"""
        return {
            'status': 'healthy',
            'model_loaded': model is not None,
            'pipeline_loaded': pipeline is not None,
            'threshold': THRESHOLD,
            'timestamp': datetime.now().isoformat()
        }, 200

@api.route('/threshold')
class Threshold(Resource):
    def get(self):
        """Get current decision threshold"""
        return {'threshold': THRESHOLD}, 200
    
    @api.expect(api.model('ThresholdUpdate', {
        'threshold': fields.Float(required=True, min=0, max=1, example=0.45)
    }))
    def put(self):
        """Update decision threshold"""
        try:
            data = request.json
            new_threshold = data.get('threshold')
            
            if not 0 <= new_threshold <= 1:
                return {'error': 'Threshold must be between 0 and 1'}, 400
            
            global THRESHOLD
            THRESHOLD = new_threshold
            
            # Save to file
            with open(config.THRESHOLD_PATH, 'w') as f:
                json.dump({'threshold': new_threshold}, f)
            
            return {'message': f'Threshold updated to {new_threshold}'}, 200
            
        except Exception as e:
            return {'error': str(e)}, 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)