"""REST API for churn prediction."""

from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import json
from datetime import datetime

app = Flask(__name__)

# Load model and feature names
try:
    model = joblib.load('models/churn_model.pkl')
    feature_names = joblib.load('models/feature_names.pkl')
    # Load optimal threshold if available
    try:
        with open('models/optimal_threshold.pkl', 'rb') as f:
            import pickle
            optimal_threshold = pickle.load(f)
    except:
        optimal_threshold = 0.5
    print("✓ Model and feature names loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    feature_names = None
    optimal_threshold = 0.5


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'model_loaded': model is not None
    }), 200


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict churn for a single customer.
    
    Expected JSON payload:
    {
        "monthly_spend": 1500,
        "contract_length_months": 12,
        "support_tickets": 5,
        "feature_usage_score": 0.75,
        "days_since_last_login": 15,
        "price_sensitivity_score": 0.3,
        "competitor_engagement": 0.1
    }
    """
    if model is None or feature_names is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        
        # Use feature_names from model
        # Create feature vector in correct order
        X = np.array([[data[f] for f in feature_names]])
        
        # No scaling needed for tree-based models
        # Predict
        probability = model.predict_proba(X)[0, 1]
        prediction = int(probability >= optimal_threshold)
        
        # Determine risk level
        if probability >= 0.7:
            risk_level = 'HIGH'
        elif probability >= 0.5:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'LOW'
        
        return jsonify({
            'churn_prediction': int(prediction),
            'churn_probability': float(probability),
            'risk_level': risk_level,
            'recommendation': 'Target for retention campaign' if prediction == 1 else 'Monitor'
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/segment', methods=['POST'])
def segment():
    """
    Classify customer into high-value at-risk segment.
    
    Expected JSON payload:
    {
        "customer_id": "C001",
        "monthly_spend": 2000,
        "churn_probability": 0.72
    }
    """
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        
        customer_id = data.get('customer_id', 'UNKNOWN')
        monthly_spend = float(data.get('monthly_spend', 0))
        churn_prob = float(data.get('churn_probability', 0))
        
        # Segmentation logic
        high_value_threshold = 1500
        churn_risk_threshold = 0.5
        
        is_high_value = monthly_spend >= high_value_threshold
        is_churn_risk = churn_prob >= churn_risk_threshold
        
        # Segment classification
        if is_high_value and is_churn_risk:
            segment = 'HIGH_VALUE_AT_RISK'
            priority = 1  # Highest
            campaign_recommendation = 'Premium retention offer'
        elif is_high_value:
            segment = 'HIGH_VALUE_STABLE'
            priority = 2
            campaign_recommendation = 'Loyalty rewards'
        elif is_churn_risk:
            segment = 'AT_RISK'
            priority = 3
            campaign_recommendation = 'Standard retention'
        else:
            segment = 'STANDARD'
            priority = 4
            campaign_recommendation = 'Monitor'
        
        return jsonify({
            'customer_id': customer_id,
            'segment': segment,
            'priority': priority,
            'monthly_spend': monthly_spend,
            'churn_probability': churn_prob,
            'is_high_value': is_high_value,
            'is_churn_risk': is_churn_risk,
            'campaign_recommendation': campaign_recommendation
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/batch_analyze', methods=['POST'])
def batch_analyze():
    """
    Analyze a batch of customers and identify high-value at-risk segment.
    
    Expected JSON payload:
    {
        "file_path": "data/test.csv"
    }
    """
    if model is None or feature_names is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        file_path = data.get('file_path')
        
        if not file_path:
            return jsonify({'error': 'Missing file_path'}), 400
        
        # Load data
        df = pd.read_csv(file_path)
        
        X = df[feature_names].values
        
        # Predict
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1]
        
        # Analyze segment
        high_value_threshold = 1500
        high_value_at_risk = (
            (df['monthly_spend'] >= high_value_threshold) & 
            (probabilities >= 0.5)
        ).sum()
        
        total_high_value = (df['monthly_spend'] >= high_value_threshold).sum()
        total_at_risk = (predictions == 1).sum()
        
        # Calculate potential ROI
        campaign_cost_per_customer = 50
        annual_value = 5000
        save_rate = 0.20
        
        potential_revenue = high_value_at_risk * annual_value * save_rate
        campaign_cost = high_value_at_risk * campaign_cost_per_customer
        potential_roi = (potential_revenue - campaign_cost) / campaign_cost if campaign_cost > 0 else 0
        
        return jsonify({
            'total_customers': len(df),
            'high_value_customers': int(total_high_value),
            'at_risk_customers': int(total_at_risk),
            'high_value_at_risk': int(high_value_at_risk),
            'high_value_at_risk_pct': float(high_value_at_risk / len(df) * 100),
            'segment_summary': {
                'total_campaign_targets': int(high_value_at_risk),
                'estimated_saves': float(high_value_at_risk * save_rate),
                'campaign_cost': float(campaign_cost),
                'potential_revenue': float(potential_revenue),
                'potential_roi': float(potential_roi)
            }
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/metrics', methods=['GET'])
def metrics():
    """Get model performance metrics."""
    try:
        with open('results/metrics.json', 'r') as f:
            metrics_data = json.load(f)
        return jsonify(metrics_data), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Endpoint not found',
        'available_endpoints': [
            'GET /health',
            'POST /predict',
            'POST /segment',
            'POST /batch_analyze',
            'GET /metrics'
        ]
    }), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    from config import API_HOST, API_PORT, API_DEBUG
    
    print(f"Starting Churn Prediction API...")
    print(f"Running on {API_HOST}:{API_PORT}")
    print(f"Debug mode: {API_DEBUG}")
    print(f"\nAvailable endpoints:")
    print(f"  GET  /health           - Health check")
    print(f"  POST /predict          - Single customer prediction")
    print(f"  POST /segment          - Customer segmentation")
    print(f"  POST /batch_analyze    - Batch analysis")
    print(f"  GET  /metrics          - Model metrics")
    
    app.run(host=API_HOST, port=API_PORT, debug=API_DEBUG)
