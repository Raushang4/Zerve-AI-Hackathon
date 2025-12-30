"""API server launcher."""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

if __name__ == '__main__':
    # Check if model exists
    if not os.path.exists('models/churn_model.pkl'):
        print("Error: Model not found. Please run 'python train.py' first.")
        sys.exit(1)
    
    from src.api import app
    from config import API_HOST, API_PORT, API_DEBUG
    
    print("\n" + "="*70)
    print("CHURN PREDICTION API SERVER")
    print("="*70)
    print(f"\nStarting server on {API_HOST}:{API_PORT}...")
    print(f"Debug mode: {API_DEBUG}")
    print(f"\nAPI Documentation:")
    print(f"  POST /predict          - Predict single customer churn")
    print(f"  POST /segment          - Classify customer segment")
    print(f"  POST /batch_analyze    - Analyze customer batch")
    print(f"  GET  /metrics          - Get model metrics")
    print(f"  GET  /health           - Health check")
    print(f"\nExample request:")
    print(f"  curl -X POST http://{API_HOST}:{API_PORT}/predict \\")
    print(f"    -H 'Content-Type: application/json' \\")
    print(f"    -d '{{\"monthly_spend\": 1500, \"contract_length_months\": 12, ")
    print(f"         \"support_tickets\": 5, \"feature_usage_score\": 0.75, ")
    print(f"         \"days_since_last_login\": 15, \"price_sensitivity_score\": 0.3, ")
    print(f"         \"competitor_engagement\": 0.1}}'")
    print(f"\n" + "="*70)
    print()
    
    app.run(host=API_HOST, port=API_PORT, debug=API_DEBUG)
