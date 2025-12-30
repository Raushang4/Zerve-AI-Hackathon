"""Evaluation pipeline - calculates ROI and business metrics."""

import sys
import os
import json
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.roi_calculator import calculate_campaign_roi

def main():
    print("="*70)
    print("EVALUATION & ROI ANALYSIS")
    print("="*70)
    
    # Load predictions and test data
    predictions_df = pd.read_csv('results/predictions.csv')
    test_df = predictions_df.drop(['actual_churn', 'predicted_churn', 'churn_probability'], axis=1)
    test_df['churn'] = predictions_df['actual_churn']
    y_pred_proba = predictions_df['churn_probability'].values
    
    # Calculate ROI
    print("\n[Calculating Business ROI...]")
    print("-"*70)
    
    roi_summary = calculate_campaign_roi(test_df, predictions_df, y_pred_proba)
    
    # Save ROI analysis
    os.makedirs('results', exist_ok=True)
    with open('results/roi_analysis.json', 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        roi_summary_serializable = {
            'baseline': {k: float(v) if isinstance(v, (int, float)) else v for k, v in roi_summary['baseline'].items()},
            'ml_driven': {k: float(v) if isinstance(v, (int, float)) else v for k, v in roi_summary['ml_driven'].items()},
            'improvement': {k: float(v) if isinstance(v, (int, float)) else v for k, v in roi_summary['improvement'].items()},
            'threshold_analysis': [
                {k: float(v) if isinstance(v, (int, float)) else v for k, v in row.items()}
                for row in roi_summary['threshold_analysis']
            ]
        }
        json.dump(roi_summary_serializable, f, indent=2)
    
    print("\n✓ ROI analysis saved to results/roi_analysis.json")
    
    # Summary
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    print(f"\nBaseline ROI (random targeting): {roi_summary['baseline']['roi']:.1%}")
    print(f"ML-driven ROI (precision targeting): {roi_summary['ml_driven']['roi']:.1%}")
    print(f"ROI improvement: {roi_summary['improvement']['roi_lift']:.1%}")
    print(f"Incremental revenue: ${roi_summary['improvement']['incremental_revenue']:,.0f}")
    
    if roi_summary['ml_driven']['roi'] >= 4.0:
        print("\n✓ ROI TARGET MET (400%+ achieved)")
    
    print("\nNext step: python run_api.py")

if __name__ == '__main__':
    main()
