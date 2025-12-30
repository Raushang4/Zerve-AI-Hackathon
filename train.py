"""Training pipeline - orchestrates data generation and model training."""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data_generator import generate_customer_data
from src.model_trainer import train_churn_model

def main():
    print("="*70)
    print("CUSTOMER CHURN PREDICTION - TRAINING PIPELINE")
    print("="*70)
    
    # Step 1: Generate synthetic data
    print("\n[Step 1/2] Generating synthetic customer data...")
    print("-"*70)
    
    data = generate_customer_data()
    os.makedirs('data', exist_ok=True)
    data.to_csv('data/generated_customers.csv', index=False)
    
    print(f"\n✓ Generated {len(data)} customers")
    print(f"✓ Churn rate: {data['churn'].mean():.1%}")
    print(f"✓ Data saved to data/generated_customers.csv")
    
    # Step 2: Train model
    print("\n[Step 2/2] Training churn prediction model...")
    print("-"*70)
    
    model, scaler, metrics, X_test, y_test, y_pred_proba, feature_importance = train_churn_model(
        data_path='data/generated_customers.csv'
    )
    
    print("\n" + "="*70)
    print("TRAINING PIPELINE COMPLETE")
    print("="*70)
    print("\n✓ Model trained and saved")
    print(f"✓ Accuracy: {metrics['accuracy']:.1%}")
    print(f"✓ Precision: {metrics['precision']:.1%}")
    print(f"✓ Recall: {metrics['recall']:.1%}")
    print(f"✓ ROC-AUC: {metrics['roc_auc']:.1%}")
    print("\nNext step: python evaluate.py")

if __name__ == '__main__':
    main()
