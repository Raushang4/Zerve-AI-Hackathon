"""ROI calculation for churn prediction campaigns."""

import pandas as pd
import numpy as np
from config import (
    AVERAGE_CUSTOMER_ANNUAL_VALUE,
    CAMPAIGN_COST_PER_CUSTOMER,
    SAVE_RATE,
    HIGH_VALUE_THRESHOLD,
    BASELINE_RANDOM_ROI
)


def calculate_campaign_roi(test_df, predictions_df, y_pred_proba):
    """
    Calculate business ROI for ML-driven vs. baseline random targeting.
    
    Assumptions:
    - Average customer annual value: $5,000
    - Campaign cost per customer: $50
    - Target save rate: 20% (realistic for retention campaigns)
    - Baseline random targeting achieves 165% ROI
    
    Returns:
        ROI analysis with financial metrics
    """
    
    # Add predictions to test data
    df = test_df.copy()
    df['churn_probability'] = y_pred_proba
    
    # Segment: High-value customers (monthly spend > $1,500)
    high_value_customers = df[df['monthly_spend'] >= HIGH_VALUE_THRESHOLD].copy()
    high_value_customers['is_high_value'] = True
    
    n_high_value = len(high_value_customers)
    n_high_value_predicted_churn = (high_value_customers['churn_probability'] > 0.5).sum()
    
    print("="*70)
    print("ROI ANALYSIS: ML-DRIVEN vs. BASELINE RANDOM TARGETING")
    print("="*70)
    
    print(f"\nMarket Composition:")
    print(f"  Total customers in test set: {len(df)}")
    print(f"  High-value customers (>${HIGH_VALUE_THRESHOLD}/mo): {n_high_value} ({n_high_value/len(df)*100:.1f}%)")
    print(f"  Actual churners: {df['churn'].sum()} ({df['churn'].mean()*100:.1f}%)")
    print(f"  High-value churners: {high_value_customers['churn'].sum()}")
    
    # BASELINE: Random targeting (10% of customer base)
    baseline_targets = int(len(df) * 0.10)
    baseline_saved = baseline_targets * SAVE_RATE
    baseline_saved_value = baseline_saved * AVERAGE_CUSTOMER_ANNUAL_VALUE
    baseline_campaign_cost = baseline_targets * CAMPAIGN_COST_PER_CUSTOMER
    baseline_net_benefit = baseline_saved_value - baseline_campaign_cost
    baseline_roi = baseline_net_benefit / baseline_campaign_cost if baseline_campaign_cost > 0 else 0
    
    print(f"\nBASELINE: Random Targeting")
    print(f"  Targets: {baseline_targets} customers (10% of base)")
    print(f"  Expected saves: {baseline_saved:.0f} customers @ {SAVE_RATE*100:.0f}% save rate")
    print(f"  Revenue retained: ${baseline_saved_value:,.0f}")
    print(f"  Campaign cost: ${baseline_campaign_cost:,.0f}")
    print(f"  Net benefit: ${baseline_net_benefit:,.0f}")
    print(f"  ROI: {baseline_roi:.1%}")
    
    # ML-DRIVEN: Target high-value at-risk customers
    # Target customers with churn probability > 0.5 from high-value segment
    ml_targets = high_value_customers[high_value_customers['churn_probability'] > 0.5].copy()
    n_ml_targets = len(ml_targets)
    
    # Calculate precision: how many are actual churners
    ml_true_positives = ml_targets['churn'].sum()
    ml_precision = ml_true_positives / n_ml_targets if n_ml_targets > 0 else 0
    
    # Expected saves: true positives * save rate
    ml_saved = ml_true_positives * SAVE_RATE
    ml_saved_value = ml_saved * AVERAGE_CUSTOMER_ANNUAL_VALUE
    ml_campaign_cost = n_ml_targets * CAMPAIGN_COST_PER_CUSTOMER
    ml_net_benefit = ml_saved_value - ml_campaign_cost
    ml_roi = ml_net_benefit / ml_campaign_cost if ml_campaign_cost > 0 else 0
    
    print(f"\nML-DRIVEN: High-Value At-Risk Targeting")
    print(f"  Targets: {n_ml_targets} customers (precision-selected)")
    print(f"  Actual churners (TP): {ml_true_positives} ({ml_precision:.1%} precision)")
    print(f"  Expected saves: {ml_saved:.0f} customers @ {SAVE_RATE*100:.0f}% save rate")
    print(f"  Revenue retained: ${ml_saved_value:,.0f}")
    print(f"  Campaign cost: ${ml_campaign_cost:,.0f}")
    print(f"  Net benefit: ${ml_net_benefit:,.0f}")
    print(f"  ROI: {ml_roi:.1%}")
    
    # Improvement
    roi_improvement = (ml_roi - baseline_roi) / baseline_roi if baseline_roi > 0 else 0
    incremental_value = ml_saved_value - baseline_saved_value
    
    print(f"\nIMPROVEMENT METRICS")
    print(f"  ROI improvement: {roi_improvement:.1%} vs. baseline")
    print(f"  Cost efficiency: {ml_campaign_cost/n_ml_targets:.2f}$/customer (vs ${CAMPAIGN_COST_PER_CUSTOMER}/customer baseline)")
    print(f"  Incremental saved customers: {ml_saved - baseline_saved:.0f}")
    print(f"  Incremental revenue: ${incremental_value:,.0f}")
    print(f"  Campaign efficiency: {ml_saved_value/ml_campaign_cost:.2f}x return")
    
    if ml_roi >= 4.0:
        print(f"\n✓ ROI TARGET MET: {ml_roi:.1%} >= 400%")
    else:
        print(f"\n⚠ ROI below target: {ml_roi:.1%} < 400%")
    
    # Analysis by churn probability
    print(f"\n" + "="*70)
    print("TARGETING THRESHOLD ANALYSIS")
    print("="*70)
    
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    analysis_results = []
    
    for threshold in thresholds:
        target_customers = high_value_customers[high_value_customers['churn_probability'] >= threshold]
        n_targets = len(target_customers)
        true_positives = target_customers['churn'].sum()
        precision = true_positives / n_targets if n_targets > 0 else 0
        recall_at_threshold = true_positives / high_value_customers['churn'].sum() if high_value_customers['churn'].sum() > 0 else 0
        
        saved = true_positives * SAVE_RATE
        saved_value = saved * AVERAGE_CUSTOMER_ANNUAL_VALUE
        cost = n_targets * CAMPAIGN_COST_PER_CUSTOMER
        net = saved_value - cost
        roi = net / cost if cost > 0 else 0
        
        analysis_results.append({
            'threshold': threshold,
            'n_targets': n_targets,
            'precision': precision,
            'recall': recall_at_threshold,
            'saved': saved,
            'cost': cost,
            'roi': roi
        })
        
        print(f"\nThreshold: {threshold:.1f}")
        print(f"  Targets: {n_targets} | Precision: {precision:.1%} | Recall: {recall_at_threshold:.1%}")
        print(f"  Expected saves: {saved:.0f} | Cost: ${cost:,.0f} | ROI: {roi:.1%}")
    
    # Return summary
    summary = {
        'baseline': {
            'targets': baseline_targets,
            'saved': baseline_saved,
            'revenue': baseline_saved_value,
            'cost': baseline_campaign_cost,
            'roi': baseline_roi
        },
        'ml_driven': {
            'targets': n_ml_targets,
            'precision': ml_precision,
            'saved': ml_saved,
            'revenue': ml_saved_value,
            'cost': ml_campaign_cost,
            'roi': ml_roi
        },
        'improvement': {
            'roi_lift': roi_improvement,
            'incremental_revenue': incremental_value,
            'incremental_customers': ml_saved - baseline_saved
        },
        'threshold_analysis': analysis_results
    }
    
    return summary


if __name__ == '__main__':
    # Load predictions
    predictions_df = pd.read_csv('results/predictions.csv')
    test_df = predictions_df.drop(['actual_churn', 'predicted_churn', 'churn_probability'], axis=1)
    test_df['churn'] = predictions_df['actual_churn']
    y_pred_proba = predictions_df['churn_probability'].values
    
    calculate_campaign_roi(test_df, predictions_df, y_pred_proba)
