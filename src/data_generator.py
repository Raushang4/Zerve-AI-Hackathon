"""Data generation module for synthetic SaaS customer data."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from config import RANDOM_STATE, NUM_CUSTOMERS


def generate_customer_data(n_customers=NUM_CUSTOMERS, random_state=RANDOM_STATE):
    """
    Generate synthetic SaaS customer data with clear churn patterns.
    
    Features:
    - monthly_spend: Customer's monthly spend (affects churn inversely)
    - contract_length_months: Length of contract (longer = lower churn)
    - support_tickets: Number of support tickets (high = satisfaction, lower churn)
    - feature_usage_score: Usage intensity 0-1 (higher = lower churn)
    - days_since_last_login: Engagement metric (higher = higher churn risk)
    - price_sensitivity_score: 0-1 (higher = more price-conscious, higher churn risk)
    - competitor_engagement: 0-1 (higher = higher churn risk)
    - churn: Binary target (1 = churned, 0 = retained)
    """
    np.random.seed(random_state)
    
    n = n_customers
    
    # Generate features with realistic distributions
    monthly_spend = np.random.lognormal(mean=6.5, sigma=0.8, size=n)
    contract_length = np.random.choice([3, 6, 12, 24, 36], size=n, p=[0.1, 0.2, 0.4, 0.2, 0.1])
    support_tickets = np.random.poisson(lam=3, size=n)
    feature_usage = np.random.beta(a=2, b=2, size=n)
    days_since_login = np.random.exponential(scale=20, size=n).astype(int)
    price_sensitivity = np.random.beta(a=2, b=3, size=n)
    competitor_engagement = np.random.beta(a=2, b=5, size=n)
    
    # Generate churn with clear, learnable patterns
    churn_probability = np.zeros(n)
    
    # High churn risk: days_since_login (engagement is THE strongest signal)
    churn_probability += (days_since_login / 60.0) * 0.4  # Engagement decay
    
    # High churn risk: price sensitivity + competitor engagement (price/competitive pressure)
    churn_probability += (price_sensitivity + competitor_engagement) / 2 * 0.25
    
    # Low usage = high churn risk
    churn_probability += (1 - feature_usage) * 0.20
    
    # Protective factors: higher spend, longer contracts, more support
    churn_probability -= (monthly_spend / monthly_spend.max()) * 0.10
    churn_probability -= (contract_length / 36.0) * 0.08
    churn_probability -= (support_tickets / 10.0) * 0.07
    
    # Clip to [0, 1] and add small noise
    churn_probability = np.clip(churn_probability, 0, 0.95)
    churn_probability = np.clip(churn_probability + np.random.normal(0, 0.05, n), 0, 0.95)
    
    churn = (np.random.uniform(0, 1, n) < churn_probability).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({
        'customer_id': [f'C{i:06d}' for i in range(n)],
        'monthly_spend': np.round(monthly_spend, 2),
        'contract_length_months': contract_length,
        'support_tickets': support_tickets,
        'feature_usage_score': np.round(feature_usage, 3),
        'days_since_last_login': days_since_login,
        'price_sensitivity_score': np.round(price_sensitivity, 3),
        'competitor_engagement': np.round(competitor_engagement, 3),
        'churn': churn
    })
    
    return df


if __name__ == '__main__':
    # Generate and save data
    print("Generating synthetic customer data...")
    data = generate_customer_data()
    
    # Create data directory if it doesn't exist
    import os
    os.makedirs('data', exist_ok=True)
    
    # Save full dataset
    data.to_csv('data/generated_customers.csv', index=False)
    print(f"✓ Generated {len(data)} customers and saved to data/generated_customers.csv")
    
    # Print summary statistics
    print("\nDataset Summary:")
    print(f"Total customers: {len(data)}")
    print(f"Churn rate: {data['churn'].mean():.1%}")
    print("\nFeature statistics:")
    print(data.describe())
