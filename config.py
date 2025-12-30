# Configuration Parameters

# Data generation
RANDOM_STATE = 42
NUM_CUSTOMERS = 10000

# Train-test split
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2

# Model hyperparameters
MODEL_MAX_DEPTH = 10
MODEL_N_ESTIMATORS = 150
MODEL_MIN_SAMPLES_SPLIT = 8
MODEL_MIN_SAMPLES_LEAF = 4

# Business parameters
BASELINE_RANDOM_ROI = 1.65  # 165%
AVERAGE_CUSTOMER_ANNUAL_VALUE = 5000
CAMPAIGN_COST_PER_CUSTOMER = 50
TARGET_ROI = 4.0  # 400%
SAVE_RATE = 0.20  # Expected save rate for targeted retention campaign

# High-value customer threshold
HIGH_VALUE_THRESHOLD = 1500  # monthly spend in dollars

# API parameters
API_HOST = '0.0.0.0'
API_PORT = 5000
API_DEBUG = False
