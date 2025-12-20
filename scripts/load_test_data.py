"""Helper script to load test data for analysis."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from scripts.train_models import prepare_training_data

def get_test_data(seasons, include_2025_week15=True):
    """Get test data from training preparation.
    
    Returns:
        Tuple of (X_test, y_test, test_features_df, feature_names)
    """
    X_train, y_train, X_val, y_val, X_test, y_test, feature_names, preprocessor, sample_weights = prepare_training_data(
        seasons,
        include_2025_week15=include_2025_week15,
        temporal_weighting=False
    )
    
    # Create DataFrame with feature names
    test_features_df = pd.DataFrame(X_test, columns=feature_names)
    
    return X_test, y_test, test_features_df, feature_names

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seasons', nargs='+', type=int, required=True)
    parser.add_argument('--include-2025-week15', action='store_true', default=True)
    args = parser.parse_args()
    
    X_test, y_test, test_features_df, feature_names = get_test_data(args.seasons, args.include_2025_week15)
    print(f"Test data shape: {X_test.shape}")
    print(f"Test labels shape: {y_test.shape}")
    print(f"Feature names: {len(feature_names)}")

