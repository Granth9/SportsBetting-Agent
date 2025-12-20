"""Test XGBoost and LightGBM with small dataset to verify fixes work."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.traditional.gradient_boost_model import GradientBoostModel
from src.models.traditional.lightgbm_model import LightGBMModel
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def generate_test_data(n_samples=100, n_features=10):
    """Generate synthetic test data.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        
    Returns:
        Tuple of (X, y)
    """
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = (np.random.rand(n_samples) > 0.5).astype(np.int32)
    return X, y


def test_xgboost():
    """Test XGBoost model with small dataset."""
    logger.info("=" * 60)
    logger.info("Testing XGBoost (Gradient Strategist)")
    logger.info("=" * 60)
    
    try:
        # Generate test data
        X_train, y_train = generate_test_data(100, 10)
        X_val, y_val = generate_test_data(20, 10)
        
        logger.info(f"Generated test data: {X_train.shape} train, {X_val.shape} val")
        
        # Create model
        model = GradientBoostModel(
            name="Gradient Strategist Test",
            n_estimators=10,  # Small number for quick test
            max_depth=3,
            learning_rate=0.1
        )
        
        # Test training
        logger.info("Testing training...")
        model.train(X_train, y_train, X_val, y_val)
        
        if model.is_trained:
            logger.info("✅ XGBoost training succeeded!")
            
            # Test prediction
            logger.info("Testing prediction...")
            X_test = np.random.randn(1, 10).astype(np.float32)
            prediction = model.predict(X_test)
            logger.info(f"✅ Prediction: {prediction.prediction.value}, confidence: {prediction.confidence:.2%}")
            
            # Test predict_proba
            proba = model.predict_proba(X_test)
            logger.info(f"✅ Probabilities: {proba}")
            
            return True
        else:
            logger.error("❌ XGBoost training failed - model not marked as trained")
            return False
            
    except Exception as e:
        logger.error(f"❌ XGBoost test failed: {e}", exc_info=True)
        return False


def test_lightgbm():
    """Test LightGBM model with small dataset."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing LightGBM (LightGBM Optimizer)")
    logger.info("=" * 60)
    
    try:
        # Generate test data
        X_train, y_train = generate_test_data(100, 10)
        X_val, y_val = generate_test_data(20, 10)
        
        logger.info(f"Generated test data: {X_train.shape} train, {X_val.shape} val")
        
        # Create model
        model = LightGBMModel(
            name="LightGBM Optimizer Test",
            n_estimators=10,  # Small number for quick test
            max_depth=3,
            learning_rate=0.1
        )
        
        # Test training
        logger.info("Testing training...")
        model.train(X_train, y_train, X_val, y_val)
        
        if model.is_trained:
            logger.info("✅ LightGBM training succeeded!")
            
            # Test prediction
            logger.info("Testing prediction...")
            X_test = np.random.randn(1, 10).astype(np.float32)
            prediction = model.predict(X_test)
            logger.info(f"✅ Prediction: {prediction.prediction.value}, confidence: {prediction.confidence:.2%}")
            
            # Test predict_proba
            proba = model.predict_proba(X_test)
            logger.info(f"✅ Probabilities: {proba}")
            
            return True
        else:
            logger.error("❌ LightGBM training failed - model not marked as trained")
            return False
            
    except Exception as e:
        logger.error(f"❌ LightGBM test failed: {e}", exc_info=True)
        return False


def test_with_sample_weights():
    """Test both models with sample weights."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing with sample weights")
    logger.info("=" * 60)
    
    X_train, y_train = generate_test_data(100, 10)
    X_val, y_val = generate_test_data(20, 10)
    sample_weight = np.random.rand(100).astype(np.float32)
    
    results = {}
    
    # Test XGBoost with sample weights
    try:
        model = GradientBoostModel(name="XGBoost Weighted Test", n_estimators=10, max_depth=3)
        model.train(X_train, y_train, X_val, y_val, sample_weight=sample_weight)
        results['xgboost'] = model.is_trained
        logger.info(f"✅ XGBoost with sample weights: {'SUCCESS' if model.is_trained else 'FAILED'}")
    except Exception as e:
        results['xgboost'] = False
        logger.error(f"❌ XGBoost with sample weights failed: {e}")
    
    # Test LightGBM with sample weights
    try:
        model = LightGBMModel(name="LightGBM Weighted Test", n_estimators=10, max_depth=3)
        model.train(X_train, y_train, X_val, y_val, sample_weight=sample_weight)
        results['lightgbm'] = model.is_trained
        logger.info(f"✅ LightGBM with sample weights: {'SUCCESS' if model.is_trained else 'FAILED'}")
    except Exception as e:
        results['lightgbm'] = False
        logger.error(f"❌ LightGBM with sample weights failed: {e}")
    
    return results


if __name__ == "__main__":
    logger.info("Starting XGBoost and LightGBM stability tests...\n")
    
    # Test 1: Basic training and prediction
    xgboost_ok = test_xgboost()
    lightgbm_ok = test_lightgbm()
    
    # Test 2: With sample weights
    weighted_results = test_with_sample_weights()
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"XGBoost basic test: {'✅ PASS' if xgboost_ok else '❌ FAIL'}")
    logger.info(f"LightGBM basic test: {'✅ PASS' if lightgbm_ok else '❌ FAIL'}")
    logger.info(f"XGBoost with weights: {'✅ PASS' if weighted_results.get('xgboost', False) else '❌ FAIL'}")
    logger.info(f"LightGBM with weights: {'✅ PASS' if weighted_results.get('lightgbm', False) else '❌ FAIL'}")
    
    if xgboost_ok and lightgbm_ok:
        logger.info("\n✅ All tests passed! Models are ready for full training.")
        sys.exit(0)
    else:
        logger.error("\n❌ Some tests failed. Review errors above.")
        sys.exit(1)

