"""Tests for feature alignment and preprocessor consistency."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil

from src.models.base_model import BaseModel
from src.models.traditional.statistical_model import StatisticalModel
from src.data.processors.data_preprocessor import DataPreprocessor
from src.utils.data_types import Outcome


class TestFeatureAlignment:
    """Test suite for feature alignment between training and prediction."""
    
    @pytest.fixture
    def sample_features_dict(self):
        """Create sample feature dictionary."""
        return {
            'feature_a': 10.0,
            'feature_b': 20.0,
            'feature_c': 30.0,
            'feature_d': 40.0,
            'feature_e': 50.0
        }
    
    @pytest.fixture
    def sample_training_data(self):
        """Create sample training data with known feature order."""
        np.random.seed(42)
        # Create DataFrame with specific feature order
        n_samples = 100
        data = {
            'feature_a': np.random.rand(n_samples) * 10,
            'feature_b': np.random.rand(n_samples) * 20,
            'feature_c': np.random.rand(n_samples) * 30,
            'feature_d': np.random.rand(n_samples) * 40,
            'feature_e': np.random.rand(n_samples) * 50
        }
        df = pd.DataFrame(data)
        y = np.random.randint(0, 2, n_samples)
        return df, y
    
    @pytest.fixture
    def preprocessor(self, sample_training_data):
        """Create and fit a preprocessor."""
        df, _ = sample_training_data
        preprocessor = DataPreprocessor(scaler_type='standard')
        preprocessor.fit(df)
        return preprocessor
    
    def test_preprocessor_feature_order(self, preprocessor, sample_training_data):
        """Test that preprocessor maintains feature order."""
        df, _ = sample_training_data
        feature_names = preprocessor.get_feature_names()
        
        # Feature names should match DataFrame columns
        assert len(feature_names) == len(df.columns)
        assert set(feature_names) == set(df.columns)
    
    def test_model_with_preprocessor(self, preprocessor, sample_training_data, sample_features_dict):
        """Test that model uses preprocessor for feature preparation."""
        df, y = sample_training_data
        
        # Create model with preprocessor
        model = StatisticalModel(name="Test Model")
        model.set_preprocessor(preprocessor)
        model.feature_names = preprocessor.get_feature_names()
        
        # Train model
        X_train = preprocessor.transform(df).values
        model.train(X_train, y)
        
        # Test feature preparation uses preprocessor
        X_prepared = model._prepare_features(sample_features_dict)
        
        # Should be transformed (scaled) by preprocessor
        assert X_prepared.shape[1] == len(preprocessor.get_feature_names())
        # Values should be scaled (not raw values)
        assert not np.array_equal(X_prepared[0], list(sample_features_dict.values()))
    
    def test_model_without_preprocessor_fallback(self, sample_features_dict):
        """Test that model falls back correctly when no preprocessor."""
        model = StatisticalModel(name="Test Model")
        # No preprocessor set
        
        # Should use alphabetical order fallback
        X_prepared = model._prepare_features(sample_features_dict)
        
        # Should have correct number of features
        assert X_prepared.shape[1] == len(sample_features_dict)
        # Should be in alphabetical order
        expected_order = sorted(sample_features_dict.keys())
        # Values should match (no scaling)
        prepared_values = X_prepared[0].tolist()
        expected_values = [sample_features_dict[k] for k in expected_order]
        assert np.allclose(prepared_values, expected_values)
    
    def test_feature_order_consistency(self, preprocessor, sample_training_data):
        """Test that feature order is consistent between training and prediction."""
        df, y = sample_training_data
        
        model = StatisticalModel(name="Test Model")
        model.set_preprocessor(preprocessor)
        model.feature_names = preprocessor.get_feature_names()
        
        # Train
        X_train = preprocessor.transform(df).values
        model.train(X_train, y)
        
        # Create prediction features (may be in different order)
        prediction_features = {
            'feature_e': 50.0,  # Different order
            'feature_a': 10.0,
            'feature_d': 40.0,
            'feature_b': 20.0,
            'feature_c': 30.0
        }
        
        # Prepare features should use preprocessor order, not dict order
        X_prepared = model._prepare_features(prediction_features)
        
        # Should have correct shape
        assert X_prepared.shape[1] == len(preprocessor.get_feature_names())
        
        # Should be able to make prediction
        prediction = model.predict(X_prepared)
        assert prediction.prediction in [Outcome.HOME_WIN, Outcome.AWAY_WIN]
    
    def test_missing_features_handling(self, preprocessor, sample_training_data):
        """Test that missing features are handled correctly."""
        df, y = sample_training_data
        
        model = StatisticalModel(name="Test Model")
        model.set_preprocessor(preprocessor)
        model.feature_names = preprocessor.get_feature_names()
        
        # Train
        X_train = preprocessor.transform(df).values
        model.train(X_train, y)
        
        # Create features with missing values
        incomplete_features = {
            'feature_a': 10.0,
            'feature_b': 20.0,
            # Missing feature_c, feature_d, feature_e
        }
        
        # Should handle missing features (preprocessor will use defaults)
        X_prepared = model._prepare_features(incomplete_features)
        
        # Should still have correct shape
        assert X_prepared.shape[1] == len(preprocessor.get_feature_names())
    
    def test_preprocessor_save_load(self, preprocessor, tmp_path):
        """Test that preprocessor can be saved and loaded."""
        import joblib
        
        # Save preprocessor
        save_path = tmp_path / "preprocessor.pkl"
        joblib.dump(preprocessor, str(save_path))
        
        # Load preprocessor
        loaded_preprocessor = joblib.load(str(save_path))
        
        # Should have same feature names
        assert loaded_preprocessor.get_feature_names() == preprocessor.get_feature_names()
        
        # Should transform identically
        test_df = pd.DataFrame({
            'feature_a': [10.0],
            'feature_b': [20.0],
            'feature_c': [30.0],
            'feature_d': [40.0],
            'feature_e': [50.0]
        })
        
        original_transformed = preprocessor.transform(test_df)
        loaded_transformed = loaded_preprocessor.transform(test_df)
        
        assert np.allclose(original_transformed.values, loaded_transformed.values)
    
    def test_model_save_load_with_preprocessor(self, preprocessor, sample_training_data, tmp_path):
        """Test that model saves and loads preprocessor correctly."""
        df, y = sample_training_data
        
        # Create and train model with preprocessor
        model = StatisticalModel(name="Test Model")
        model.set_preprocessor(preprocessor)
        model.feature_names = preprocessor.get_feature_names()
        
        X_train = preprocessor.transform(df).values
        model.train(X_train, y)
        
        # Save model
        model_path = tmp_path / "model.pkl"
        model.save(str(model_path))
        
        # Load model
        loaded_model = StatisticalModel(name="Test Model")
        loaded_model.load(str(model_path))
        
        # Should have preprocessor
        assert loaded_model.preprocessor is not None
        assert loaded_model.preprocessor.get_feature_names() == preprocessor.get_feature_names()
        
        # Should make same predictions
        test_features = {
            'feature_a': 10.0,
            'feature_b': 20.0,
            'feature_c': 30.0,
            'feature_d': 40.0,
            'feature_e': 50.0
        }
        
        original_pred = model.predict(model._prepare_features(test_features))
        loaded_pred = loaded_model.predict(loaded_model._prepare_features(test_features))
        
        # Predictions should match (same model, same features)
        assert original_pred.prediction == loaded_pred.prediction
        assert abs(original_pred.confidence - loaded_pred.confidence) < 0.01

