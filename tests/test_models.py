"""Tests for ML models."""

import pytest
import numpy as np

from src.models.traditional.gradient_boost_model import GradientBoostModel
from src.models.traditional.random_forest_model import RandomForestModel
from src.models.traditional.statistical_model import StatisticalModel
from src.utils.data_types import Outcome


class TestModels:
    """Test suite for ML models."""
    
    @pytest.fixture
    def sample_training_data(self):
        """Create sample training data."""
        np.random.seed(42)
        X_train = np.random.rand(100, 20)
        y_train = np.random.randint(0, 2, 100)
        X_test = np.random.rand(20, 20)
        y_test = np.random.randint(0, 2, 20)
        return X_train, y_train, X_test, y_test
    
    def test_gradient_boost_model(self, sample_training_data):
        """Test GradientBoostModel training and prediction."""
        X_train, y_train, X_test, y_test = sample_training_data
        
        model = GradientBoostModel(name="Test Gradient", n_estimators=10)
        
        # Test training
        model.train(X_train, y_train)
        assert model.is_trained == True
        
        # Test prediction
        prediction = model.predict(X_test[0:1])
        assert prediction.model_name == "Test Gradient"
        assert prediction.prediction in [Outcome.HOME_WIN, Outcome.AWAY_WIN]
        assert 0.0 <= prediction.confidence <= 1.0
        
        # Test probability prediction
        probs = model.predict_proba(X_test[0:1])
        assert Outcome.HOME_WIN in probs
        assert Outcome.AWAY_WIN in probs
        assert abs(sum(probs.values()) - 1.0) < 0.01
    
    def test_random_forest_model(self, sample_training_data):
        """Test RandomForestModel training and prediction."""
        X_train, y_train, X_test, y_test = sample_training_data
        
        model = RandomForestModel(name="Test Forest", n_estimators=10)
        
        # Test training
        model.train(X_train, y_train)
        assert model.is_trained == True
        
        # Test prediction
        prediction = model.predict(X_test[0:1])
        assert prediction.model_name == "Test Forest"
        assert prediction.prediction in [Outcome.HOME_WIN, Outcome.AWAY_WIN]
        assert 0.0 <= prediction.confidence <= 1.0
        
        # Test that confidence includes uncertainty adjustment
        assert prediction.reasoning is not None
        assert "uncertainty" in prediction.reasoning.lower()
    
    def test_statistical_model(self, sample_training_data):
        """Test StatisticalModel training and prediction."""
        X_train, y_train, X_test, y_test = sample_training_data
        
        model = StatisticalModel(name="Test Stats")
        
        # Test training
        model.train(X_train, y_train)
        assert model.is_trained == True
        
        # Test prediction
        prediction = model.predict(X_test[0:1])
        assert prediction.model_name == "Test Stats"
        assert prediction.prediction in [Outcome.HOME_WIN, Outcome.AWAY_WIN]
        assert 0.0 <= prediction.confidence <= 1.0
        
        # Test feature importance
        importance = model.get_feature_importance()
        assert importance is None or isinstance(importance, dict)
    
    def test_model_save_load(self, sample_training_data, tmp_path):
        """Test model save and load functionality."""
        X_train, y_train, X_test, y_test = sample_training_data
        
        model = GradientBoostModel(name="Test Save")
        model.train(X_train, y_train)
        
        # Save model
        save_path = tmp_path / "test_model.pkl"
        model.save(str(save_path))
        assert save_path.exists()
        
        # Load model
        new_model = GradientBoostModel(name="Test Load")
        new_model.load(str(save_path))
        
        assert new_model.is_trained == True
        assert new_model.name == "Test Save"
        
        # Test that loaded model can predict
        prediction = new_model.predict(X_test[0:1])
        assert prediction is not None
    
    def test_model_performance_tracking(self):
        """Test model performance tracking."""
        model = StatisticalModel(name="Test Performance")
        
        # Initially no history
        assert len(model.performance_history) == 0
        
        # Add performance metrics
        model.update_performance({'accuracy': 0.65, 'precision': 0.70})
        assert len(model.performance_history) == 1
        
        model.update_performance({'accuracy': 0.68, 'precision': 0.72})
        assert len(model.performance_history) == 2
        
        # Test recent accuracy calculation
        recent_acc = model.get_recent_accuracy(n=2)
        assert 0.0 <= recent_acc <= 1.0

