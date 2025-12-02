"""Tests for base model functionality."""

import pytest
import numpy as np
from src.models.base_model import BaseModel
from src.utils.data_types import ModelPrediction, Outcome


class DummyModel(BaseModel):
    """Dummy model for testing."""
    
    def train(self, X, y, **kwargs):
        self.is_trained = True
    
    def predict(self, X):
        return ModelPrediction(
            model_name=self.name,
            prediction=Outcome.HOME_WIN,
            confidence=0.75,
            reasoning="Test prediction"
        )
    
    def predict_proba(self, X):
        return {
            Outcome.HOME_WIN: 0.75,
            Outcome.AWAY_WIN: 0.25
        }
    
    def _prepare_features(self, features):
        return np.array([[1.0, 2.0, 3.0]])


def test_base_model_initialization():
    """Test base model initialization."""
    model = DummyModel("Test Model", "test")
    
    assert model.name == "Test Model"
    assert model.model_type == "test"
    assert not model.is_trained
    assert len(model.performance_history) == 0


def test_model_training():
    """Test model training."""
    model = DummyModel("Test Model", "test")
    
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 2, 100)
    
    model.train(X, y)
    
    assert model.is_trained


def test_model_prediction():
    """Test model prediction."""
    model = DummyModel("Test Model", "test")
    model.train(np.random.rand(10, 5), np.random.randint(0, 2, 10))
    
    X = np.random.rand(1, 5)
    prediction = model.predict(X)
    
    assert isinstance(prediction, ModelPrediction)
    assert prediction.model_name == "Test Model"
    assert isinstance(prediction.prediction, Outcome)
    assert 0 <= prediction.confidence <= 1


def test_performance_tracking():
    """Test performance history tracking."""
    model = DummyModel("Test Model", "test")
    
    model.update_performance({'accuracy': 0.8})
    model.update_performance({'accuracy': 0.85})
    
    assert len(model.performance_history) == 2
    assert model.get_recent_accuracy(2) == 0.825


if __name__ == "__main__":
    pytest.main([__file__])

