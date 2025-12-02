"""Base model interface for all ML models."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from pathlib import Path
import joblib
import numpy as np

from src.utils.data_types import ModelPrediction, Proposition, Outcome


class BaseModel(ABC):
    """Abstract base class for all betting prediction models."""
    
    def __init__(self, name: str, model_type: str):
        """Initialize the base model.
        
        Args:
            name: Human-readable model name
            model_type: Type of model (neural_net, gradient_boosting, etc.)
        """
        self.name = name
        self.model_type = model_type
        self.model = None
        self.is_trained = False
        self.performance_history: List[Dict[str, float]] = []
        self.feature_names: List[str] = []
    
    @abstractmethod
    def train(self, X: Any, y: Any, **kwargs) -> None:
        """Train the model on data.
        
        Args:
            X: Training features
            y: Training targets
            **kwargs: Additional training parameters
        """
        pass
    
    @abstractmethod
    def predict(self, X: Any) -> ModelPrediction:
        """Make a prediction on input data.
        
        Args:
            X: Input features
            
        Returns:
            ModelPrediction with prediction, confidence, and reasoning
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: Any) -> Dict[Outcome, float]:
        """Get probability distribution over outcomes.
        
        Args:
            X: Input features
            
        Returns:
            Dictionary mapping outcomes to probabilities
        """
        pass
    
    def predict_proposition(self, proposition: Proposition, features: Dict[str, Any]) -> ModelPrediction:
        """Make a prediction for a betting proposition.
        
        Args:
            proposition: The betting proposition
            features: Extracted features for the proposition
            
        Returns:
            ModelPrediction object
        """
        # Convert features to model input format
        X = self._prepare_features(features)
        
        # Get prediction
        prediction = self.predict(X)
        
        # Get probabilities
        probabilities = self.predict_proba(X)
        
        # Enhance prediction with probability info
        if prediction.prediction in probabilities:
            prediction.probability = probabilities[prediction.prediction]
        
        return prediction
    
    def _prepare_features(self, features: Dict[str, Any]) -> np.ndarray:
        """Prepare features for model input.
        
        Common implementation that converts feature dictionary to numpy array.
        Can be overridden by subclasses if different behavior is needed.
        
        Args:
            features: Raw feature dictionary
            
        Returns:
            NumPy array of feature values
        """
        feature_values = []
        for key in sorted(features.keys()):
            value = features[key]
            if isinstance(value, (int, float)):
                feature_values.append(float(value))
            elif isinstance(value, bool):
                feature_values.append(float(value))
        
        return np.array([feature_values])
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance scores if available.
        
        Returns:
            Dictionary mapping feature names to importance scores, or None
        """
        return None
    
    def save(self, path: str) -> None:
        """Save the model to disk.
        
        Args:
            path: Path to save the model
        """
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'name': self.name,
            'model_type': self.model_type,
            'is_trained': self.is_trained,
            'performance_history': self.performance_history,
            'feature_names': self.feature_names
        }
        
        joblib.dump(model_data, save_path)
    
    def load(self, path: str) -> None:
        """Load the model from disk.
        
        Args:
            path: Path to load the model from
        """
        model_data = joblib.load(path)
        
        self.model = model_data['model']
        self.name = model_data['name']
        self.model_type = model_data['model_type']
        self.is_trained = model_data['is_trained']
        self.performance_history = model_data.get('performance_history', [])
        self.feature_names = model_data.get('feature_names', [])
    
    def update_performance(self, metrics: Dict[str, float]) -> None:
        """Update performance history with new metrics.
        
        Args:
            metrics: Dictionary of performance metrics
        """
        self.performance_history.append(metrics)
    
    def get_recent_accuracy(self, n: int = 10) -> float:
        """Get average accuracy from recent predictions.
        
        Args:
            n: Number of recent predictions to consider
            
        Returns:
            Average accuracy
        """
        if not self.performance_history:
            return 0.5  # Default to 50% if no history
        
        recent = self.performance_history[-n:]
        accuracies = [m.get('accuracy', 0.5) for m in recent]
        return sum(accuracies) / len(accuracies)
    
    def __str__(self) -> str:
        """String representation of the model."""
        return f"{self.name} ({self.model_type})"
    
    def __repr__(self) -> str:
        """Detailed representation of the model."""
        return f"<{self.__class__.__name__} name='{self.name}' type='{self.model_type}' trained={self.is_trained}>"

