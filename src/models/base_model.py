"""Base model interface for all ML models."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from pathlib import Path
import joblib
import numpy as np

from src.utils.data_types import ModelPrediction, Proposition, Outcome
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# Try to import calibration, but make it optional
try:
    from sklearn.calibration import CalibratedClassifierCV
    CALIBRATION_AVAILABLE = True
except ImportError:
    CALIBRATION_AVAILABLE = False
    logger.warning("sklearn.calibration not available. Probability calibration will be disabled.")


class BaseModel(ABC):
    """Abstract base class for all betting prediction models."""
    
    def __init__(self, name: str, model_type: str, preprocessor: Optional[Any] = None):
        """Initialize the base model.
        
        Args:
            name: Human-readable model name
            model_type: Type of model (neural_net, gradient_boosting, etc.)
            preprocessor: Optional DataPreprocessor instance for feature transformation
        """
        self.name = name
        self.model_type = model_type
        self.model = None
        self.is_trained = False
        self.performance_history: List[Dict[str, float]] = []
        self.feature_names: List[str] = []
        self.preprocessor = preprocessor
        self.calibrator = None  # Probability calibrator (Platt scaling)
        self.use_calibration = False  # Whether to use calibration
    
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
        
        # Get probabilities (calibrated if available)
        probabilities = self.predict_proba(X)
        
        # Apply calibration if available
        if self.use_calibration and self.calibrator is not None:
            probabilities = self._apply_calibration(X, probabilities)
        
        # Enhance prediction with probability info
        if prediction.prediction in probabilities:
            prediction.probability = probabilities[prediction.prediction]
            # Update confidence with calibrated probability
            prediction.confidence = probabilities[prediction.prediction]
        
        return prediction
    
    def _apply_calibration(self, X: np.ndarray, probabilities: Dict[Outcome, float]) -> Dict[Outcome, float]:
        """Apply probability calibration to raw probabilities.
        
        Args:
            X: Input features
            probabilities: Raw probability dictionary
            
        Returns:
            Calibrated probability dictionary
        """
        if not CALIBRATION_AVAILABLE or self.calibrator is None:
            return probabilities
        
        try:
            # Get raw probabilities as array [P(AWAY_WIN), P(HOME_WIN)]
            raw_probs = np.array([[probabilities.get(Outcome.AWAY_WIN, 0.0), 
                                  probabilities.get(Outcome.HOME_WIN, 0.0)]])
            
            # Apply calibration
            calibrated_probs = self.calibrator.predict_proba(raw_probs)[0]
            
            return {
                Outcome.AWAY_WIN: float(calibrated_probs[0]),
                Outcome.HOME_WIN: float(calibrated_probs[1])
            }
        except Exception as e:
            logger.warning(f"Error applying calibration for {self.name}: {e}")
            return probabilities
    
    def fit_calibration(self, X_val: np.ndarray, y_val: np.ndarray) -> None:
        """Fit probability calibrator on validation data.
        
        Args:
            X_val: Validation features
            y_val: Validation labels
        """
        if not CALIBRATION_AVAILABLE:
            logger.warning(f"Calibration not available for {self.name}")
            return
        
        if not self.is_trained:
            logger.warning(f"Cannot calibrate {self.name}: model not trained")
            return
        
        try:
            # Get raw probabilities from model
            raw_probs = []
            for i in range(len(X_val)):
                probs = self.predict_proba(X_val[i:i+1])
                raw_probs.append([probs.get(Outcome.AWAY_WIN, 0.0), probs.get(Outcome.HOME_WIN, 0.0)])
            
            raw_probs = np.array(raw_probs)
            
            # Fit Platt scaling (sigmoid calibration)
            from sklearn.calibration import CalibratedClassifierCV
            from sklearn.linear_model import LogisticRegression
            
            # Use a simple logistic regression as the calibrator
            calibrator = LogisticRegression()
            calibrator.fit(raw_probs, y_val)
            
            self.calibrator = calibrator
            self.use_calibration = True
            logger.info(f"Fitted probability calibrator for {self.name}")
            
        except Exception as e:
            logger.warning(f"Error fitting calibration for {self.name}: {e}")
            self.use_calibration = False
    
    def _prepare_features(self, features: Dict[str, Any]) -> np.ndarray:
        """Prepare features for model input.
        
        Uses preprocessor if available to ensure feature order and scaling match training.
        Falls back to alphabetical sorting if no preprocessor is available.
        
        Args:
            features: Raw feature dictionary
            
        Returns:
            NumPy array of feature values (scaled and ordered correctly)
        """
        # If preprocessor is available, use it for proper feature transformation
        if self.preprocessor is not None:
            try:
                # Use preprocessor's prepare_features_dict method which handles:
                # - Feature ordering (matches training order)
                # - Scaling/normalization (matches training scaling)
                # - Missing value handling
                X = self.preprocessor.prepare_features_dict(features)
                return X
            except Exception as e:
                logger.warning(f"Error using preprocessor for {self.name}, falling back to manual preparation: {e}")
                # Fall through to manual preparation
        
        # Fallback: Manual feature preparation (alphabetical order)
        # This maintains backward compatibility but may cause issues if features
        # were trained with different order/scaling
        if self.feature_names:
            # Use stored feature names if available to maintain order
            feature_values = []
            for feature_name in self.feature_names:
                if feature_name in features:
                    value = features[feature_name]
                    if isinstance(value, (int, float)):
                        feature_values.append(float(value))
                    elif isinstance(value, bool):
                        feature_values.append(float(value))
                    else:
                        feature_values.append(0.0)  # Default for non-numeric
                else:
                    feature_values.append(0.0)  # Default for missing features
                    logger.warning(f"Feature '{feature_name}' missing from input, using default 0.0")
        else:
            # No feature names stored, use alphabetical order (legacy behavior)
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
            'feature_names': self.feature_names,
            'preprocessor': self.preprocessor,  # Save preprocessor reference
            'calibrator': self.calibrator,  # Save calibrator
            'use_calibration': self.use_calibration
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
        self.preprocessor = model_data.get('preprocessor', None)  # Load preprocessor if available
        self.calibrator = model_data.get('calibrator', None)  # Load calibrator if available
        self.use_calibration = model_data.get('use_calibration', False)
        
        if self.preprocessor is None:
            logger.warning(f"Model {self.name} loaded without preprocessor. Feature scaling may be incorrect.")
        
        if self.use_calibration and self.calibrator is None:
            logger.warning(f"Model {self.name} expects calibration but calibrator not found.")
            self.use_calibration = False
    
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
    
    def set_preprocessor(self, preprocessor: Any) -> None:
        """Set the preprocessor for this model.
        
        Args:
            preprocessor: DataPreprocessor instance
        """
        self.preprocessor = preprocessor
        # Update feature names from preprocessor if available
        if hasattr(preprocessor, 'get_feature_names'):
            self.feature_names = preprocessor.get_feature_names()
            logger.info(f"Set preprocessor for {self.name} with {len(self.feature_names)} features")
    
    def __str__(self) -> str:
        """String representation of the model."""
        return f"{self.name} ({self.model_type})"
    
    def __repr__(self) -> str:
        """Detailed representation of the model."""
        return f"<{self.__class__.__name__} name='{self.name}' type='{self.model_type}' trained={self.is_trained}>"

