"""Statistical model using logistic regression."""

from sklearn.linear_model import LogisticRegression
import numpy as np
from typing import Dict, Any, Optional

from src.models.base_model import BaseModel
from src.utils.data_types import ModelPrediction, Outcome
from src.utils.logger import setup_logger


logger = setup_logger(__name__)


class StatisticalModel(BaseModel):
    """Conservative statistical predictor using logistic regression."""
    
    def __init__(
        self,
        name: str = "Statistical Conservative",
        regularization: float = 0.1,
        **kwargs
    ):
        """Initialize the statistical model.
        
        Args:
            name: Model name
            regularization: Regularization strength (C parameter)
            **kwargs: Additional sklearn parameters
        """
        super().__init__(name, "statistical")
        
        self.regularization = regularization
        
        self.model = LogisticRegression(
            C=regularization,
            max_iter=1000,
            random_state=42,
            **kwargs
        )
    
    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """Train the logistic regression model.
        
        Args:
            X: Training features
            y: Training targets
            **kwargs: Additional training parameters
        """
        logger.info(f"Training {self.name} on {len(X)} samples")
        
        self.model.fit(X, y)
        
        # Store feature names if available
        if hasattr(X, 'columns'):
            self.feature_names = list(X.columns)
        
        self.is_trained = True
        logger.info(f"{self.name} training completed")
    
    def predict(self, X: np.ndarray) -> ModelPrediction:
        """Make a prediction.
        
        Args:
            X: Input features
            
        Returns:
            ModelPrediction object
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Predict probability
        probs = self.model.predict_proba(X)[0]
        pred_class = int(self.model.predict(X)[0])
        
        # Determine outcome
        prediction = Outcome.HOME_WIN if pred_class == 1 else Outcome.AWAY_WIN
        confidence = float(probs[pred_class])
        
        # Get coefficients for interpretability
        key_features = self._get_significant_features(X)
        
        reasoning = f"Statistical analysis predicts {prediction.value} with {confidence:.1%} confidence. "
        reasoning += f"This prediction is based on proven statistical relationships and historical trends."
        
        return ModelPrediction(
            model_name=self.name,
            prediction=prediction,
            confidence=confidence,
            probability=float(probs[pred_class]),
            key_features=key_features,
            reasoning=reasoning
        )
    
    def predict_proba(self, X: np.ndarray) -> Dict[Outcome, float]:
        """Get probability distribution.
        
        Args:
            X: Input features
            
        Returns:
            Dictionary of outcome probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        probs = self.model.predict_proba(X)[0]
        
        return {
            Outcome.AWAY_WIN: float(probs[0]),
            Outcome.HOME_WIN: float(probs[1])
        }
    
    def _prepare_features(self, features: Dict[str, Any]) -> np.ndarray:
        """Prepare features for model input.
        
        Args:
            features: Feature dictionary
            
        Returns:
            NumPy array
        """
        feature_values = []
        for key in sorted(features.keys()):
            value = features[key]
            if isinstance(value, (int, float)):
                feature_values.append(float(value))
            elif isinstance(value, bool):
                feature_values.append(float(value))
        
        return np.array([feature_values])
    
    def _get_significant_features(self, X: np.ndarray) -> Dict[str, Any]:
        """Get significant features based on coefficients.
        
        Args:
            X: Input features
            
        Returns:
            Dictionary of significant features
        """
        if not self.feature_names:
            return {}
        
        # Get coefficients
        coef = self.model.coef_[0]
        
        # Find top absolute coefficients
        abs_coef = np.abs(coef)
        top_indices = np.argsort(abs_coef)[-5:]  # Top 5
        
        significant = {}
        for idx in top_indices:
            if idx < len(self.feature_names):
                feature_name = self.feature_names[idx]
                significant[feature_name] = float(coef[idx])
        
        return significant
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance based on coefficient magnitudes.
        
        Returns:
            Dictionary of feature importances
        """
        if not self.is_trained or not self.feature_names:
            return None
        
        coef = np.abs(self.model.coef_[0])
        
        # Normalize
        total = np.sum(coef)
        if total > 0:
            coef = coef / total
        
        return dict(zip(self.feature_names, coef))

