"""Random Forest model for betting predictions."""

from sklearn.ensemble import RandomForestClassifier
import numpy as np
from typing import Dict, Any, Optional

from src.models.base_model import BaseModel
from src.utils.data_types import ModelPrediction, Outcome
from src.utils.logger import setup_logger


logger = setup_logger(__name__)


class RandomForestModel(BaseModel):
    """Random Forest predictor."""
    
    def __init__(
        self,
        name: str = "Forest Evaluator",
        n_estimators: int = 150,
        max_depth: int = 10,
        min_samples_split: int = 5,
        **kwargs
    ):
        """Initialize the random forest model.
        
        Args:
            name: Model name
            n_estimators: Number of trees
            max_depth: Maximum tree depth
            min_samples_split: Minimum samples to split
            **kwargs: Additional sklearn parameters
        """
        super().__init__(name, "random_forest")
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42,
            **kwargs
        )
    
    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """Train the random forest model.
        
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
        
        # Get uncertainty estimate (entropy)
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        uncertainty = float(entropy / np.log(len(probs)))
        
        # Determine outcome
        prediction = Outcome.HOME_WIN if pred_class == 1 else Outcome.AWAY_WIN
        confidence = float(probs[pred_class]) * (1 - uncertainty)  # Adjust by uncertainty
        
        # Get feature importance
        key_features = self.get_feature_importance()
        if key_features:
            top_features = dict(sorted(key_features.items(), key=lambda x: x[1], reverse=True)[:5])
        else:
            top_features = {}
        
        reasoning = f"Random Forest predicts {prediction.value} with {confidence:.1%} confidence "
        reasoning += f"(uncertainty: {uncertainty:.2f}). "
        if top_features:
            reasoning += f"Key factors: {', '.join(top_features.keys())}"
        
        return ModelPrediction(
            model_name=self.name,
            prediction=prediction,
            confidence=confidence,
            probability=float(probs[pred_class]),
            key_features=top_features,
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
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance scores.
        
        Returns:
            Dictionary of feature importances
        """
        if not self.is_trained or not self.feature_names:
            return None
        
        importances = self.model.feature_importances_
        return dict(zip(self.feature_names, importances))

