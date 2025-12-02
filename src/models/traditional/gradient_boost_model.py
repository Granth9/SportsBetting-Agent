"""Gradient boosting model using XGBoost."""

import xgboost as xgb
import numpy as np
from typing import Dict, Any, Optional

from src.models.base_model import BaseModel
from src.utils.data_types import ModelPrediction, Outcome
from src.utils.logger import setup_logger


logger = setup_logger(__name__)


class GradientBoostModel(BaseModel):
    """Gradient boosting predictor using XGBoost."""
    
    def __init__(
        self,
        name: str = "Gradient Strategist",
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        **kwargs
    ):
        """Initialize the gradient boosting model.
        
        Args:
            name: Model name
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate
            **kwargs: Additional XGBoost parameters
        """
        super().__init__(name, "gradient_boosting")
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.params = {
            'objective': 'binary:logistic',
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'n_estimators': n_estimators,
            'eval_metric': 'logloss',
            **kwargs
        }
    
    def train(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None) -> None:
        """Train the gradient boosting model.
        
        Args:
            X: Training features
            y: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
        """
        logger.info(f"Training {self.name} on {len(X)} samples")
        
        # Convert to DMatrix
        dtrain = xgb.DMatrix(X, label=y)
        
        # Add validation set if provided
        evals = [(dtrain, 'train')]
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            evals.append((dval, 'val'))
        
        # Train model
        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=self.n_estimators,
            evals=evals,
            verbose_eval=False
        )
        
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
        dtest = xgb.DMatrix(X)
        prob = self.model.predict(dtest)[0]
        
        # Determine outcome
        prediction = Outcome.HOME_WIN if prob > 0.5 else Outcome.AWAY_WIN
        confidence = float(prob if prob > 0.5 else 1 - prob)
        
        # Get feature importance
        key_features = self.get_feature_importance()
        top_features = dict(sorted(key_features.items(), key=lambda x: x[1], reverse=True)[:5])
        
        reasoning = f"Gradient boosting model predicts {prediction.value} with {confidence:.1%} confidence. "
        reasoning += f"Top features: {', '.join(top_features.keys())}"
        
        return ModelPrediction(
            model_name=self.name,
            prediction=prediction,
            confidence=confidence,
            probability=prob if prediction == Outcome.HOME_WIN else 1 - prob,
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
        
        dtest = xgb.DMatrix(X)
        prob_home = float(self.model.predict(dtest)[0])
        
        return {
            Outcome.HOME_WIN: prob_home,
            Outcome.AWAY_WIN: 1 - prob_home
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
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance scores.
        
        Returns:
            Dictionary of feature importances
        """
        if not self.is_trained:
            return None
        
        importance_dict = self.model.get_score(importance_type='weight')
        
        # Normalize
        total = sum(importance_dict.values())
        if total > 0:
            importance_dict = {k: v/total for k, v in importance_dict.items()}
        
        return importance_dict

