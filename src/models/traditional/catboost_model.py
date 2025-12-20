"""CatBoost model for betting predictions."""

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

import numpy as np
from typing import Dict, Any, Optional

from src.models.base_model import BaseModel
from src.utils.data_types import ModelPrediction, Outcome
from src.utils.logger import setup_logger


logger = setup_logger(__name__)


class CatBoostModel(BaseModel):
    """CatBoost gradient boosting predictor."""
    
    def __init__(
        self,
        name: str = "CatBoost Optimizer",
        n_estimators: int = 200,
        max_depth: int = 7,
        learning_rate: float = 0.05,
        **kwargs
    ):
        """Initialize the CatBoost model.
        
        Args:
            name: Model name
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate
            **kwargs: Additional CatBoost parameters
        """
        super().__init__(name, "catboost")
        
        if not CATBOOST_AVAILABLE:
            raise ImportError("CatBoost not installed. Install with: pip install catboost")
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        
        self.params = {
            'iterations': n_estimators,
            'depth': max_depth,
            'learning_rate': learning_rate,
            'loss_function': 'Logloss',
            'eval_metric': 'Logloss',
            'random_seed': 42,
            'verbose': False,
            **kwargs
        }
    
    def train(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None, 
              sample_weight: np.ndarray = None, **kwargs) -> None:
        """Train the CatBoost model.
        
        Args:
            X: Training features
            y: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            sample_weight: Optional sample weights for training
        """
        logger.info(f"Training {self.name} on {len(X)} samples")
        if sample_weight is not None:
            logger.info(f"Using temporal weighting (sample weights provided)")
        
        # Create CatBoost Pool with sample weights if provided
        train_pool = cb.Pool(
            X, 
            label=y,
            weight=sample_weight if sample_weight is not None else None
        )
        
        # Add validation set if provided
        val_pool = None
        if X_val is not None and y_val is not None:
            val_pool = cb.Pool(X_val, label=y_val)
        
        # Train model
        self.model = cb.CatBoostClassifier(**self.params)
        self.model.fit(
            train_pool,
            eval_set=val_pool,
            verbose=False,
            plot=False
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
        prob = float(self.model.predict_proba(X)[0][1])  # Probability of class 1 (HOME_WIN)
        
        # Determine outcome
        prediction = Outcome.HOME_WIN if prob > 0.5 else Outcome.AWAY_WIN
        confidence = float(prob if prob > 0.5 else 1 - prob)
        
        # Get feature importance
        key_features = self.get_feature_importance()
        if key_features:
            top_features = dict(sorted(key_features.items(), key=lambda x: x[1], reverse=True)[:5])
        else:
            top_features = {}
        
        reasoning = f"CatBoost model predicts {prediction.value} with {confidence:.1%} confidence. "
        if top_features:
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
        
        probs = self.model.predict_proba(X)[0]
        prob_home = float(probs[1])  # Class 1 is HOME_WIN
        
        return {
            Outcome.HOME_WIN: prob_home,
            Outcome.AWAY_WIN: 1 - prob_home
        }
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance scores.
        
        Returns:
            Dictionary of feature importances
        """
        if not self.is_trained:
            return None
        
        importances = self.model.get_feature_importance()
        
        # Map to feature names if available
        if self.feature_names and len(self.feature_names) == len(importances):
            importance_dict = dict(zip(self.feature_names, importances))
        else:
            # Use indices if no feature names
            importance_dict = {f'feature_{i}': float(imp) for i, imp in enumerate(importances)}
        
        # Normalize
        total = sum(importance_dict.values())
        if total > 0:
            importance_dict = {k: v/total for k, v in importance_dict.items()}
        
        return importance_dict

