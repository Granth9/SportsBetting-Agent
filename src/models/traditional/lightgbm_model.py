"""LightGBM model for betting predictions."""

import lightgbm as lgb
import numpy as np
from typing import Dict, Any, Optional

from src.models.base_model import BaseModel
from src.utils.data_types import ModelPrediction, Outcome
from src.utils.logger import setup_logger


logger = setup_logger(__name__)


class LightGBMModel(BaseModel):
    """LightGBM gradient boosting predictor."""
    
    def __init__(
        self,
        name: str = "LightGBM Optimizer",
        n_estimators: int = 200,
        max_depth: int = 7,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        **kwargs
    ):
        """Initialize the LightGBM model.
        
        Args:
            name: Model name
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate
            num_leaves: Number of leaves in one tree
            **kwargs: Additional LightGBM parameters
        """
        super().__init__(name, "lightgbm")
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        
        self.params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': num_leaves,
            'learning_rate': learning_rate,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'num_threads': 1,  # Use single thread to avoid crashes
            'force_col_wise': True,  # Force column-wise for stability
            **kwargs
        }
    
    def train(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None) -> None:
        """Train the LightGBM model.
        
        Args:
            X: Training features
            y: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
        """
        logger.info(f"Training {self.name} on {len(X)} samples")
        
        # Create dataset
        train_data = lgb.Dataset(X, label=y)
        
        # Add validation set if provided
        valid_sets = [train_data]
        valid_names = ['train']
        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets.append(val_data)
            valid_names.append('val')
        
        # Train model
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=self.n_estimators,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=[lgb.early_stopping(stopping_rounds=20), lgb.log_evaluation(period=0)]
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
        # Handle best_iteration - it might not exist if early stopping didn't trigger
        best_iter = getattr(self.model, 'best_iteration', None)
        if best_iter is not None:
            prob = self.model.predict(X, num_iteration=best_iter)[0]
        else:
            prob = self.model.predict(X)[0]
        
        # Determine outcome
        prediction = Outcome.HOME_WIN if prob > 0.5 else Outcome.AWAY_WIN
        confidence = float(prob if prob > 0.5 else 1 - prob)
        
        # Get feature importance
        key_features = self.get_feature_importance()
        if key_features:
            top_features = dict(sorted(key_features.items(), key=lambda x: x[1], reverse=True)[:5])
        else:
            top_features = {}
        
        reasoning = f"LightGBM model predicts {prediction.value} with {confidence:.1%} confidence. "
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
        
        # Handle best_iteration - it might not exist if early stopping didn't trigger
        best_iter = getattr(self.model, 'best_iteration', None)
        if best_iter is not None:
            prob_home = float(self.model.predict(X, num_iteration=best_iter)[0])
        else:
            prob_home = float(self.model.predict(X)[0])
        
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
        
        importance_dict = self.model.feature_importance(importance_type='gain')
        
        # Map to feature names if available
        if self.feature_names and len(self.feature_names) == len(importance_dict):
            importance_dict = dict(zip(self.feature_names, importance_dict))
        else:
            # Use indices if no feature names
            importance_dict = {f'feature_{i}': float(imp) for i, imp in enumerate(importance_dict)}
        
        # Normalize
        total = sum(importance_dict.values())
        if total > 0:
            importance_dict = {k: v/total for k, v in importance_dict.items()}
        
        return importance_dict
