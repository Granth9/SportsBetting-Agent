"""Gradient boosting model using XGBoost."""

import os
# Set environment variables before importing xgboost to prevent segmentation faults
# These must be set before the xgboost module is imported
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    XGBClassifier = None

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
        
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not installed. Install with: pip install xgboost")
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        # Store params for XGBClassifier initialization
        # Use conservative settings to avoid segmentation faults
        self.model_params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'tree_method': 'hist',  # Use histogram method for better performance and stability
            'n_jobs': 1,  # Single thread to avoid segmentation faults
            'random_state': 42,
            'verbosity': 0,  # Suppress warnings
            'use_label_encoder': False,  # Avoid deprecated warnings
            **kwargs
        }
    
    def train(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None, sample_weight: np.ndarray = None, **kwargs) -> None:
        """Train the gradient boosting model using XGBClassifier (sklearn-compatible API).
        
        Args:
            X: Training features
            y: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            sample_weight: Optional sample weights for training
            **kwargs: Additional training parameters (may include class_weight)
        """
        logger.info(f"Training {self.name} on {len(X)} samples")
        if sample_weight is not None:
            logger.info(f"Using temporal weighting (sample weights provided)")
        
        # Ensure data types are correct and arrays are contiguous
        # This is critical for avoiding segmentation faults
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float32), dtype=np.float32)
        y = np.ascontiguousarray(np.asarray(y, dtype=np.int32), dtype=np.int32)
        
        # Verify array properties
        if not X.flags['C_CONTIGUOUS']:
            logger.warning(f"X array is not contiguous, forcing contiguous layout")
            X = np.ascontiguousarray(X)
        if not y.flags['C_CONTIGUOUS']:
            logger.warning(f"y array is not contiguous, forcing contiguous layout")
            y = np.ascontiguousarray(y)
        
        # Convert sample weights to float32 if provided
        if sample_weight is not None:
            sample_weight = np.ascontiguousarray(np.asarray(sample_weight, dtype=np.float32), dtype=np.float32)
            # Normalize weights to avoid numerical issues
            if sample_weight.sum() > 0:
                sample_weight = sample_weight / sample_weight.mean()
        
        # Extract class_weight if provided and convert to scale_pos_weight
        class_weight = kwargs.pop('class_weight', None)
        params = self.model_params.copy()
        
        if class_weight is not None:
            logger.info(f"Using class weights: {class_weight}")
            # XGBoost uses scale_pos_weight for binary classification
            if isinstance(class_weight, dict):
                # Calculate scale_pos_weight = negative_count / positive_count
                pos_count = np.sum(y == 1)
                neg_count = np.sum(y == 0)
                if pos_count > 0:
                    scale_pos_weight = (neg_count * class_weight.get(0, 1.0)) / (pos_count * class_weight.get(1, 1.0))
                    params['scale_pos_weight'] = scale_pos_weight
            elif class_weight == 'balanced':
                # Calculate balanced scale_pos_weight
                pos_count = np.sum(y == 1)
                neg_count = np.sum(y == 0)
                if pos_count > 0:
                    params['scale_pos_weight'] = neg_count / pos_count
        
        # Use XGBClassifier (sklearn-compatible, more stable than xgb.train)
        try:
            logger.debug(f"Starting XGBoost training with {self.n_estimators} estimators...")
            logger.debug(f"Data shapes: X={X.shape}, y={y.shape}, X dtype={X.dtype}, y dtype={y.dtype}")
            
            self.model = XGBClassifier(**params)
            
            # Fit with validation set if provided (for early stopping)
            # Note: early_stopping_rounds is deprecated in newer XGBoost versions
            # Use callbacks instead if needed, but for now just fit without early stopping
            if X_val is not None and y_val is not None:
                # Ensure validation data is also contiguous and correct type
                X_val = np.ascontiguousarray(np.asarray(X_val, dtype=np.float32), dtype=np.float32)
                y_val = np.ascontiguousarray(np.asarray(y_val, dtype=np.int32), dtype=np.int32)
                
                # Fit with validation set for monitoring (but no early stopping in sklearn API)
                self.model.fit(
                    X, y,
                    sample_weight=sample_weight,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )
            else:
                self.model.fit(X, y, sample_weight=sample_weight)
            
            logger.debug(f"XGBoost training completed successfully")
        except Exception as e:
            logger.error(f"Error during XGBoost training: {e}", exc_info=True)
            # Try fallback with different tree_method if hist fails
            if 'tree_method' in params and params['tree_method'] == 'hist':
                logger.warning("Histogram method failed, trying approximate method...")
                try:
                    params_fallback = params.copy()
                    params_fallback['tree_method'] = 'approx'
                    self.model = XGBClassifier(**params_fallback)
                    if X_val is not None and y_val is not None:
                        self.model.fit(X, y, sample_weight=sample_weight, eval_set=[(X_val, y_val)], verbose=False)
                    else:
                        self.model.fit(X, y, sample_weight=sample_weight)
                    logger.info("XGBoost training succeeded with approximate tree method")
                except Exception as e2:
                    logger.error(f"Fallback method also failed: {e2}", exc_info=True)
                    raise
            else:
                raise
        
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
        
        # Ensure correct data type
        X = np.asarray(X, dtype=np.float32)
        
        # Predict probability using XGBClassifier
        prob = float(self.model.predict_proba(X)[0, 1])  # Probability of class 1
        
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
        
        # Ensure correct data type
        X = np.asarray(X, dtype=np.float32)
        
        # Predict probability using XGBClassifier
        prob_home = float(self.model.predict_proba(X)[0, 1])  # Probability of class 1
        
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
        
        # XGBClassifier uses feature_importances_ attribute
        try:
            importances = self.model.feature_importances_
            
            # Create dictionary with feature names or indices
            if hasattr(self, 'feature_names') and self.feature_names:
                importance_dict = {name: float(imp) for name, imp in zip(self.feature_names, importances)}
            else:
                # Use feature indices if names not available
                importance_dict = {f'feature_{i}': float(imp) for i, imp in enumerate(importances)}
            
            # Normalize
            total = sum(importance_dict.values())
            if total > 0:
                importance_dict = {k: v/total for k, v in importance_dict.items()}
            
            return importance_dict
        except Exception as e:
            logger.warning(f"Could not get feature importance: {e}")
            return None

