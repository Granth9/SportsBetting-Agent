"""LightGBM model for betting predictions."""

import os
# Set environment variables before importing lightgbm to prevent segmentation faults
# These must be set before the lightgbm module is imported
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')

# Try to use LGBMClassifier (sklearn API) for better stability
try:
    from lightgbm import LGBMClassifier
    LGBM_SKLEARN_AVAILABLE = True
except ImportError:
    LGBM_SKLEARN_AVAILABLE = False
    LGBMClassifier = None

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
        
        # Use sklearn-compatible API if available (more stable)
        self.use_sklearn_api = LGBM_SKLEARN_AVAILABLE
        
        # Parameters for native API (lgb.train)
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
        
        # Parameters for sklearn API (LGBMClassifier)
        self.sklearn_params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'num_leaves': num_leaves,
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'n_jobs': 1,  # Single thread
            'force_col_wise': True,
            'random_state': 42,
            **kwargs
        }
    
    def train(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None, sample_weight: np.ndarray = None, **kwargs) -> None:
        """Train the LightGBM model.
        
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
        
        # Extract class_weight if provided and convert to LightGBM format
        class_weight = kwargs.pop('class_weight', None)
        
        # Try sklearn API first (more stable)
        if self.use_sklearn_api:
            try:
                logger.debug("Attempting to train with LGBMClassifier (sklearn API)...")
                params = self.sklearn_params.copy()
                
                # Handle class weights
                if class_weight is not None:
                    logger.info(f"Using class weights: {class_weight}")
                    if isinstance(class_weight, dict):
                        pos_count = np.sum(y == 1)
                        neg_count = np.sum(y == 0)
                        if pos_count > 0:
                            scale_pos_weight = (neg_count * class_weight.get(0, 1.0)) / (pos_count * class_weight.get(1, 1.0))
                            params['scale_pos_weight'] = scale_pos_weight
                    elif class_weight == 'balanced':
                        pos_count = np.sum(y == 1)
                        neg_count = np.sum(y == 0)
                        if pos_count > 0:
                            params['scale_pos_weight'] = neg_count / pos_count
                
                self.model = LGBMClassifier(**params)
                
                # Fit with validation set if provided
                if X_val is not None and y_val is not None:
                    X_val = np.ascontiguousarray(np.asarray(X_val, dtype=np.float32), dtype=np.float32)
                    y_val = np.ascontiguousarray(np.asarray(y_val, dtype=np.int32), dtype=np.int32)
                    self.model.fit(
                        X, y,
                        sample_weight=sample_weight,
                        eval_set=[(X_val, y_val)],
                        callbacks=[lgb.early_stopping(stopping_rounds=20), lgb.log_evaluation(period=0)]
                    )
                else:
                    self.model.fit(X, y, sample_weight=sample_weight)
                
                self.is_trained = True
                logger.info(f"{self.name} training completed (using sklearn API)")
                
                # Store feature names for later use
                # If feature_names are set on the model, use those
                if hasattr(self, 'feature_names') and self.feature_names:
                    # Feature names already set from BaseModel
                    pass
                elif hasattr(X, 'columns'):
                    self.feature_names = list(X.columns)
                
                return
                
            except Exception as e:
                logger.warning(f"LGBMClassifier (sklearn API) failed: {e}. Falling back to native API...")
                # Fall through to native API
        
        # Fallback to native API (lgb.train)
        logger.debug("Using native LightGBM API (lgb.train)...")
        
        if class_weight is not None:
            logger.info(f"Using class weights: {class_weight}")
            # LightGBM doesn't support class_weight in params dict
            # Instead, we calculate scale_pos_weight for binary classification
            if isinstance(class_weight, dict):
                # Calculate scale_pos_weight = negative_count / positive_count
                pos_count = np.sum(y == 1)
                neg_count = np.sum(y == 0)
                if pos_count > 0:
                    scale_pos_weight = (neg_count * class_weight.get(0, 1.0)) / (pos_count * class_weight.get(1, 1.0))
                    self.params['scale_pos_weight'] = scale_pos_weight
            elif class_weight == 'balanced':
                # Calculate balanced scale_pos_weight
                pos_count = np.sum(y == 1)
                neg_count = np.sum(y == 0)
                if pos_count > 0:
                    self.params['scale_pos_weight'] = neg_count / pos_count
        
        # Create dataset with sample weights if provided
        train_data = lgb.Dataset(X, label=y, weight=sample_weight if sample_weight is not None else None)
        
        # Add validation set if provided
        valid_sets = [train_data]
        valid_names = ['train']
        if X_val is not None and y_val is not None:
            X_val = np.ascontiguousarray(np.asarray(X_val, dtype=np.float32), dtype=np.float32)
            y_val = np.ascontiguousarray(np.asarray(y_val, dtype=np.int32), dtype=np.int32)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets.append(val_data)
            valid_names.append('val')
        
        # Train model
        try:
            self.model = lgb.train(
                self.params,
                train_data,
                num_boost_round=self.n_estimators,
                valid_sets=valid_sets,
                valid_names=valid_names,
                callbacks=[lgb.early_stopping(stopping_rounds=20), lgb.log_evaluation(period=0)]
            )
        except Exception as e:
            logger.error(f"Error during LightGBM training: {e}", exc_info=True)
            raise
        
        # Store feature names if available
        if hasattr(X, 'columns'):
            self.feature_names = list(X.columns)
        
        self.is_trained = True
        logger.info(f"{self.name} training completed (using native API)")
    
    def predict(self, X: np.ndarray) -> ModelPrediction:
        """Make a prediction.
        
        Args:
            X: Input features
            
        Returns:
            ModelPrediction object
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Ensure correct data type and contiguous layout
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float32), dtype=np.float32)
        
        # Predict probability
        # Handle sklearn API vs native API
        if self.use_sklearn_api and isinstance(self.model, LGBMClassifier):
            # Convert to DataFrame with feature names if available to avoid warnings
            if hasattr(self, 'feature_names') and self.feature_names and len(self.feature_names) == X.shape[1]:
                import pandas as pd
                X_df = pd.DataFrame(X, columns=self.feature_names)
                prob = float(self.model.predict_proba(X_df)[0, 1])
            else:
                # Suppress feature name warning for numpy arrays
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=UserWarning, message='.*feature names.*')
                    prob = float(self.model.predict_proba(X)[0, 1])
        else:
            # Native API
            # Handle best_iteration - it might not exist if early stopping didn't trigger
            best_iter = getattr(self.model, 'best_iteration', None)
            if best_iter is not None:
                prob = float(self.model.predict(X, num_iteration=best_iter)[0])
            else:
                prob = float(self.model.predict(X)[0])
        
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
        
        # Ensure correct data type and contiguous layout
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float32), dtype=np.float32)
        
        # Handle sklearn API vs native API
        if self.use_sklearn_api and isinstance(self.model, LGBMClassifier):
            # Convert to DataFrame with feature names if available to avoid warnings
            if hasattr(self, 'feature_names') and self.feature_names and len(self.feature_names) == X.shape[1]:
                import pandas as pd
                X_df = pd.DataFrame(X, columns=self.feature_names)
                prob_home = float(self.model.predict_proba(X_df)[0, 1])
            else:
                # Suppress feature name warning for numpy arrays
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=UserWarning, message='.*feature names.*')
                    prob_home = float(self.model.predict_proba(X)[0, 1])
        else:
            # Native API
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
        
        # Handle sklearn API vs native API
        if self.use_sklearn_api and isinstance(self.model, LGBMClassifier):
            importance_dict = self.model.feature_importances_
        else:
            # Native API
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
