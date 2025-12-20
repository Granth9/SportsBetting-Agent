"""Stacking meta-learner that learns optimal combination of base models."""

import numpy as np
from typing import Dict, Any, List, Optional
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb

from src.models.base_model import BaseModel
from src.utils.data_types import ModelPrediction, Outcome
from src.utils.logger import setup_logger


logger = setup_logger(__name__)


class StackingModel(BaseModel):
    """Stacking meta-learner that learns optimal combination of base models.
    
    Uses out-of-fold predictions to train a meta-learner, which learns
    how to best combine base model predictions. This typically outperforms
    simple voting by 2-5% accuracy.
    """
    
    def __init__(
        self,
        name: str = "Stacking Meta-Learner",
        base_models: Optional[List[BaseModel]] = None,
        meta_learner_type: str = "logistic",
        n_folds: int = 5,
        **kwargs
    ):
        """Initialize the stacking model.
        
        Args:
            name: Model name
            base_models: List of base models to stack
            meta_learner_type: Type of meta-learner ('logistic', 'lightgbm', 'neural')
            n_folds: Number of folds for out-of-fold predictions
            **kwargs: Additional parameters for meta-learner
        """
        super().__init__(name, "stacking")
        
        self.base_models = base_models or []
        self.meta_learner_type = meta_learner_type
        self.n_folds = n_folds
        self.meta_learner = None
        self.meta_kwargs = kwargs
        
        # Out-of-fold predictions storage
        self.oof_predictions: Optional[np.ndarray] = None
        self.oof_labels: Optional[np.ndarray] = None
    
    def train(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None, sample_weight: np.ndarray = None, **kwargs) -> None:
        """Train the stacking model.
        
        Uses out-of-fold predictions to train meta-learner, preventing overfitting.
        
        Args:
            X: Training features
            y: Training labels
            X_val: Validation features (optional, used for final meta-learner training)
            y_val: Validation labels (optional)
            sample_weight: Optional sample weights for training
            **kwargs: Additional training parameters
        """
        if not self.base_models:
            raise ValueError("No base models to stack")
        
        logger.info(f"Training {self.name} with {len(self.base_models)} base models using {self.n_folds}-fold stacking")
        
        # Step 1: Train base models and generate out-of-fold predictions
        logger.info("Step 1: Generating out-of-fold predictions from base models...")
        oof_predictions = self._generate_oof_predictions(X, y, sample_weight=sample_weight, **kwargs)
        
        # Step 2: Train meta-learner on out-of-fold predictions
        logger.info("Step 2: Training meta-learner on out-of-fold predictions...")
        self._train_meta_learner(oof_predictions, y, X_val, y_val)
        
        # Store feature names from first base model
        if self.base_models and self.base_models[0].feature_names:
            self.feature_names = self.base_models[0].feature_names
        
        self.is_trained = True
        logger.info(f"{self.name} training completed")
    
    def _generate_oof_predictions(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray = None, **kwargs) -> np.ndarray:
        """Generate out-of-fold predictions from base models.
        
        Args:
            X: Training features
            y: Training labels
            sample_weight: Optional sample weights for full dataset (will be recalculated per fold)
            **kwargs: Additional training parameters
            
        Returns:
            Array of shape (n_samples, n_models) with out-of-fold predictions
        """
        n_samples = len(X)
        n_models = len(self.base_models)
        oof_predictions = np.zeros((n_samples, n_models))
        
        # Extract sample_weight from kwargs if provided there
        if sample_weight is None:
            sample_weight = kwargs.pop('sample_weight', None)
        
        # Use KFold for out-of-fold predictions
        kf = KFold(n_splits=self.n_folds, shuffle=False)  # No shuffle to respect temporal order
        
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
            logger.debug(f"Processing fold {fold_idx + 1}/{self.n_folds}")
            
            X_train_fold = X[train_idx]
            y_train_fold = y[train_idx]
            X_val_fold = X[val_idx]
            
            # Recalculate sample weights for this fold if provided
            fold_sample_weight = None
            if sample_weight is not None:
                fold_sample_weight = sample_weight[train_idx]
            
            # Prepare kwargs for this fold (without sample_weight, we'll pass it separately)
            fold_kwargs = kwargs.copy()
            if 'sample_weight' in fold_kwargs:
                del fold_kwargs['sample_weight']
            
            # Train each base model on this fold's training data
            for model_idx, model in enumerate(self.base_models):
                # XGBoost and LightGBM are now fixed with environment variables and data type safety
                # No need to skip them anymore
                
                try:
                    # For stacking, we need to retrain models on each fold
                    # Save original training state
                    original_trained = model.is_trained
                    
                    # Train model on this fold's training data with fold-specific sample weights
                    if fold_sample_weight is not None:
                        model.train(X_train_fold, y_train_fold, sample_weight=fold_sample_weight, **fold_kwargs)
                    else:
                        model.train(X_train_fold, y_train_fold, **fold_kwargs)
                    
                    # Get predictions on validation fold
                    val_preds = []
                    for i in range(len(X_val_fold)):
                        pred = model.predict(X_val_fold[i:i+1])
                        # Convert prediction to probability
                        proba = model.predict_proba(X_val_fold[i:i+1])
                        # Use probability of HOME_WIN (class 1)
                        prob = proba.get(Outcome.HOME_WIN, pred.confidence if pred.prediction == Outcome.HOME_WIN else 1 - pred.confidence)
                        val_preds.append(prob)
                    
                    # Store out-of-fold predictions
                    oof_predictions[val_idx, model_idx] = val_preds
                    
                    # Restore original state (will be retrained on full data later)
                    if not original_trained:
                        model.is_trained = False
                    
                except Exception as e:
                    logger.warning(f"Error training {model.name} in fold {fold_idx + 1}: {e}")
                    # Use default prediction (0.5) if model fails
                    oof_predictions[val_idx, model_idx] = 0.5
        
        # Retrain all base models on full training data for final use
        # Note: Base models should already be trained, but we ensure they're trained on full data
        logger.info("Ensuring base models are trained on full training data...")
        for model in self.base_models:
            # XGBoost and LightGBM are now fixed - no need to skip them
                
            try:
                # Only retrain if not already trained on full data
                # (Some models may have been trained during OOF generation)
                if not model.is_trained:
                    model.train(X, y, **kwargs)
                else:
                    # Model is already trained, but we may want to retrain on full data
                    # For now, we'll keep the existing training
                    logger.debug(f"Base model {model.name} already trained, skipping retrain")
            except Exception as e:
                logger.warning(f"Error ensuring {model.name} is trained on full data: {e}")
        
        # Store for later use
        self.oof_predictions = oof_predictions
        self.oof_labels = y
        
        logger.info(f"Generated out-of-fold predictions: shape {oof_predictions.shape}")
        return oof_predictions
    
    def _train_meta_learner(self, oof_predictions: np.ndarray, y: np.ndarray, 
                           X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> None:
        """Train the meta-learner on out-of-fold predictions.
        
        Args:
            oof_predictions: Out-of-fold predictions from base models
            y: Training labels
            X_val: Optional validation features (for final training)
            y_val: Optional validation labels
        """
        if self.meta_learner_type == "logistic":
            # Use Logistic Regression as meta-learner
            self.meta_learner = LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=42,
                **self.meta_kwargs
            )
            self.meta_learner.fit(oof_predictions, y)
            logger.info("Trained Logistic Regression meta-learner")
            
        elif self.meta_learner_type == "lightgbm":
            # Use LightGBM as meta-learner
            train_data = lgb.Dataset(oof_predictions, label=y)
            
            params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'verbose': -1,
                **self.meta_kwargs
            }
            
            valid_sets = [train_data]
            if X_val is not None and y_val is not None:
                # Get base model predictions on validation set
                val_predictions = self._get_base_predictions(X_val)
                val_data = lgb.Dataset(val_predictions, label=y_val, reference=train_data)
                valid_sets.append(val_data)
            
            self.meta_learner = lgb.train(
                params,
                train_data,
                num_boost_round=100,
                valid_sets=valid_sets,
                callbacks=[lgb.early_stopping(stopping_rounds=20), lgb.log_evaluation(period=0)]
            )
            logger.info("Trained LightGBM meta-learner")
            
        else:
            raise ValueError(f"Unknown meta-learner type: {self.meta_learner_type}")
    
    def _get_base_predictions(self, X: np.ndarray) -> np.ndarray:
        """Get predictions from all base models.
        
        Args:
            X: Input features
            
        Returns:
            Array of shape (n_samples, n_models) with base model predictions
        """
        n_samples = len(X)
        n_models = len(self.base_models)
        base_predictions = np.zeros((n_samples, n_models))
        
        for model_idx, model in enumerate(self.base_models):
            if not model.is_trained:
                logger.warning(f"Base model {model.name} not trained, using default prediction")
                base_predictions[:, model_idx] = 0.5
                continue
            
            try:
                for i in range(n_samples):
                    pred = model.predict(X[i:i+1])
                    proba = model.predict_proba(X[i:i+1])
                    # Use probability of HOME_WIN (class 1)
                    prob = proba.get(Outcome.HOME_WIN, pred.confidence if pred.prediction == Outcome.HOME_WIN else 1 - pred.confidence)
                    base_predictions[i, model_idx] = prob
            except Exception as e:
                logger.warning(f"Error getting prediction from {model.name}: {e}")
                base_predictions[:, model_idx] = 0.5
        
        return base_predictions
    
    def predict(self, X: np.ndarray) -> ModelPrediction:
        """Make a prediction using the meta-learner.
        
        Args:
            X: Input features
            
        Returns:
            ModelPrediction object
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        if self.meta_learner is None:
            raise ValueError("Meta-learner not trained")
        
        # Get predictions from all base models
        base_predictions = self._get_base_predictions(X)
        
        # Get meta-learner prediction
        if self.meta_learner_type == "logistic":
            meta_probs = self.meta_learner.predict_proba(base_predictions)[0]
            meta_pred = int(self.meta_learner.predict(base_predictions)[0])
        elif self.meta_learner_type == "lightgbm":
            meta_prob = float(self.meta_learner.predict(base_predictions)[0])
            meta_probs = np.array([1 - meta_prob, meta_prob])
            meta_pred = 1 if meta_prob > 0.5 else 0
        else:
            raise ValueError(f"Unknown meta-learner type: {self.meta_learner_type}")
        
        # Determine outcome
        prediction = Outcome.HOME_WIN if meta_pred == 1 else Outcome.AWAY_WIN
        confidence = float(meta_probs[meta_pred])
        
        # Get feature importance from meta-learner
        key_features = self._get_meta_importance()
        
        reasoning = f"Stacking meta-learner combines {len(self.base_models)} base models. "
        reasoning += f"Meta-learner ({self.meta_learner_type}) predicts {prediction.value} with {confidence:.1%} confidence."
        
        return ModelPrediction(
            model_name=self.name,
            prediction=prediction,
            confidence=confidence,
            probability=confidence,
            key_features=key_features,
            reasoning=reasoning
        )
    
    def predict_proba(self, X: np.ndarray) -> Dict[Outcome, float]:
        """Get probability distribution from meta-learner.
        
        Args:
            X: Input features
            
        Returns:
            Dictionary of outcome probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Get predictions from all base models
        base_predictions = self._get_base_predictions(X)
        
        # Get meta-learner probabilities
        if self.meta_learner_type == "logistic":
            meta_probs = self.meta_learner.predict_proba(base_predictions)[0]
        elif self.meta_learner_type == "lightgbm":
            meta_prob = float(self.meta_learner.predict(base_predictions)[0])
            meta_probs = np.array([1 - meta_prob, meta_prob])
        else:
            raise ValueError(f"Unknown meta-learner type: {self.meta_learner_type}")
        
        return {
            Outcome.AWAY_WIN: float(meta_probs[0]),
            Outcome.HOME_WIN: float(meta_probs[1])
        }
    
    def _get_meta_importance(self) -> Dict[str, Any]:
        """Get feature importance from meta-learner (which models are most important).
        
        Returns:
            Dictionary mapping model names to importance scores
        """
        if self.meta_learner is None:
            return {}
        
        importance = {}
        
        if self.meta_learner_type == "logistic":
            # Get coefficients from logistic regression
            coef = self.meta_learner.coef_[0]
            for i, model in enumerate(self.base_models):
                if i < len(coef):
                    importance[model.name] = float(abs(coef[i]))
        
        elif self.meta_learner_type == "lightgbm":
            # Get feature importance from LightGBM
            importances = self.meta_learner.feature_importance(importance_type='gain')
            for i, model in enumerate(self.base_models):
                if i < len(importances):
                    importance[model.name] = float(importances[i])
        
        # Normalize
        total = sum(importance.values())
        if total > 0:
            importance = {k: v / total for k, v in importance.items()}
        
        return importance
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance (meta-learner shows which base models are important).
        
        Returns:
            Dictionary mapping model names to importance scores
        """
        return self._get_meta_importance()

