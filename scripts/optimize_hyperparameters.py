"""Hyperparameter optimization script using Optuna."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import optuna
from typing import Dict, Any
from sklearn.metrics import accuracy_score, roc_auc_score
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def optimize_model_hyperparameters(model_class, model_name: str, X_train: np.ndarray, y_train: np.ndarray,
                                   X_val: np.ndarray, y_val: np.ndarray, n_trials: int = 50) -> Dict[str, Any]:
    """Optimize hyperparameters for a model using Optuna.
    
    Args:
        model_class: Model class to optimize
        model_name: Name of the model
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        n_trials: Number of optimization trials
        
    Returns:
        Dictionary with best hyperparameters
    """
    def objective(trial):
        # Suggest hyperparameters based on model type
        if 'GradientBoost' in model_name or 'XGBoost' in model_name:
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
            }
            model = model_class(name=model_name, **params)
        elif 'RandomForest' in model_name:
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 5, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10)
            }
            model = model_class(name=model_name, **params)
        elif 'LightGBM' in model_name:
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 100)
            }
            model = model_class(name=model_name, **params)
        elif 'CatBoost' in model_name:
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True)
            }
            model = model_class(name=model_name, **params)
        elif 'Statistical' in model_name:
            params = {
                'C': trial.suggest_float('C', 0.1, 10.0, log=True),
                'penalty': trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet']),
                'solver': trial.suggest_categorical('solver', ['lbfgs', 'liblinear', 'saga'])
            }
            model = model_class(name=model_name, **params)
        elif 'SVM' in model_name:
            params = {
                'C': trial.suggest_float('C', 0.1, 100.0, log=True),
                'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']) + \
                         [trial.suggest_float('gamma_custom', 0.001, 1.0, log=True)],
                'kernel': trial.suggest_categorical('kernel', ['rbf', 'poly', 'sigmoid'])
            }
            model = model_class(name=model_name, **params)
        else:
            # Default: no optimization
            model = model_class(name=model_name)
        
        # Train model
        try:
            model.train(X_train, y_train, X_val, y_val)
            
            # Evaluate on validation set
            predictions = []
            probabilities = []
            for i in range(len(X_val)):
                pred = model.predict(X_val[i:i+1])
                proba = model.predict_proba(X_val[i:i+1])
                predictions.append(1 if pred.prediction.value == 'HOME_WIN' else 0)
                probabilities.append(proba.get('HOME_WIN', 0.5))
            
            accuracy = accuracy_score(y_val, predictions)
            roc_auc = roc_auc_score(y_val, probabilities)
            
            # Return combined score (weighted towards accuracy)
            return accuracy * 0.7 + roc_auc * 0.3
        except Exception as e:
            logger.warning(f"Trial failed for {model_name}: {e}")
            return 0.0
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    logger.info(f"\nBest hyperparameters for {model_name}:")
    logger.info(f"  Value: {study.best_value:.4f}")
    logger.info(f"  Params: {study.best_params}")
    
    return study.best_params


if __name__ == "__main__":
    # This would be called from train_models.py if hyperparameter tuning is enabled
    logger.info("Hyperparameter optimization module loaded")

