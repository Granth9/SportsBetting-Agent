"""Hyperparameter tuning for models using Optuna."""

import optuna
import numpy as np
from typing import Dict, Any, List
from sklearn.model_selection import cross_val_score
import joblib

from src.models.neural_nets.deep_predictor import DeepPredictor
from src.models.traditional.gradient_boost_model import GradientBoostModel
from src.models.traditional.random_forest_model import RandomForestModel
from src.models.traditional.statistical_model import StatisticalModel
from src.models.traditional.lightgbm_model import LightGBMModel
from src.models.traditional.svm_model import SVMModel
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def tune_neural_network(X_train, y_train, X_val, y_val, n_trials=20):
    """Tune neural network hyperparameters."""
    def objective(trial):
        hidden_layers = [
            trial.suggest_int('layer1', 128, 512),
            trial.suggest_int('layer2', 64, 256),
            trial.suggest_int('layer3', 32, 128)
        ]
        dropout = trial.suggest_float('dropout', 0.2, 0.5)
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
        
        model = DeepPredictor(
            name="Neural Analyst",
            hidden_layers=hidden_layers,
            dropout=dropout,
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=150,
            early_stopping_patience=15
        )
        
        try:
            model.train(X_train, y_train, X_val, y_val)
            
            # Evaluate on validation set
            from sklearn.metrics import accuracy_score
            from src.utils.data_types import Outcome
            
            val_preds = []
            for i in range(len(X_val)):
                try:
                    pred = model.predict(X_val[i:i+1])
                    val_preds.append(1 if pred.prediction.value in ["home_win", "HOME_WIN"] else 0)
                except:
                    val_preds.append(0)
            
            accuracy = accuracy_score(y_val, val_preds)
            return accuracy
        except Exception as e:
            logger.warning(f"Trial failed: {e}")
            return 0.0
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def tune_xgboost(X_train, y_train, X_val, y_val, n_trials=20):
    """Tune XGBoost hyperparameters."""
    def objective(trial):
        n_estimators = trial.suggest_int('n_estimators', 100, 500)
        max_depth = trial.suggest_int('max_depth', 4, 10)
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.1, log=True)
        subsample = trial.suggest_float('subsample', 0.6, 1.0)
        
        model = GradientBoostModel(
            name="Gradient Strategist",
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate
        )
        model.params['subsample'] = subsample
        
        try:
            model.train(X_train, y_train, X_val, y_val)
            
            from sklearn.metrics import accuracy_score
            from src.utils.data_types import Outcome
            
            val_preds = []
            for i in range(len(X_val)):
                try:
                    pred = model.predict(X_val[i:i+1])
                    val_preds.append(1 if pred.prediction.value in ["home_win", "HOME_WIN"] else 0)
                except:
                    val_preds.append(0)
            
            accuracy = accuracy_score(y_val, val_preds)
            return accuracy
        except Exception as e:
            logger.warning(f"Trial failed: {e}")
            return 0.0
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def tune_lightgbm(X_train, y_train, X_val, y_val, n_trials=20):
    """Tune LightGBM hyperparameters."""
    def objective(trial):
        n_estimators = trial.suggest_int('n_estimators', 100, 500)
        max_depth = trial.suggest_int('max_depth', 4, 10)
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.1, log=True)
        num_leaves = trial.suggest_int('num_leaves', 31, 127)
        feature_fraction = trial.suggest_float('feature_fraction', 0.7, 1.0)
        
        model = LightGBMModel(
            name="LightGBM Optimizer",
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            num_leaves=num_leaves
        )
        model.params['feature_fraction'] = feature_fraction
        
        try:
            model.train(X_train, y_train, X_val, y_val)
            
            from sklearn.metrics import accuracy_score
            
            val_preds = []
            for i in range(len(X_val)):
                try:
                    pred = model.predict(X_val[i:i+1])
                    val_preds.append(1 if pred.prediction.value in ["home_win", "HOME_WIN"] else 0)
                except:
                    val_preds.append(0)
            
            accuracy = accuracy_score(y_val, val_preds)
            return accuracy
        except Exception as e:
            logger.warning(f"Trial failed: {e}")
            return 0.0
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def tune_random_forest(X_train, y_train, X_val, y_val, n_trials=20):
    """Tune Random Forest hyperparameters."""
    def objective(trial):
        n_estimators = trial.suggest_int('n_estimators', 200, 500)
        max_depth = trial.suggest_int('max_depth', 8, 20)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 5)
        
        model = RandomForestModel(
            name="Forest Evaluator",
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf
        )
        
        try:
            model.train(X_train, y_train, X_val, y_val)
            
            from sklearn.metrics import accuracy_score
            
            val_preds = []
            for i in range(len(X_val)):
                try:
                    pred = model.predict(X_val[i:i+1])
                    val_preds.append(1 if pred.prediction.value in ["home_win", "HOME_WIN"] else 0)
                except:
                    val_preds.append(0)
            
            accuracy = accuracy_score(y_val, val_preds)
            return accuracy
        except Exception as e:
            logger.warning(f"Trial failed: {e}")
            return 0.0
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params

