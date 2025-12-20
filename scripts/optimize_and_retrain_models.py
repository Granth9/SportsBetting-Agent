"""Optimize and re-train all models with all recommendations implemented.

This script:
1. Fixes neural network training issues
2. Ensures all base models are trained
3. Applies feature selection
4. Performs hyperparameter tuning
5. Re-trains all models with optimizations
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import joblib
import json
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import logging

from src.utils.logger import setup_logger
from scripts.train_models import (
    prepare_training_data,
    select_important_features,
    train_single_model
)
from sklearn.utils.class_weight import compute_class_weight

logger = setup_logger(__name__)


def get_batch_predictions(model, X_val, model_type: str):
    """Get batch predictions efficiently based on model type.
    
    Args:
        model: Trained model
        X_val: Validation features
        model_type: Type of model
        
    Returns:
        Array of predictions (0 or 1)
    """
    try:
        # For sklearn models, use direct predict_proba
        if hasattr(model, 'model') and hasattr(model.model, 'predict_proba'):
            probs = model.model.predict_proba(X_val)
            return np.argmax(probs, axis=1)
        # For LightGBM
        elif hasattr(model, 'model') and hasattr(model.model, 'predict'):
            probs = model.model.predict(X_val)
            return (probs > 0.5).astype(int)
        # Fallback to one-by-one
        else:
            predictions = []
            for i in range(len(X_val)):
                pred = model.predict(X_val[i:i+1])
                predictions.append(1 if pred.prediction.value == 'HOME_WIN' else 0)
            return np.array(predictions)
    except Exception as e:
        # Fallback to one-by-one if batch fails
        predictions = []
        for i in range(len(X_val)):
            pred = model.predict(X_val[i:i+1])
            predictions.append(1 if pred.prediction.value == 'HOME_WIN' else 0)
        return np.array(predictions)


def optimize_hyperparameters(model, X_train, y_train, X_val, y_val, model_type: str):
    """Optimize hyperparameters for a model using simple grid search.
    
    Args:
        model: Model instance
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        model_type: Type of model ('neural', 'gradient_boost', 'random_forest', etc.)
        
    Returns:
        Optimized model with best hyperparameters
    """
    from sklearn.metrics import accuracy_score
    
    logger.info(f"Optimizing hyperparameters for {model.name} ({model_type})...")
    
    best_score = 0.0
    best_model = None
    
    if model_type == 'neural':
        # Neural network hyperparameter optimization
        learning_rates = [0.0001, 0.0003, 0.001]
        hidden_layers_configs = [
            [256, 128],
            [512, 256, 128],
            [512, 256, 128, 64]
        ]
        
        for lr in learning_rates:
            for hidden_layers in hidden_layers_configs:
                try:
                    from src.models.neural_nets.deep_predictor import DeepPredictor
                    test_model = DeepPredictor(
                        name=model.name,
                        hidden_layers=hidden_layers,
                        learning_rate=lr,
                        epochs=50,  # Reduced for faster search
                        batch_size=32,
                        dropout=0.3
                    )
                    test_model.train(X_train, y_train, X_val, y_val)
                    
                    # Use batch predictions (much faster)
                    predictions = get_batch_predictions(test_model, X_val, model_type)
                    score = accuracy_score(y_val, predictions)
                    if score > best_score:
                        best_score = score
                        best_model = test_model
                        logger.info(f"  New best: LR={lr}, Layers={hidden_layers}, Acc={score:.3f}")
                except Exception as e:
                    logger.debug(f"  Failed LR={lr}, Layers={hidden_layers}: {e}")
                    continue
    
    elif model_type == 'gradient_boost':
        # XGBoost hyperparameter optimization
        learning_rates = [0.01, 0.05, 0.1]
        max_depths = [5, 7, 10]
        n_estimators_list = [100, 200, 300]
        
        for lr in learning_rates:
            for max_depth in max_depths:
                for n_est in n_estimators_list:
                    try:
                        from src.models.traditional.gradient_boost_model import GradientBoostModel
                        test_model = GradientBoostModel(
                            name=model.name,
                            n_estimators=n_est,
                            max_depth=max_depth,
                            learning_rate=lr
                        )
                        test_model.train(X_train, y_train, X_val, y_val)
                        
                        # Use batch prediction instead of one-by-one (much faster)
                        val_probs = test_model.model.predict_proba(X_val)
                        predictions = np.argmax(val_probs, axis=1)
                        
                        score = accuracy_score(y_val, predictions)
                        if score > best_score:
                            best_score = score
                            best_model = test_model
                            logger.info(f"  New best: LR={lr}, Depth={max_depth}, Est={n_est}, Acc={score:.3f}")
                    except Exception as e:
                        logger.debug(f"  Failed LR={lr}, Depth={max_depth}, Est={n_est}: {e}")
                        continue
    
    elif model_type == 'random_forest':
        # Random Forest hyperparameter optimization
        # Reduced search space for faster optimization
        n_estimators_list = [100, 200]  # Reduced from [100, 200, 300]
        max_depths = [10, 15]  # Reduced from [10, 15, 20]
        min_samples_splits = [5, 10]  # Reduced from [5, 10, 20]
        
        for n_est in n_estimators_list:
            for max_depth in max_depths:
                for min_split in min_samples_splits:
                    try:
                        from src.models.traditional.random_forest_model import RandomForestModel
                        test_model = RandomForestModel(
                            name=model.name,
                            n_estimators=n_est,
                            max_depth=max_depth,
                            min_samples_split=min_split,
                            n_jobs=-1  # Enable parallelization
                        )
                        test_model.train(X_train, y_train, X_val, y_val)
                        
                        # Use batch prediction instead of one-by-one (much faster)
                        val_probs = test_model.model.predict_proba(X_val)
                        predictions = np.argmax(val_probs, axis=1)
                        
                        score = accuracy_score(y_val, predictions)
                        if score > best_score:
                            best_score = score
                            best_model = test_model
                            logger.info(f"  New best: Est={n_est}, Depth={max_depth}, MinSplit={min_split}, Acc={score:.3f}")
                    except Exception as e:
                        logger.debug(f"  Failed Est={n_est}, Depth={max_depth}, MinSplit={min_split}: {e}")
                        continue
    
    elif model_type == 'lightgbm':
        # LightGBM hyperparameter optimization
        learning_rates = [0.01, 0.05, 0.1]
        max_depths = [5, 7, 10]
        num_leaves_list = [31, 63, 127]
        
        for lr in learning_rates:
            for max_depth in max_depths:
                for num_leaves in num_leaves_list:
                    try:
                        from src.models.traditional.lightgbm_model import LightGBMModel
                        test_model = LightGBMModel(
                            name=model.name,
                            n_estimators=200,
                            max_depth=max_depth,
                            learning_rate=lr,
                            num_leaves=num_leaves
                        )
                        test_model.train(X_train, y_train, X_val, y_val)
                        
                        # Use batch prediction instead of one-by-one (much faster)
                        val_probs = test_model.model.predict_proba(X_val)
                        predictions = np.argmax(val_probs, axis=1)
                        
                        score = accuracy_score(y_val, predictions)
                        if score > best_score:
                            best_score = score
                            best_model = test_model
                            logger.info(f"  New best: LR={lr}, Depth={max_depth}, Leaves={num_leaves}, Acc={score:.3f}")
                    except Exception as e:
                        logger.debug(f"  Failed LR={lr}, Depth={max_depth}, Leaves={num_leaves}: {e}")
                        continue
    
    elif model_type == 'svm':
        # SVM hyperparameter optimization
        C_values = [0.1, 1.0, 10.0]
        gamma_values = ['scale', 'auto', 0.001, 0.01]
        
        for C in C_values:
            for gamma in gamma_values:
                try:
                    from src.models.traditional.svm_model import SVMModel
                    test_model = SVMModel(
                        name=model.name,
                        C=C,
                        kernel='rbf',
                        gamma=gamma
                    )
                    test_model.train(X_train, y_train, X_val, y_val)
                    
                    # Use batch predictions (much faster)
                    predictions = get_batch_predictions(test_model, X_val, model_type)
                    score = accuracy_score(y_val, predictions)
                    if score > best_score:
                        best_score = score
                        best_model = test_model
                        logger.info(f"  New best: C={C}, Gamma={gamma}, Acc={score:.3f}")
                except Exception as e:
                    logger.debug(f"  Failed C={C}, Gamma={gamma}: {e}")
                    continue
    
    elif model_type == 'statistical':
        # Logistic Regression hyperparameter optimization
        C_values = [0.01, 0.1, 1.0, 10.0]
        
        for C in C_values:
            try:
                from src.models.traditional.statistical_model import StatisticalModel
                test_model = StatisticalModel(
                    name=model.name,
                    regularization=C
                )
                test_model.train(X_train, y_train, X_val, y_val)
                
                predictions = []
                for i in range(len(X_val)):
                    pred = test_model.predict(X_val[i:i+1])
                    predictions.append(1 if pred.prediction.value == 'HOME_WIN' else 0)
                
                score = accuracy_score(y_val, predictions)
                if score > best_score:
                    best_score = score
                    best_model = test_model
                    logger.info(f"  New best: C={C}, Acc={score:.3f}")
            except Exception as e:
                logger.debug(f"  Failed C={C}: {e}")
                continue
    
    if best_model is None:
        logger.warning(f"Could not optimize {model.name}, using original model")
        return model
    
    logger.info(f"Best hyperparameters for {model.name}: Accuracy={best_score:.3f}")
    return best_model


def create_all_models():
    """Create all model instances to ensure all are available."""
    models = []
    
    # Neural Network (with fixed gradient computation)
    from src.models.neural_nets.deep_predictor import DeepPredictor
    models.append(DeepPredictor(
        name="Neural Analyst",
        epochs=100,
        batch_size=32,
        early_stopping_patience=20,
        hidden_layers=[512, 256, 128, 64],
        dropout=0.35,
        learning_rate=0.0003
    ))
    
    # Statistical Models
    from src.models.traditional.statistical_model import StatisticalModel
    models.append(StatisticalModel(name="Statistical Conservative"))
    
    from src.models.traditional.svm_model import SVMModel
    models.append(SVMModel(name="SVM Strategist"))
    
    # Tree-based Models
    from src.models.traditional.gradient_boost_model import GradientBoostModel
    models.append(GradientBoostModel(
        name="Gradient Strategist",
        n_estimators=200,
        max_depth=7,
        learning_rate=0.05
    ))
    
    from src.models.traditional.random_forest_model import RandomForestModel
    models.append(RandomForestModel(
        name="Forest Evaluator",
        n_estimators=200,
        max_depth=15,
        min_samples_split=10
    ))
    
    from src.models.traditional.lightgbm_model import LightGBMModel
    models.append(LightGBMModel(
        name="LightGBM Optimizer",
        n_estimators=200,
        max_depth=7,
        learning_rate=0.05,
        num_leaves=31
    ))
    
    # CatBoost if available
    try:
        from src.models.traditional.catboost_model import CatBoostModel
        models.append(CatBoostModel(
            name="CatBoost Optimizer",
            iterations=300,
            depth=7,
            learning_rate=0.03
        ))
        logger.info("Added CatBoost model")
    except ImportError:
        logger.info("CatBoost not available, skipping")
    except Exception as e:
        logger.warning(f"Could not add CatBoost: {e}")
    
    # Ensemble models
    from src.models.ensemble.ensemble_model import EnsembleModel
    ensemble = EnsembleModel(
        name="Ensemble Council",
        base_models=models.copy(),
        voting_strategy="unanimous_high_confidence"
    )
    models.append(ensemble)
    
    # Stacking meta-learner
    try:
        from src.models.ensemble.stacking_model import StackingModel
        stacking = StackingModel(
            name="Stacking Meta-Learner",
            base_models=models[:-1].copy(),  # All except ensemble
            meta_learner_type="logistic",
            n_folds=5
        )
        models.append(stacking)
        logger.info("Added Stacking Meta-Learner")
    except Exception as e:
        logger.warning(f"Could not add Stacking Meta-Learner: {e}")
    
    return models


def main():
    """Main optimization and training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimize and re-train all models")
    parser.add_argument('--seasons', nargs='+', type=int, 
                       default=[2020, 2021, 2022, 2023, 2024],
                       help='Seasons to train on')
    parser.add_argument('--output-dir', default='models/optimized',
                       help='Directory to save optimized models')
    parser.add_argument('--n-features', type=int, default=75,
                       help='Number of top features to select')
    parser.add_argument('--optimize-hyperparams', action='store_true', default=True,
                       help='Perform hyperparameter optimization')
    parser.add_argument('--skip-optimization', dest='optimize_hyperparams', action='store_false',
                       help='Skip hyperparameter optimization')
    
    args = parser.parse_args()
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*60)
    logger.info("MODEL OPTIMIZATION AND RE-TRAINING")
    logger.info("="*60)
    logger.info(f"Seasons: {args.seasons}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Feature selection: Top {args.n_features} features")
    logger.info(f"Hyperparameter optimization: {args.optimize_hyperparams}")
    
    # Step 1: Prepare training data
    logger.info("\n" + "="*60)
    logger.info("Step 1: Preparing Training Data")
    logger.info("="*60)
    
    X_train, y_train, X_val, y_val, X_test, y_test, feature_names, preprocessor, sample_weights = prepare_training_data(
        args.seasons,
        include_2025_week15=True,
        temporal_weighting=True,
        decay_factor=0.15
    )
    
    logger.info(f"Training set: {X_train.shape}")
    logger.info(f"Validation set: {X_val.shape}")
    logger.info(f"Test set: {X_test.shape}")
    logger.info(f"Total features: {len(feature_names)}")
    
    # Step 2: Feature Selection
    logger.info("\n" + "="*60)
    logger.info("Step 2: Feature Selection")
    logger.info("="*60)
    
    selected_features = None
    if len(feature_names) > args.n_features:
        selected_features = select_important_features(
            X_train, y_train, feature_names, n_features=args.n_features
        )
        
        if selected_features:
            # Filter features
            feature_indices = [feature_names.index(f) for f in selected_features if f in feature_names]
            X_train = X_train[:, feature_indices]
            X_val = X_val[:, feature_indices]
            X_test = X_test[:, feature_indices]
            feature_names = selected_features
            
            logger.info(f"Selected {len(feature_names)} top features")
        else:
            logger.warning("Feature selection failed, using all features")
    else:
        logger.info(f"Feature count ({len(feature_names)}) <= target ({args.n_features}), skipping selection")
    
    # Step 3: Calculate class weights
    logger.info("\n" + "="*60)
    logger.info("Step 3: Calculating Class Weights")
    logger.info("="*60)
    
    # Calculate class weights to handle imbalance
    classes = np.unique(y_train)
    class_weights_array = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weights = dict(zip(classes, class_weights_array))
    logger.info(f"Class weights: {class_weights}")
    
    # Step 4: Create all models
    logger.info("\n" + "="*60)
    logger.info("Step 4: Creating All Models")
    logger.info("="*60)
    
    models = create_all_models()
    logger.info(f"Created {len(models)} models")
    
    # Step 5: Optimize and train models
    logger.info("\n" + "="*60)
    logger.info("Step 5: Optimizing and Training Models")
    logger.info("="*60)
    
    model_type_map = {
        'Neural Analyst': 'neural',
        'Statistical Conservative': 'statistical',
        'SVM Strategist': 'svm',
        'Gradient Strategist': 'gradient_boost',
        'Forest Evaluator': 'random_forest',
        'LightGBM Optimizer': 'lightgbm',
        'CatBoost Optimizer': 'catboost',
    }
    
    results = []
    for model in models:
        if model.name in ['Ensemble Council', 'Stacking Meta-Learner']:
            # Ensemble models will be trained after base models
            continue
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {model.name}")
        logger.info(f"{'='*60}")
        
        # Optimize hyperparameters if requested
        if args.optimize_hyperparams and model.name in model_type_map:
            model_type = model_type_map[model.name]
            try:
                model = optimize_hyperparameters(model, X_train, y_train, X_val, y_val, model_type)
            except Exception as e:
                logger.warning(f"Hyperparameter optimization failed for {model.name}: {e}")
                logger.info("Using default hyperparameters")
        
        # Train model
        try:
            result = train_single_model(
                model, X_train, y_train, X_val, y_val, X_test, y_test,
                feature_names, output_path, sample_weight=sample_weights,
                preprocessor=preprocessor, class_weight=class_weights
            )
            results.append(result)
            logger.info(f"✅ {model.name}: {result.get('accuracy', 0):.1%} accuracy")
        except Exception as e:
            logger.error(f"❌ Failed to train {model.name}: {e}", exc_info=True)
            results.append({
                'model': model.name,
                'success': False,
                'error': str(e)
            })
    
    # Step 6: Train ensemble models (after base models)
    logger.info("\n" + "="*60)
    logger.info("Step 6: Training Ensemble Models")
    logger.info("="*60)
    
    for model in models:
        if model.name in ['Ensemble Council', 'Stacking Meta-Learner']:
            try:
                result = train_single_model(
                    model, X_train, y_train, X_val, y_val, X_test, y_test,
                    feature_names, output_path, sample_weight=sample_weights,
                    preprocessor=preprocessor, class_weight=class_weights
                )
                results.append(result)
                logger.info(f"✅ {model.name}: {result.get('accuracy', 0):.1%} accuracy")
            except Exception as e:
                logger.error(f"❌ Failed to train {model.name}: {e}", exc_info=True)
                results.append({
                    'model': model.name,
                    'success': False,
                    'error': str(e)
                })
    
    # Step 7: Save metadata
    logger.info("\n" + "="*60)
    logger.info("Step 7: Saving Metadata")
    logger.info("="*60)
    
    successful_results = [r for r in results if r.get('success', False)]
    model_accuracies = {r['model']: r['accuracy'] for r in successful_results}
    
    metadata = {
        'trained_at': datetime.now().isoformat(),
        'seasons': args.seasons,
        'feature_names': feature_names,
        'num_features': len(feature_names),
        'selected_features': selected_features is not None,
        'n_features_selected': len(feature_names) if selected_features else None,
        'model_accuracies': model_accuracies,
        'hyperparameter_optimization': args.optimize_hyperparams,
        'temporal_weighting': True,
        'class_weights': class_weights
    }
    
    metadata_path = output_path / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Saved metadata to {metadata_path}")
    
    # Step 8: Summary
    logger.info("\n" + "="*60)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info("="*60)
    
    logger.info(f"\nSuccessfully trained {len(successful_results)} models:")
    for result in successful_results:
        logger.info(f"  {result['model']}: {result['accuracy']:.1%} accuracy")
    
    failed_results = [r for r in results if not r.get('success', False)]
    if failed_results:
        logger.warning(f"\nFailed to train {len(failed_results)} models:")
        for result in failed_results:
            logger.warning(f"  {result['model']}: {result.get('error', 'Unknown error')}")
    
    logger.info(f"\nModels saved to: {output_path}/")
    logger.info("="*60)


if __name__ == '__main__':
    main()

