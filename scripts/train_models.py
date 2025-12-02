"""Script to train all models on historical data.

This is a template script. Adapt it to your specific data sources and requirements.
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

from src.models.neural_nets.deep_predictor import DeepPredictor
from src.models.traditional.gradient_boost_model import GradientBoostModel
from src.models.traditional.random_forest_model import RandomForestModel
from src.models.traditional.statistical_model import StatisticalModel
from src.models.traditional.lightgbm_model import LightGBMModel
from src.models.traditional.svm_model import SVMModel
from src.data.collectors.nfl_data_collector import NFLDataCollector
from src.data.processors.feature_engineer import FeatureEngineer
from src.data.processors.data_preprocessor import DataPreprocessor, DataSplitter
from src.utils.logger import setup_logger


logger = setup_logger(__name__)


def prepare_training_data(seasons):
    """Prepare training data from historical NFL games.
    
    Args:
        seasons: List of seasons to include
        
    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test, feature_names, preprocessor)
    """
    logger.info(f"Preparing training data for seasons: {seasons}")
    
    # Collect data
    collector = NFLDataCollector()
    logger.info("Collecting schedule data...")
    schedule_df = collector.get_schedule(seasons)
    
    logger.info("Collecting team stats...")
    team_stats_df = collector.get_team_stats(seasons)
    
    # Initialize feature engineer
    feature_engineer = FeatureEngineer()
    
    # Extract features for each game
    logger.info("Extracting features from games...")
    all_features = []
    all_labels = []
    
    for idx, game in schedule_df.iterrows():
        # Only process completed games (with scores)
        if pd.isna(game['home_score']) or pd.isna(game['away_score']):
            continue
        
        try:
            # Create a minimal proposition for feature extraction
            from src.utils.data_types import GameInfo, Proposition, BetType
            from datetime import datetime
            
            game_info = GameInfo(
                game_id=game['game_id'],
                home_team=game['home_team'],
                away_team=game['away_team'],
                game_date=pd.to_datetime(game['gameday']),
                season=int(game['season']),
                week=int(game['week']),
                home_score=int(game['home_score']),
                away_score=int(game['away_score'])
            )
            
            # Create proposition
            prop = Proposition(
                prop_id=game['game_id'],
                game_info=game_info,
                bet_type=BetType.GAME_OUTCOME
            )
            
            # Extract features
            features = feature_engineer.extract_features(
                prop,
                schedule_df,
                team_stats_df,
                None
            )
            
            # Create label (1 if home team won, 0 if away team won)
            label = 1 if game['home_score'] > game['away_score'] else 0
            
            all_features.append(features)
            all_labels.append(label)
            
        except Exception as e:
            logger.debug(f"Error processing game {game.get('game_id', 'unknown')}: {e}")
            continue
    
    logger.info(f"Extracted features from {len(all_features)} games")
    
    if len(all_features) == 0:
        raise ValueError("No features extracted. Check data collection and feature engineering.")
    
    # Convert to DataFrame
    features_df = pd.DataFrame(all_features)
    labels = np.array(all_labels)
    
    # Validate data
    validate_data(features_df, labels)
    
    # Split by season to avoid data leakage
    # Get season info from original schedule
    season_info = []
    feature_idx = 0
    for idx, game in schedule_df.iterrows():
        if pd.notna(game.get('home_score')) and pd.notna(game.get('away_score')):
            season_info.append(int(game['season']))
            feature_idx += 1
    
    features_df['season'] = season_info[:len(features_df)]
    features_df['label'] = labels
    
    # Use DataSplitter to split by date
    splitter = DataSplitter()
    
    # Determine split seasons
    train_seasons = seasons[:-2] if len(seasons) > 2 else [seasons[0]]
    val_seasons = [seasons[-2]] if len(seasons) > 2 else [seasons[0]]
    test_seasons = [seasons[-1]] if len(seasons) > 1 else [seasons[0]]
    
    logger.info(f"Train seasons: {train_seasons}")
    logger.info(f"Validation seasons: {val_seasons}")
    logger.info(f"Test seasons: {test_seasons}")
    
    train_df, val_df, test_df = splitter.split_by_season(
        features_df,
        'season',
        train_seasons,
        val_seasons,
        test_seasons
    )
    
    # Separate features and labels
    feature_cols = [col for col in features_df.columns if col not in ['season', 'label']]
    
    # Prepare data for preprocessing
    train_features_df = train_df[feature_cols].copy()
    val_features_df = val_df[feature_cols].copy()
    test_features_df = test_df[feature_cols].copy()
    
    # Initialize and fit preprocessor on training data
    preprocessor = DataPreprocessor(scaler_type='standard')
    preprocessor.fit(train_features_df)
    
    # Transform all datasets
    train_features_df = preprocessor.transform(train_features_df)
    val_features_df = preprocessor.transform(val_features_df)
    test_features_df = preprocessor.transform(test_features_df)
    
    # Convert to numpy arrays
    X_train = train_features_df.values
    y_train = train_df['label'].values
    X_val = val_features_df.values
    y_val = val_df['label'].values
    X_test = test_features_df.values
    y_test = test_df['label'].values
    
    logger.info(f"Training set: {X_train.shape}")
    logger.info(f"Validation set: {X_val.shape}")
    logger.info(f"Test set: {X_test.shape}")
    
    # Store feature names from preprocessor
    processed_feature_names = preprocessor.get_feature_names()
    
    return X_train, y_train, X_val, y_val, X_test, y_test, processed_feature_names, preprocessor


def validate_data(features_df: pd.DataFrame, labels: np.ndarray) -> None:
    """Validate training data for quality issues.
    
    Args:
        features_df: Features DataFrame
        labels: Labels array
        
    Raises:
        ValueError: If data validation fails
    """
    logger.info("Validating training data...")
    
    # Check for NaN values
    nan_count = features_df.isna().sum().sum()
    if nan_count > 0:
        logger.warning(f"Found {nan_count} NaN values in features, will be handled by preprocessor")
    
    # Check for infinite values
    inf_count = np.isinf(features_df.select_dtypes(include=[np.number])).sum().sum()
    if inf_count > 0:
        raise ValueError(f"Found {inf_count} infinite values in features")
    
    # Check for empty dataset
    if len(features_df) == 0:
        raise ValueError("Empty dataset")
    
    # Check label distribution
    unique_labels, counts = np.unique(labels, return_counts=True)
    logger.info(f"Label distribution: {dict(zip(unique_labels, counts))}")
    
    # Check for class imbalance (warn if > 70/30 split)
    min_class_ratio = min(counts) / max(counts)
    if min_class_ratio < 0.3:
        logger.warning(f"Severe class imbalance detected (ratio: {min_class_ratio:.2f})")
    
    logger.info("Data validation passed")


def train_single_model(model, X_train, y_train, X_val, y_val, X_test, y_test, feature_names, output_path):
    """Train a single model and return results.
    
    Args:
        model: Model instance to train
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        X_test: Test features
        y_test: Test labels
        feature_names: List of feature names
        output_path: Path to save model
        
    Returns:
        Dictionary with training results
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Training {model.name}...")
    logger.info('='*60)
    
    try:
        # Set feature names
        model.feature_names = feature_names
        
        # Train model
        model.train(X_train, y_train, X_val, y_val)
        
        # Evaluate on test set
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, 
            f1_score, roc_auc_score, confusion_matrix
        )
        from src.utils.data_types import Outcome
        
        test_predictions = []
        test_probs = []
        for i in range(len(X_test)):
            pred = model.predict(X_test[i:i+1])
            test_predictions.append(1 if pred.prediction.value in ["home_win", "HOME_WIN"] else 0)
            # Get probability for ROC-AUC
            proba = model.predict_proba(X_test[i:i+1])
            test_probs.append(proba.get(Outcome.HOME_WIN, 0.5))
        
        # Calculate comprehensive metrics
        test_accuracy = accuracy_score(y_test, test_predictions)
        test_precision = precision_score(y_test, test_predictions, zero_division=0)
        test_recall = recall_score(y_test, test_predictions, zero_division=0)
        test_f1 = f1_score(y_test, test_predictions, zero_division=0)
        
        # ROC-AUC
        try:
            test_roc_auc = roc_auc_score(y_test, test_probs)
        except Exception:
            test_roc_auc = 0.0
        
        # Confusion matrix
        cm = confusion_matrix(y_test, test_predictions)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        logger.info(f"Test Accuracy:  {test_accuracy:.3f}")
        logger.info(f"Test Precision: {test_precision:.3f}")
        logger.info(f"Test Recall:    {test_recall:.3f}")
        logger.info(f"Test F1:        {test_f1:.3f}")
        logger.info(f"Test ROC-AUC:   {test_roc_auc:.3f}")
        
        # Save model
        model_file = output_path / f"{model.name.lower().replace(' ', '_')}.pkl"
        model.save(str(model_file))
        
        logger.info(f"Saved {model.name} to {model_file}")
        
        return {
            'model': model.name,
            'accuracy': test_accuracy,
            'precision': test_precision,
            'recall': test_recall,
            'f1': test_f1,
            'roc_auc': test_roc_auc,
            'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)},
            'success': True
        }
    
    except Exception as e:
        logger.error(f"Error training {model.name}: {e}", exc_info=True)
        return {
            'model': model.name,
            'success': False,
            'error': str(e)
        }


def train_all_models(X_train, y_train, X_val, y_val, X_test, y_test, feature_names, output_dir, parallel=True):
    """Train all models.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        X_test: Test features
        y_test: Test labels
        feature_names: List of feature names
        output_dir: Directory to save models
        parallel: Whether to train models in parallel
    """
    from src.utils.data_types import Outcome
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize all 6 models
    models = [
        DeepPredictor(
            name="Neural Analyst",
            epochs=100,
            early_stopping_patience=10
        ),
        GradientBoostModel(name="Gradient Strategist"),
        RandomForestModel(name="Forest Evaluator"),
        StatisticalModel(name="Statistical Conservative"),
        LightGBMModel(name="LightGBM Optimizer"),
        SVMModel(name="SVM Strategist"),
    ]
    
    results = []
    
    if parallel:
        # Train models in parallel using ThreadPoolExecutor
        logger.info("Training models in parallel...")
        with ThreadPoolExecutor(max_workers=min(len(models), 4)) as executor:
            # Submit all training tasks
            future_to_model = {
                executor.submit(
                    train_single_model,
                    model, X_train, y_train, X_val, y_val, X_test, y_test,
                    feature_names, output_path
                ): model for model in models
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_model):
                result = future.result()
                results.append(result)
    else:
        # Train models sequentially
        logger.info("Training models sequentially...")
        for model in models:
            result = train_single_model(
                model, X_train, y_train, X_val, y_val, X_test, y_test,
                feature_names, output_path
            )
            results.append(result)
    
    # Filter successful results
    successful_results = [r for r in results if r.get('success', False)]
    failed_results = [r for r in results if not r.get('success', False)]
    
    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("TRAINING SUMMARY")
    logger.info('='*60)
    
    if successful_results:
        logger.info(f"\nSuccessfully trained {len(successful_results)} models:")
        for result in successful_results:
            logger.info(f"\n{result['model']}:")
            logger.info(f"  Accuracy:  {result['accuracy']:.1%}")
            logger.info(f"  Precision: {result['precision']:.1%}")
            logger.info(f"  Recall:    {result['recall']:.1%}")
            logger.info(f"  F1 Score:  {result['f1']:.1%}")
            logger.info(f"  ROC-AUC:   {result['roc_auc']:.3f}")
    
    if failed_results:
        logger.warning(f"\nFailed to train {len(failed_results)} models:")
        for result in failed_results:
            logger.warning(f"  {result['model']}: {result.get('error', 'Unknown error')}")
    
    return successful_results


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(
        description="Train betting prediction models on historical NFL data"
    )
    parser.add_argument('--seasons', nargs='+', type=int, required=True,
                       help='Seasons to train on (e.g., 2020 2021 2022 2023)')
    parser.add_argument('--output-dir', default='models',
                       help='Directory to save trained models')
    parser.add_argument('--no-parallel', action='store_true',
                       help='Disable parallel model training')
    
    args = parser.parse_args()
    
    logger.info("\n" + "="*60)
    logger.info("NFL BETTING AGENT COUNCIL - MODEL TRAINING")
    logger.info("="*60)
    logger.info(f"Seasons: {args.seasons}")
    logger.info(f"Output directory: {args.output_dir}\n")
    
    # Prepare data
    logger.info("Preparing training data...")
    X_train, y_train, X_val, y_val, X_test, y_test, feature_names, preprocessor = prepare_training_data(args.seasons)
    
    logger.info(f"\nDataset sizes:")
    logger.info(f"  Training:   {X_train.shape[0]} games, {X_train.shape[1]} features")
    logger.info(f"  Validation: {X_val.shape[0]} games")
    logger.info(f"  Test:       {X_test.shape[0]} games")
    
    # Save preprocessor
    preprocessor_path = Path(args.output_dir) / "preprocessor.pkl"
    import joblib
    joblib.dump(preprocessor, preprocessor_path)
    logger.info(f"Saved preprocessor to {preprocessor_path}")
    
    # Train models (with parallel training by default)
    train_all_models(
        X_train, y_train, X_val, y_val, X_test, y_test, 
        feature_names, args.output_dir, 
        parallel=not args.no_parallel
    )
    
    logger.info(f"\n{'='*60}")
    logger.info("Training complete!")
    logger.info(f"Models saved to: {args.output_dir}/")
    logger.info("="*60 + "\n")
    
    logger.info("Next steps:")
    logger.info("1. Set your ANTHROPIC_API_KEY in environment")
    logger.info("2. Run analysis: python -m src.cli analyze --help")
    logger.info("3. Run backtest: python -m src.cli backtest --help")


if __name__ == '__main__':
    main()

