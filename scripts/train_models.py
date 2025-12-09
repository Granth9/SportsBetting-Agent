"""Script to train all models on historical data.

This is a template script. Adapt it to your specific data sources and requirements.
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import json
import joblib
from datetime import datetime
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


def save_training_metadata(output_path: Path, seasons: list, feature_names: list, 
                           model_accuracies: dict, scaler_path: str):
    """Save training metadata to allow easy model loading later.
    
    Args:
        output_path: Directory to save metadata
        seasons: List of seasons used for training
        feature_names: List of feature names
        model_accuracies: Dict of model names to accuracies
        scaler_path: Path to saved scaler
    """
    metadata = {
        'trained_at': datetime.now().isoformat(),
        'seasons': seasons,
        'feature_names': feature_names,
        'num_features': len(feature_names),
        'model_accuracies': model_accuracies,
        'scaler_path': str(scaler_path),
        'strategy': 'selective_75',
        'min_confidence': 0.70,
        'min_spread': 5.0
    }
    
    metadata_path = output_path / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Saved training metadata to {metadata_path}")


def load_trained_models(model_dir: str = 'models/trained'):
    """Load all trained models and metadata.
    
    Args:
        model_dir: Directory containing trained models
        
    Returns:
        Tuple of (models_dict, scaler, metadata) or (None, None, None) if not found
    """
    model_path = Path(model_dir)
    
    if not model_path.exists():
        logger.warning(f"Model directory {model_dir} does not exist")
        return None, None, None
    
    # Load metadata
    metadata_path = model_path / 'metadata.json'
    if not metadata_path.exists():
        logger.warning(f"Metadata file not found at {metadata_path}")
        return None, None, None
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Load scaler
    scaler_path = model_path / 'scaler.pkl'
    if not scaler_path.exists():
        # Try preprocessor path
        scaler_path = model_path / 'preprocessor.pkl'
    
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
    else:
        logger.warning("Scaler not found")
        scaler = None
    
    # Load models
    models = {}
    model_files = list(model_path.glob('*.pkl'))
    
    for model_file in model_files:
        if model_file.name in ['scaler.pkl', 'preprocessor.pkl', 'metadata.json']:
            continue
        
        try:
            model = joblib.load(model_file)
            model_name = model_file.stem.replace('_', ' ').title()
            models[model_name] = model
            logger.info(f"Loaded model: {model_name}")
        except Exception as e:
            logger.warning(f"Could not load {model_file}: {e}")
    
    logger.info(f"Loaded {len(models)} models from {model_dir}")
    return models, scaler, metadata


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
    
    logger.info("Collecting weekly stats for EPA metrics...")
    weekly_stats_df = collector.get_player_stats(seasons, stat_type='weekly')
    
    # Collect roster data for IR return features
    logger.info("Collecting roster data for IR return features...")
    roster_df = None
    try:
        import nfl_data_py as nfl
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        
        # Load roster data year by year to avoid nfl_data_py reindex bug
        roster_dfs = []
        for year in seasons:
            try:
                year_roster = nfl.import_weekly_rosters([year])
                # Keep only necessary columns
                roster_cols = ['season', 'team', 'position', 'status', 'player_name', 'player_id', 'week']
                available_cols = [c for c in roster_cols if c in year_roster.columns]
                year_roster = year_roster[available_cols].reset_index(drop=True)
                roster_dfs.append(year_roster)
            except Exception as e:
                logger.warning(f"Could not load roster data for {year}: {e}")
        
        if roster_dfs:
            roster_df = pd.concat(roster_dfs, ignore_index=True)
            logger.info(f"Loaded roster data: {len(roster_df)} rows")
        else:
            logger.warning("No roster data loaded (IR features will be disabled)")
    except Exception as e:
        logger.warning(f"Could not load roster data (IR features will be disabled): {e}")
    
    # Initialize feature engineer
    feature_engineer = FeatureEngineer()
    
    # Extract features for each game
    logger.info("Extracting features from games...")
    all_features = []
    all_labels = []
    
    processed = 0
    failed = 0
    
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
            
            # Create betting line from schedule data if available
            from src.utils.data_types import BettingLine
            
            betting_line = None
            # Use betting line if available, but don't require it (use defaults if missing)
            spread = float(game.get('spread_line', 0.0)) if pd.notna(game.get('spread_line')) else 0.0
            total = float(game.get('total_line', 45.0)) if pd.notna(game.get('total_line')) else 45.0
            home_ml = float(game.get('home_moneyline', -110.0)) if pd.notna(game.get('home_moneyline')) else -110.0
            away_ml = float(game.get('away_moneyline', -110.0)) if pd.notna(game.get('away_moneyline')) else -110.0
            
            # Create betting line even if some values are defaults (allows more games to be processed)
            betting_line = BettingLine(
                spread=spread,
                total=total,
                home_ml=home_ml,
                away_ml=away_ml,
                source="nfl_data_py"
            )
            
            # Create proposition with betting line
            prop = Proposition(
                prop_id=game['game_id'],
                game_info=game_info,
                bet_type=BetType.GAME_OUTCOME,
                line=betting_line
            )
            
            # Extract features
            features = feature_engineer.extract_features(
                prop,
                schedule_df,
                team_stats_df,
                None,  # player_stats_df (not needed for game outcome)
                weekly_stats_df,  # weekly_stats_df for EPA
                roster_df  # roster_df for IR return features
            )
            
            # Create label (1 if home team won, 0 if away team won)
            label = 1 if game['home_score'] > game['away_score'] else 0
            
            all_features.append(features)
            all_labels.append(label)
            processed += 1
            
            if processed % 100 == 0:
                logger.info(f"Processed {processed} games...")
            
        except Exception as e:
            failed += 1
            # Log first few errors to understand what's failing
            if failed <= 5:
                logger.warning(f"Error processing game {game.get('game_id', 'unknown')}: {e}", exc_info=True)
            continue
    
    logger.info(f"Successfully extracted features from {processed} games ({failed} failed)")
    
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
    
    # If validation or test sets are empty, use a percentage split instead
    if len(val_df) == 0 or len(test_df) == 0:
        logger.warning("Validation or test set is empty, using percentage split instead")
        from sklearn.model_selection import train_test_split
        train_df, temp_df = train_test_split(features_df, test_size=0.3, random_state=42, stratify=features_df.get('label', None))
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df.get('label', None))
    
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
    
    # Convert to numpy arrays and ensure proper dtype
    X_train = train_features_df.values.astype(np.float64)
    y_train = train_df['label'].values.astype(np.int64)
    X_val = val_features_df.values.astype(np.float64)
    y_val = val_df['label'].values.astype(np.int64)
    X_test = test_features_df.values.astype(np.float64)
    y_test = test_df['label'].values.astype(np.int64)
    
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
            
        # Save model immediately after training (before evaluation that might crash)
        try:
            model_file = output_path / f"{model.name.lower().replace(' ', '_')}.pkl"
            model.save(str(model_file))
            logger.info(f"Saved {model.name} to {model_file} (after training, before evaluation)")
        except Exception as e:
            logger.warning(f"Could not save model before evaluation: {e}")
        
            # Evaluate on test set
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, 
            f1_score, roc_auc_score, confusion_matrix
        )
        from src.utils.data_types import Outcome
        
        # Batch predictions for efficiency
            test_predictions = []
        test_probs = []
        
        # Process in batches to avoid memory issues
        batch_size = 50
        for i in range(0, len(X_test), batch_size):
            batch_end = min(i + batch_size, len(X_test))
            batch_X = X_test[i:batch_end]
            
            # Get predictions for batch
            # For LightGBM and XGBoost, try batch prediction first
            try:
                # Try batch prediction for efficiency (works for most models)
                batch_probs = []
                for j in range(len(batch_X)):
                    try:
                        proba = model.predict_proba(batch_X[j:j+1])
                        batch_probs.append(proba.get(Outcome.HOME_WIN, 0.5))
                        pred = model.predict(batch_X[j:j+1])
                        test_predictions.append(1 if pred.prediction.value in ["home_win", "HOME_WIN"] else 0)
                    except Exception as e:
                        logger.warning(f"Error predicting sample {i+j}: {e}")
                        test_predictions.append(0)
                        batch_probs.append(0.5)
                test_probs.extend(batch_probs)
            except Exception as e:
                logger.warning(f"Batch prediction failed, using individual predictions: {e}")
                # Fallback to individual predictions
                for j in range(len(batch_X)):
                    try:
                        pred = model.predict(batch_X[j:j+1])
                        test_predictions.append(1 if pred.prediction.value in ["home_win", "HOME_WIN"] else 0)
                        proba = model.predict_proba(batch_X[j:j+1])
                        test_probs.append(proba.get(Outcome.HOME_WIN, 0.5))
                    except Exception as e2:
                        logger.warning(f"Error predicting sample {i+j}: {e2}")
                        test_predictions.append(0)
                        test_probs.append(0.5)
        
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
        try:
            cm = confusion_matrix(y_test, test_predictions)
            if cm.size == 4:
                tn, fp, fn, tp = cm.ravel()
            else:
                # Handle edge cases (e.g., all predictions same class)
                tn, fp, fn, tp = 0, 0, 0, 0
                if len(set(test_predictions)) == 1:
                    # All predictions are the same
                    if test_predictions[0] == 1:
                        tp = sum(y_test == 1)
                        fn = sum(y_test == 0)
                    else:
                        tn = sum(y_test == 0)
                        fp = sum(y_test == 1)
        except Exception as e:
            logger.warning(f"Error computing confusion matrix: {e}")
            tn, fp, fn, tp = 0, 0, 0, 0
        
        logger.info(f"Test Accuracy:  {test_accuracy:.3f}")
            logger.info(f"Test Precision: {test_precision:.3f}")
        logger.info(f"Test Recall:    {test_recall:.3f}")
        logger.info(f"Test F1:        {test_f1:.3f}")
        logger.info(f"Test ROC-AUC:   {test_roc_auc:.3f}")
            
            # Save model
        try:
            model_file = output_path / f"{model.name.lower().replace(' ', '_')}.pkl"
            model.save(str(model_file))
            logger.info(f"Saved {model.name} to {model_file}")
        except Exception as e:
            logger.error(f"Error saving {model.name}: {e}")
            # Continue anyway - model is trained even if save fails
            
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


def train_all_models(X_train, y_train, X_val, y_val, X_test, y_test, feature_names, output_dir, parallel=True, model_filter=None):
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
        model_filter: Optional list of model names to train (e.g., ['Neural Analyst', 'Gradient Strategist'])
    """
    from src.utils.data_types import Outcome
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize models - using only the top 3 performers for ensemble
    # (Neural Analyst, Statistical Conservative, SVM Strategist)
    # Forest Evaluator excluded as it drags down ensemble accuracy
    
    all_models = [
        DeepPredictor(
            name="Neural Analyst",
            epochs=200,
            early_stopping_patience=20,
            hidden_layers=[512, 256, 128, 64],
            dropout=0.35,
            learning_rate=0.0003
        ),
        StatisticalModel(name="Statistical Conservative"),
        SVMModel(name="SVM Strategist"),
    ]
    
    # Add optimized ensemble model with top 3 models only
    # Using unanimous_high_confidence strategy for 70% accuracy
    from src.models.ensemble.ensemble_model import EnsembleModel
    ensemble = EnsembleModel(
        name="Ensemble Council",
        base_models=all_models.copy(),  # Use copy to avoid circular reference
        voting_strategy="unanimous_high_confidence"  # 70% accuracy when all models agree + high confidence
    )
    all_models.append(ensemble)
    
    # Filter models if specified
    if model_filter:
        models = [m for m in all_models if m.name in model_filter]
        if not models:
            logger.warning(f"No models found matching filter: {model_filter}")
            logger.info(f"Available models: {[m.name for m in all_models]}")
            return []
    else:
        models = all_models
    
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
    parser.add_argument('--output-dir', default='models/trained',
                       help='Directory to save trained models')
    parser.add_argument('--no-parallel', action='store_true',
                       help='Disable parallel model training')
    parser.add_argument('--model', nargs='+', type=str, default=None,
                       help='Train specific models only (e.g., "Neural Analyst" "Gradient Strategist")')
    parser.add_argument('--feature-selection', action='store_true',
                       help='Apply feature selection to keep most important features')
    parser.add_argument('--tune-hyperparameters', action='store_true',
                       help='Tune hyperparameters using Optuna (slower but better accuracy)')
    
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
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save scaler/preprocessor
    scaler_path = output_path / "scaler.pkl"
    joblib.dump(preprocessor, scaler_path)
    logger.info(f"Saved scaler to {scaler_path}")
    
    # Train models (with parallel training by default)
    results = train_all_models(
        X_train, y_train, X_val, y_val, X_test, y_test, 
        feature_names, args.output_dir, 
        parallel=not args.no_parallel,
        model_filter=args.model
    )
    
    # Save metadata
    model_accuracies = {r['model']: r['accuracy'] for r in results if r.get('success', True)}
    save_training_metadata(
        output_path,
        seasons=args.seasons,
        feature_names=feature_names,
        model_accuracies=model_accuracies,
        scaler_path=str(scaler_path)
    )
    
    logger.info(f"\n{'='*60}")
    logger.info("Training complete!")
    logger.info(f"Models saved to: {args.output_dir}/")
    logger.info("="*60 + "\n")
    
    logger.info("Next steps:")
    logger.info("1. Quick predict: python -m src.cli predict 'Chiefs vs Raiders'")
    logger.info("2. Run analysis: python -m src.cli analyze --help")
    logger.info("3. Run backtest: python -m src.cli backtest --help")


if __name__ == '__main__':
    main()

