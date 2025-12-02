"""Script to train all models on historical data.

This is a template script. Adapt it to your specific data sources and requirements.
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

from src.models.neural_nets.deep_predictor import DeepPredictor
from src.models.traditional.gradient_boost_model import GradientBoostModel
from src.models.traditional.random_forest_model import RandomForestModel
from src.models.traditional.statistical_model import StatisticalModel
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
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test, feature_names)
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
    
    # Convert to DataFrame
    features_df = pd.DataFrame(all_features)
    labels = np.array(all_labels)
    
    # Handle any remaining NaN values
    features_df = features_df.fillna(0)
    
    # Split by season to avoid data leakage
    features_df['season'] = [schedule_df.iloc[i]['season'] for i in range(len(all_features))]
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
    
    X_train = train_df[feature_cols].values
    y_train = train_df['label'].values
    X_val = val_df[feature_cols].values
    y_val = val_df['label'].values
    X_test = test_df[feature_cols].values
    y_test = test_df['label'].values
    
    logger.info(f"Training set: {X_train.shape}")
    logger.info(f"Validation set: {X_val.shape}")
    logger.info(f"Test set: {X_test.shape}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols


def train_all_models(X_train, y_train, X_val, y_val, X_test, y_test, feature_names, output_dir):
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
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize models
    models = [
        GradientBoostModel(name="Gradient Strategist"),
        RandomForestModel(name="Forest Evaluator"),
        StatisticalModel(name="Statistical Conservative"),
        # Neural network takes longer and requires more data, train separately if needed
        # DeepPredictor(name="Neural Analyst"),
    ]
    
    results = []
    
    # Train each model
    for model in models:
        logger.info(f"\n{'='*60}")
        logger.info(f"Training {model.name}...")
        logger.info('='*60)
        
        try:
            # Set feature names
            model.feature_names = feature_names
            
            # Train model
            model.train(X_train, y_train, X_val, y_val)
            
            # Evaluate on test set
            from sklearn.metrics import accuracy_score, precision_score, recall_score
            
            test_predictions = []
            for i in range(len(X_test)):
                pred = model.predict(X_test[i:i+1])
                test_predictions.append(1 if pred.prediction.value == "home_win" or pred.prediction.value == "HOME_WIN" else 0)
            
            test_accuracy = accuracy_score(y_test, test_predictions)
            test_precision = precision_score(y_test, test_predictions, zero_division=0)
            test_recall = recall_score(y_test, test_predictions, zero_division=0)
            
            logger.info(f"Test Accuracy: {test_accuracy:.3f}")
            logger.info(f"Test Precision: {test_precision:.3f}")
            logger.info(f"Test Recall: {test_recall:.3f}")
            
            # Save model
            model_file = output_path / f"{model.name.lower().replace(' ', '_')}.pkl"
            model.save(str(model_file))
            
            logger.info(f"Saved {model.name} to {model_file}")
            
            results.append({
                'model': model.name,
                'accuracy': test_accuracy,
                'precision': test_precision,
                'recall': test_recall
            })
        
        except Exception as e:
            logger.error(f"Error training {model.name}: {e}", exc_info=True)
    
    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("TRAINING SUMMARY")
    logger.info('='*60)
    
    for result in results:
        logger.info(f"\n{result['model']}:")
        logger.info(f"  Accuracy:  {result['accuracy']:.1%}")
        logger.info(f"  Precision: {result['precision']:.1%}")
        logger.info(f"  Recall:    {result['recall']:.1%}")


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(
        description="Train betting prediction models on historical NFL data"
    )
    parser.add_argument('--seasons', nargs='+', type=int, required=True,
                       help='Seasons to train on (e.g., 2020 2021 2022 2023)')
    parser.add_argument('--output-dir', default='models',
                       help='Directory to save trained models')
    
    args = parser.parse_args()
    
    logger.info("\n" + "="*60)
    logger.info("NFL BETTING AGENT COUNCIL - MODEL TRAINING")
    logger.info("="*60)
    logger.info(f"Seasons: {args.seasons}")
    logger.info(f"Output directory: {args.output_dir}\n")
    
    # Prepare data
    logger.info("Preparing training data...")
    X_train, y_train, X_val, y_val, X_test, y_test, feature_names = prepare_training_data(args.seasons)
    
    logger.info(f"\nDataset sizes:")
    logger.info(f"  Training:   {X_train.shape[0]} games, {X_train.shape[1]} features")
    logger.info(f"  Validation: {X_val.shape[0]} games")
    logger.info(f"  Test:       {X_test.shape[0]} games")
    
    # Train models
    train_all_models(X_train, y_train, X_val, y_val, X_test, y_test, feature_names, args.output_dir)
    
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

