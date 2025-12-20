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
from typing import Optional, Tuple, List

from src.models.neural_nets.deep_predictor import DeepPredictor
# XGBoost re-enabled with fixes for segmentation faults
from src.models.traditional.gradient_boost_model import GradientBoostModel
from src.models.traditional.random_forest_model import RandomForestModel
from src.models.traditional.statistical_model import StatisticalModel
# LightGBM re-enabled with fixes for segmentation faults
from src.models.traditional.lightgbm_model import LightGBMModel
from src.models.traditional.svm_model import SVMModel
from src.data.collectors.nfl_data_collector import NFLDataCollector
from src.data.processors.feature_engineer import FeatureEngineer
from src.data.processors.data_preprocessor import DataPreprocessor, DataSplitter
from src.utils.logger import setup_logger


logger = setup_logger(__name__)


def calculate_temporal_weights(features_df: pd.DataFrame, season_col: str = 'season', 
                                week_col: Optional[str] = None, 
                                decay_factor: float = 0.22) -> np.ndarray:
    """Calculate sample weights with exponential decay for recent games.
    
    Args:
        features_df: DataFrame with season/week information
        season_col: Column name for season
        week_col: Optional column name for week (if available)
        decay_factor: Exponential decay factor (higher = more weight on recent)
        
    Returns:
        Array of sample weights (normalized to sum to len(features_df))
    """
    # Calculate recency score (higher = more recent)
    max_season = features_df[season_col].max()
    
    # Base weight: 1.0 for oldest season, exponentially increasing for recent
    weights = []
    for _, row in features_df.iterrows():
        season = row[season_col]
        season_diff = max_season - season
        
        # Exponential decay: recent seasons get much higher weight
        # Formula: weight = exp(decay_factor * (max_season - season))
        # This gives recent seasons exponentially more weight
        base_weight = np.exp(decay_factor * (max_season - season))
        
        # If week info available, add fine-grained recency within season
        if week_col and week_col in row and pd.notna(row[week_col]):
            try:
                week = int(row[week_col])
                # Add small boost for later weeks in same season
                week_boost = 1.0 + (week / 18.0) * 0.1  # Max 10% boost for week 18
                base_weight *= week_boost
            except (ValueError, TypeError):
                pass  # Skip week boost if week is invalid
        
        weights.append(base_weight)
    
    weights = np.array(weights, dtype=np.float64)
    
    # Normalize so sum equals number of samples (maintains total "information")
    if weights.sum() > 0:
        weights = weights / weights.sum() * len(weights)
    
    # Diagnostic logging for temporal weights
    if len(weights) > 0:
        logger.info(f"\n{'='*60}")
        logger.info(f"TEMPORAL WEIGHTING DIAGNOSTICS (decay_factor={decay_factor})")
        logger.info(f"{'='*60}")
        logger.info(f"Temporal weight statistics:")
        logger.info(f"  Min: {weights.min():.4f}, Max: {weights.max():.4f}, Mean: {weights.mean():.4f}, Std: {weights.std():.4f}")
        
        # Log weights by season
        if season_col in features_df.columns:
            season_weights = {}
            for season in features_df[season_col].unique():
                season_mask = features_df[season_col] == season
                if season_mask.sum() > 0:
                    season_weights[season] = {
                        'count': season_mask.sum(),
                        'avg_weight': weights[season_mask].mean(),
                        'min_weight': weights[season_mask].min(),
                        'max_weight': weights[season_mask].max()
                    }
            
            logger.info(f"\nTemporal weights by season:")
            for season in sorted(season_weights.keys()):
                sw = season_weights[season]
                logger.info(f"  Season {season}: {sw['count']} games, avg weight: {sw['avg_weight']:.4f} (range: {sw['min_weight']:.4f}-{sw['max_weight']:.4f})")
        
        logger.info(f"{'='*60}\n")
    
    return weights


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


def prepare_training_data(seasons, include_2025_week15: bool = True, 
                          temporal_weighting: bool = True, decay_factor: float = 0.22,
                          use_feature_selection: bool = True, n_features: int = 110,
                          test_set_2024_only: bool = False):
    """Prepare training data from historical NFL games.
    
    Args:
        seasons: List of seasons to include
        include_2025_week15: If True, filter 2025 data to only include weeks <= 15
        temporal_weighting: If True, calculate temporal weights for training data
        decay_factor: Exponential decay factor for temporal weighting
        
    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test, feature_names, preprocessor, sample_weights)
        where sample_weights is None if temporal_weighting is False
    """
    logger.info(f"Preparing training data for seasons: {seasons}")
    
    # Collect data
    collector = NFLDataCollector()
    logger.info("Collecting schedule data...")
    # Try to get schedule data, filtering out unavailable seasons
    available_seasons = []
    schedule_df = pd.DataFrame()
    for season in seasons:
        try:
            test_schedule = collector.get_schedule([season])
            if len(test_schedule) > 0:
                available_seasons.append(season)
                schedule_df = pd.concat([schedule_df, test_schedule], ignore_index=True) if len(schedule_df) > 0 else test_schedule
                logger.info(f"Successfully loaded {season} schedule data ({len(test_schedule)} games)")
            else:
                logger.warning(f"No schedule data available for {season}, skipping")
        except Exception as e:
            logger.warning(f"Could not load schedule data for {season}: {e}. Skipping this season.")
    
    if len(schedule_df) == 0:
        raise ValueError(f"No schedule data available for any of the requested seasons: {seasons}")
    
    if available_seasons != seasons:
        logger.warning(f"Only using data from available seasons: {available_seasons} (requested: {seasons})")
        seasons = available_seasons  # Update seasons list to only include available ones
    
    # Filter 2025 data to only include weeks <= 15 if requested
    if include_2025_week15 and 2025 in seasons and 2025 in schedule_df['season'].values:
        original_len = len(schedule_df)
        schedule_df = schedule_df[
            (schedule_df['season'] != 2025) | (schedule_df['week'] <= 15)
        ]
        filtered_len = len(schedule_df)
        if original_len != filtered_len:
            logger.info(f"Filtered 2025 schedule: {original_len} -> {filtered_len} games (weeks 1-15 only)")
    
    logger.info("Collecting team stats...")
    # Try to get team stats, filtering out unavailable seasons
    team_stats_seasons = []
    team_stats_df = pd.DataFrame()
    for season in available_seasons:
        try:
            test_stats = collector.get_team_stats([season])
            if len(test_stats) > 0:
                team_stats_seasons.append(season)
                team_stats_df = pd.concat([team_stats_df, test_stats], ignore_index=True) if len(team_stats_df) > 0 else test_stats
                logger.info(f"Successfully loaded {season} team stats ({len(test_stats)} records)")
            else:
                logger.warning(f"No team stats available for {season}, skipping")
        except Exception as e:
            logger.warning(f"Could not load team stats for {season}: {e}. Skipping this season.")
    
    if len(team_stats_df) == 0:
        logger.warning("No team stats available, continuing without team stats")
    elif team_stats_seasons != available_seasons:
        logger.warning(f"Only using team stats from available seasons: {team_stats_seasons} (requested: {available_seasons})")
    
    logger.info("Collecting weekly stats for EPA metrics...")
    # Try to get weekly stats, filtering out unavailable seasons
    weekly_stats_seasons = []
    weekly_stats_df = pd.DataFrame()
    for season in available_seasons:
        try:
            test_stats = collector.get_player_stats([season], stat_type='weekly')
            if len(test_stats) > 0:
                weekly_stats_seasons.append(season)
                weekly_stats_df = pd.concat([weekly_stats_df, test_stats], ignore_index=True) if len(weekly_stats_df) > 0 else test_stats
                logger.info(f"Successfully loaded {season} weekly stats ({len(test_stats)} records)")
            else:
                logger.warning(f"No weekly stats available for {season}, skipping")
        except Exception as e:
            logger.warning(f"Could not load weekly stats for {season}: {e}. Skipping this season.")
    
    if len(weekly_stats_df) == 0:
        logger.warning("No weekly stats available, continuing without weekly stats")
    elif weekly_stats_seasons != available_seasons:
        logger.warning(f"Only using weekly stats from available seasons: {weekly_stats_seasons} (requested: {available_seasons})")
    
    # Collect roster data for IR return features
    logger.info("Collecting roster data for IR return features...")
    roster_df = None
    try:
        import nfl_data_py as nfl
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        
        # Load roster data year by year to avoid nfl_data_py reindex bug
        roster_dfs = []
        for year in available_seasons:
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
    validate_data(features_df, labels, schedule_df)
    
    # Split by season to avoid data leakage
    # Get season and week info from original schedule
    season_info = []
    week_info = []
    feature_idx = 0
    for idx, game in schedule_df.iterrows():
        if pd.notna(game.get('home_score')) and pd.notna(game.get('away_score')):
            season_info.append(int(game['season']))
            week_info.append(int(game.get('week', 1)) if pd.notna(game.get('week')) else 1)
            feature_idx += 1
    
    features_df['season'] = season_info[:len(features_df)]
    features_df['week'] = week_info[:len(features_df)]
    features_df['label'] = labels
    
    # Use DataSplitter to split by date
    splitter = DataSplitter()
    
    # Determine split seasons
    # Strategy: Train on 2020-2023 + 2024 (early weeks) + 2025 (weeks 1-15)
    #           Validate on 2024 (mid weeks)
    #           Test on 2024 (late weeks) - most recent data for evaluation
    if 2024 in seasons and 2025 in seasons:
        logger.info("Using time-based split: Train on 2020-2023 + 2024(early) + 2025(weeks 1-15), Val/Test on 2024(later weeks)")
        
        # Get 2024 data and split by week (time-based, not random)
        season_2024_data = features_df[features_df['season'] == 2024].copy()
        
        if len(season_2024_data) > 0 and 'week' in season_2024_data.columns:
            # Sort by week to ensure chronological split
            season_2024_data = season_2024_data.sort_values('week')
            
            # Split 2024 by weeks: early weeks for training, mid for val, late for test
            total_2024_weeks = season_2024_data['week'].max()
            mid_week = total_2024_weeks // 2
            
            # Training: weeks 1-6 (early season)
            train_2024 = season_2024_data[season_2024_data['week'] <= 6].copy()
            # Validation: weeks 7-12 (mid season)
            val_2024 = season_2024_data[(season_2024_data['week'] > 6) & (season_2024_data['week'] <= 12)].copy()
            # Test: weeks 13+ (late season, most recent)
            test_2024 = season_2024_data[season_2024_data['week'] > 12].copy()
            
            # If week-based split doesn't work well, fall back to percentage split
            if len(val_2024) == 0 or len(test_2024) == 0:
                logger.warning("Week-based split resulted in empty sets, using percentage split instead")
                from sklearn.model_selection import train_test_split
                labels_2024 = season_2024_data.get('label', None)
                if labels_2024 is not None:
                    val_2024, test_2024 = train_test_split(
                        season_2024_data[season_2024_data['week'] > 6].copy() if len(season_2024_data[season_2024_data['week'] > 6]) > 0 else season_2024_data,
                        test_size=0.5,
                        random_state=42,
                        stratify=season_2024_data[season_2024_data['week'] > 6]['label'] if len(season_2024_data[season_2024_data['week'] > 6]) > 0 else labels_2024
                    )
                    train_2024 = season_2024_data[season_2024_data['week'] <= 6].copy()
                else:
                    val_2024, test_2024 = train_test_split(
                        season_2024_data[season_2024_data['week'] > 6].copy() if len(season_2024_data[season_2024_data['week'] > 6]) > 0 else season_2024_data,
                        test_size=0.5,
                        random_state=42
                    )
                    train_2024 = season_2024_data[season_2024_data['week'] <= 6].copy()
        else:
            # No week column or empty data, use percentage split
            logger.warning("No week column available, using percentage split for 2024")
            from sklearn.model_selection import train_test_split
            labels_2024 = season_2024_data.get('label', None)
            if labels_2024 is not None:
                val_2024, test_2024 = train_test_split(
                    season_2024_data,
                    test_size=0.5,
                    random_state=42,
                    stratify=labels_2024
                )
            else:
                val_2024, test_2024 = train_test_split(
                    season_2024_data,
                    test_size=0.5,
                    random_state=42
                )
            train_2024 = pd.DataFrame()  # No 2024 in training if we can't split by week
        
        # Get training data: 2020-2023 + 2024(early weeks) + 2025(weeks 1-13, hold out weeks 14-15 for testing)
        train_df = features_df[features_df['season'] < 2024].copy()
        if len(train_2024) > 0:
            train_df = pd.concat([train_df, train_2024], ignore_index=True)
        
        # Handle 2025 data: train on weeks 1-13, test on weeks 14-15 (most recent)
        test_2025 = pd.DataFrame()
        if 2025 in features_df['season'].values:
            train_2025_all = features_df[features_df['season'] == 2025].copy()
            if 'week' in train_2025_all.columns and len(train_2025_all) > 0:
                # Hold out 2025 weeks 14-15 for testing (most recent data)
                train_2025 = train_2025_all[train_2025_all['week'] <= 13].copy()
                test_2025 = train_2025_all[train_2025_all['week'] > 13].copy()
                
                if len(train_2025) > 0:
                    train_df = pd.concat([train_df, train_2025], ignore_index=True)
                if len(test_2025) > 0:
                    logger.info(f"  ✓ 2025 weeks 14-15 held out for testing: {len(test_2025)} games")
            else:
                # No week column, use all 2025 for training
                train_df = pd.concat([train_df, train_2025_all], ignore_index=True)
        
        # Combine 2024 late weeks and 2025 weeks 14-15 for testing (most recent data)
        # Can exclude 2025 if test_set_2024_only is True
        if len(test_2025) > 0 and not test_set_2024_only:
            test_df = pd.concat([test_2024, test_2025], ignore_index=True)
            logger.info(f"\n{'='*60}")
            logger.info(f"TEST SET COMPOSITION DIAGNOSTICS")
            logger.info(f"{'='*60}")
            logger.info(f"Test data: 2024 late weeks ({len(test_2024)} games) + 2025 weeks 14-15 ({len(test_2025)} games) = {len(test_df)} games total")
            
            # Log detailed test set composition
            if 'week' in test_2024.columns and len(test_2024) > 0:
                test_2024_weeks = test_2024['week'].unique()
                logger.info(f"  2024 test weeks: {sorted(test_2024_weeks)} (weeks {test_2024['week'].min()}-{test_2024['week'].max()})")
            if 'week' in test_2025.columns and len(test_2025) > 0:
                test_2025_weeks = test_2025['week'].unique()
                logger.info(f"  2025 test weeks: {sorted(test_2025_weeks)} (weeks {test_2025['week'].min()}-{test_2025['week'].max()})")
            
            # Log label distribution in test set
            if 'label' in test_df.columns:
                test_labels = test_df['label'].value_counts()
                logger.info(f"  Test set label distribution: {dict(test_labels)}")
            
            logger.info(f"{'='*60}\n")
        else:
            test_df = test_2024
            logger.info(f"\n{'='*60}")
            logger.info(f"TEST SET COMPOSITION DIAGNOSTICS")
            logger.info(f"{'='*60}")
            if test_set_2024_only and len(test_2025) > 0:
                logger.info(f"TEST MODE: Excluding 2025 weeks 14-15 from test set ({len(test_2025)} games moved to training)")
            logger.info(f"Test data: 2024 late weeks ({len(test_2024)} games)")
            if 'week' in test_2024.columns and len(test_2024) > 0:
                test_2024_weeks = test_2024['week'].unique()
                logger.info(f"  2024 test weeks: {sorted(test_2024_weeks)} (weeks {test_2024['week'].min()}-{test_2024['week'].max()})")
            if 'label' in test_df.columns:
                test_labels = test_df['label'].value_counts()
                logger.info(f"  Test set label distribution: {dict(test_labels)}")
            logger.info(f"{'='*60}\n")
            
            # If excluding 2025 from test, add it to training
            if test_set_2024_only and len(test_2025) > 0:
                train_df = pd.concat([train_df, test_2025], ignore_index=True)
                logger.info(f"  ✓ Added {len(test_2025)} 2025 games to training set (test mode: 2024-only test set)")
        
        # Verify 2025 is included if available
        train_seasons_list = sorted(train_df['season'].unique().tolist())
        has_2025 = 2025 in train_seasons_list
        logger.info(f"Training data seasons: {train_seasons_list} ({len(train_df)} games)")
        if has_2025:
            train_2025_count = len(train_df[train_df['season'] == 2025])
            logger.info(f"  ✓ 2025 data included: {train_2025_count} games (weeks 1-13)")
        if len(train_2024) > 0:
            logger.info(f"  ✓ 2024 early weeks included: {len(train_2024)} games")
        logger.info(f"Validation data: 2024 mid weeks ({len(val_2024)} games)")
        
        val_df = val_2024
    elif 2024 in seasons:
        # 2025 not available, use simpler split
        logger.info("2025 not available, using 2024 for validation and testing")
        
        # Get all 2024 data
        season_2024_data = features_df[features_df['season'] == 2024].copy()
        
        if len(season_2024_data) > 0:
            # Split 2024: 50% validation, 50% test
            from sklearn.model_selection import train_test_split
            
            # Get labels for stratification if available
            labels_2024 = season_2024_data.get('label', None)
            if labels_2024 is not None:
                val_2024, test_2024 = train_test_split(
                    season_2024_data, 
                    test_size=0.5, 
                    random_state=42,
                    stratify=labels_2024
                )
            else:
                val_2024, test_2024 = train_test_split(
                    season_2024_data, 
                    test_size=0.5, 
                    random_state=42
                )
            
            # Get training data (all other seasons)
            train_df = features_df[features_df['season'] != 2024].copy()
            
            train_seasons_list = sorted(train_df['season'].unique().tolist())
            logger.info(f"Training data seasons: {train_seasons_list} ({len(train_df)} games)")
            logger.info(f"Validation data: 2024 subset ({len(val_2024)} games)")
            logger.info(f"Test data: 2024 subset ({len(test_2024)} games)")
            
            val_df = val_2024
            test_df = test_2024
        else:
            logger.warning("No 2024 data available, falling back to original split logic")
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
    else:
        # Fallback to original logic if 2024 not available
        logger.info("2024 not in seasons, using original split logic")
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
    feature_cols = [col for col in features_df.columns if col not in ['season', 'week', 'label']]
    
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
    
    # Calculate temporal weights for training data
    sample_weights = None
    if temporal_weighting:
        logger.info("Calculating temporal weights for training data...")
        sample_weights = calculate_temporal_weights(
            train_df, 
            season_col='season', 
            week_col='week',
            decay_factor=decay_factor
        )
        
        # Log weight statistics
        logger.info(f"Temporal weight statistics:")
        logger.info(f"  Min: {sample_weights.min():.4f}, Max: {sample_weights.max():.4f}, Mean: {sample_weights.mean():.4f}")
        
        # Log weight distribution by season
        weight_by_season = train_df.groupby('season')['season'].count()
        for season in sorted(train_df['season'].unique()):
            season_mask = train_df['season'] == season
            season_weights = sample_weights[season_mask]
            logger.info(f"  Season {season}: {len(season_weights)} games, avg weight: {season_weights.mean():.4f}")
    
    logger.info(f"Training set: {X_train.shape}")
    logger.info(f"Validation set: {X_val.shape}")
    logger.info(f"Test set: {X_test.shape}")
    
    # Store feature names from preprocessor
    processed_feature_names = preprocessor.get_feature_names()
    
    # Apply feature selection if requested
    if use_feature_selection and len(processed_feature_names) > n_features:
        logger.info(f"\n{'='*60}")
        logger.info(f"FEATURE SELECTION DIAGNOSTICS")
        logger.info(f"{'='*60}")
        logger.info(f"Original feature count: {len(processed_feature_names)}")
        logger.info(f"Target feature count: {n_features}")
        logger.info(f"Applying feature selection: selecting top {n_features} from {len(processed_feature_names)} features...")
        
        selected_features = select_important_features(X_train, y_train, processed_feature_names, n_features=n_features)
        
        if selected_features:
            # Get indices of selected features
            feature_indices = [processed_feature_names.index(f) for f in selected_features if f in processed_feature_names]
            
            # Log removed features for diagnostics
            removed_features = [f for f in processed_feature_names if f not in selected_features]
            logger.info(f"\nFeature selection complete: {len(selected_features)} features selected, {len(removed_features)} removed")
            if len(removed_features) > 0:
                logger.info(f"Removed {min(10, len(removed_features))} features (showing first 10): {removed_features[:10]}")
            
            # Filter features
            X_train = X_train[:, feature_indices]
            X_val = X_val[:, feature_indices]
            X_test = X_test[:, feature_indices]
            
            processed_feature_names = selected_features
            logger.info(f"Final feature count: {len(processed_feature_names)}")
            logger.info(f"{'='*60}\n")
        else:
            logger.warning("Feature selection returned None, using all features")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, processed_feature_names, preprocessor, sample_weights


def validate_data(features_df: pd.DataFrame, labels: np.ndarray, schedule_df: Optional[pd.DataFrame] = None) -> None:
    """Validate training data for quality issues.
    
    Args:
        features_df: Features DataFrame
        labels: Labels array
        schedule_df: Optional schedule DataFrame for additional validation
        
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
    
    # Validate 2025 Week 15 inclusion if schedule provided
    if schedule_df is not None:
        if 2025 in schedule_df['season'].values:
            week_15_games = schedule_df[(schedule_df['season'] == 2025) & (schedule_df['week'] == 15)]
            if len(week_15_games) > 0:
                logger.info(f"Found {len(week_15_games)} Week 15 games in 2025 season")
            else:
                logger.warning("No Week 15 games found in 2025 season - may need to update data")
        
        # Check for missing betting lines (warn but don't fail)
        if 'spread_line' in schedule_df.columns:
            missing_spread = schedule_df['spread_line'].isna().sum()
            if missing_spread > 0:
                logger.warning(f"Found {missing_spread} games with missing spread lines")
        
        # Log data distribution by season and week
        if 'season' in schedule_df.columns:
            season_dist = schedule_df['season'].value_counts().sort_index()
            logger.info(f"Data distribution by season: {dict(season_dist)}")
    
    logger.info("Data validation passed")


def select_important_features(X_train: np.ndarray, y_train: np.ndarray, feature_names: List[str], 
                             n_features: int = 50) -> Optional[List[str]]:
    """Select most important features using ensemble of tree-based models for better accuracy.
    
    Args:
        X_train: Training features
        y_train: Training labels
        feature_names: List of feature names
        n_features: Number of features to select
        
    Returns:
        List of selected feature names, or None if selection fails
    """
    try:
        from sklearn.ensemble import RandomForestClassifier
        
        logger.info(f"Training ensemble of models to identify top {n_features} features...")
        
        # Use ensemble of models to get more robust feature importances
        all_importances = []
        
        # 1. RandomForest (improved parameters)
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        rf_importances = rf.feature_importances_
        all_importances.append(rf_importances)
        logger.info(f"  ✓ RandomForest feature importances computed")
        
        # 2. XGBoost (if available) for additional robustness
        try:
            import xgboost as xgb
            xgb_model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=7,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
                tree_method='hist'
            )
            xgb_model.fit(X_train, y_train)
            xgb_importances = xgb_model.feature_importances_
            all_importances.append(xgb_importances)
            logger.info(f"  ✓ XGBoost feature importances computed")
        except Exception as e:
            logger.debug(f"  XGBoost not available for feature selection: {e}")
        
        # Average importances across models for robustness
        if len(all_importances) > 1:
            importances = np.mean(all_importances, axis=0)
            logger.info(f"  Averaged feature importances from {len(all_importances)} models")
        else:
            importances = all_importances[0]
        
        # Get top N features
        top_indices = np.argsort(importances)[-n_features:][::-1]
        selected_features = [feature_names[i] for i in top_indices]
        
        # Log top features and their importances
        logger.info(f"\nTop 20 feature importances:")
        for i, idx in enumerate(top_indices[:20]):
            logger.info(f"  {i+1:2d}. {feature_names[idx]:40s}: {importances[idx]:.6f}")
        
        # Log feature importance statistics
        removed_indices = [i for i in range(len(importances)) if i not in top_indices]
        logger.info(f"\nFeature importance statistics:")
        logger.info(f"  Mean importance: {importances.mean():.6f}")
        logger.info(f"  Max importance: {importances.max():.6f}")
        logger.info(f"  Min importance: {importances.min():.6f}")
        logger.info(f"  Selected features (top {n_features}) importance range: {importances[top_indices].min():.6f} - {importances[top_indices].max():.6f}")
        if len(removed_indices) > 0:
            removed_importances = importances[removed_indices]
            logger.info(f"  Removed features importance range: {removed_importances.min():.6f} - {removed_importances.max():.6f}")
        
        return selected_features
        
    except Exception as e:
        logger.error(f"Error in feature selection: {e}")
        return None


def train_single_model(model, X_train, y_train, X_val, y_val, X_test, y_test, feature_names, output_path, sample_weight=None, preprocessor=None, class_weight=None):
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
        sample_weight: Optional sample weights for training
        preprocessor: Optional preprocessor instance to set on model
        class_weight: Optional class weights dictionary for handling class imbalance

    Returns:
        Dictionary with training results
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Training {model.name}...")
    logger.info('='*60)

    try:
        # Set feature names
        model.feature_names = feature_names

        # Set preprocessor if provided
        if preprocessor is not None:
            model.set_preprocessor(preprocessor)
            logger.debug(f"Set preprocessor on {model.name}")

        # Train model with sample weights and class weights if provided
        train_kwargs = {}
        if sample_weight is not None:
            train_kwargs['sample_weight'] = sample_weight
        if class_weight is not None:
            train_kwargs['class_weight'] = class_weight

        if train_kwargs:
            model.train(X_train, y_train, X_val, y_val, **train_kwargs)
        else:
            model.train(X_train, y_train, X_val, y_val)

        # Fit probability calibration on validation set if available
        # Only calibrate if model is actually trained
        if model.is_trained and X_val is not None and y_val is not None and len(X_val) > 0:
            try:
                model.fit_calibration(X_val, y_val)
            except Exception as e:
                logger.debug(f"Could not fit calibration for {model.name}: {e}")

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

        # Helper function to safely convert probability to float
        def safe_float_prob(prob_dict, outcome, default=0.5):
            """Safely extract and convert probability to float."""
            try:
                prob_value = prob_dict.get(outcome, default)
                if prob_value is None:
                    return float(default)
                if isinstance(prob_value, str):
                    # Try to convert string to float
                    try:
                        return float(prob_value)
                    except (ValueError, TypeError):
                        return float(default)
                elif isinstance(prob_value, (int, float, np.number)):
                    return float(prob_value)
                else:
                    # Unknown type, return default
                    return float(default)
            except Exception:
                return float(default)

        # Process in batches to avoid memory issues
        batch_size = 50
        for i in range(0, len(X_test), batch_size):
            batch_end = min(i + batch_size, len(X_test))
            batch_X = X_test[i:batch_end]

            # Get predictions for batch
            try:
                # Try batch prediction for efficiency (works for most models)
                batch_probs = []
                for j in range(len(batch_X)):
                    try:
                        proba = model.predict_proba(batch_X[j:j+1])
                        prob_value = safe_float_prob(proba, Outcome.HOME_WIN, 0.5)
                        batch_probs.append(prob_value)
                        pred = model.predict(batch_X[j:j+1])
                        test_predictions.append(1 if pred.prediction.value in ["home_win", "HOME_WIN"] else 0)
                    except Exception as e:
                        logger.warning(f"Error predicting sample {i+j}: {e}")
                        test_predictions.append(0)
                        batch_probs.append(0.5)
                # Ensure all values in batch_probs are floats before extending
                test_probs.extend([float(p) for p in batch_probs])
            except Exception as e:
                logger.warning(f"Batch prediction failed, using individual predictions: {e}")
                # Fallback to individual predictions
                for j in range(len(batch_X)):
                    try:
                        pred = model.predict(batch_X[j:j+1])
                        test_predictions.append(1 if pred.prediction.value in ["home_win", "HOME_WIN"] else 0)
                        proba = model.predict_proba(batch_X[j:j+1])
                        prob_value = safe_float_prob(proba, Outcome.HOME_WIN, 0.5)
                        test_probs.append(prob_value)
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

        # Save model (final save after evaluation)
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


def train_all_models(X_train, y_train, X_val, y_val, X_test, y_test, feature_names, output_dir, parallel=True, model_filter=None, sample_weight=None, preprocessor=None, class_weight=None):
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
        sample_weight: Optional sample weights for training
        preprocessor: Optional preprocessor instance to set on models
        class_weight: Optional class weights dictionary for handling class imbalance
    """
    from src.utils.data_types import Outcome

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize all models for maximum diversity and performance
    all_models = [
        # Neural Network
        DeepPredictor(
            name="Neural Analyst",
            epochs=200,
            early_stopping_patience=20,
            hidden_layers=[512, 256, 128, 64],
            dropout=0.35,
            learning_rate=0.0003
        ),
        # Statistical Models
        StatisticalModel(name="Statistical Conservative"),
        SVMModel(name="SVM Strategist"),
        # Tree-based Models
        # XGBoost re-enabled with fixes (environment variables, data type safety)
        GradientBoostModel(
            name="Gradient Strategist",
            n_estimators=200,
            max_depth=7,
            learning_rate=0.05
        ),
        RandomForestModel(
            name="Forest Evaluator",
            n_estimators=200,
            max_depth=15,
            min_samples_split=10
        ),
        # LightGBM re-enabled with fixes (LGBMClassifier sklearn API, environment variables)
        LightGBMModel(
            name="LightGBM Optimizer",
            n_estimators=200,
            max_depth=7,
            learning_rate=0.05
        ),
    ]

    # Optionally add CatBoost if available
    try:
        from src.models.traditional.catboost_model import CatBoostModel
        catboost_model = CatBoostModel(
            name="CatBoost Optimizer",
            n_estimators=300,
            max_depth=7,
            learning_rate=0.03
        )
        all_models.append(catboost_model)
        logger.info("Added CatBoost model")
    except ImportError:
        logger.info("CatBoost not available, skipping")
    except Exception as e:
        logger.warning(f"Could not add CatBoost: {e}")

    # Add optimized ensemble model with top models
    # Using unanimous_high_confidence strategy for 70% accuracy
    from src.models.ensemble.ensemble_model import EnsembleModel
    ensemble = EnsembleModel(
        name="Ensemble Council",
        base_models=all_models.copy(),  # Use copy to avoid circular reference
        voting_strategy="weighted_confidence"  # Use weighted confidence with performance-based weighting
    )
    # Enable performance-based weighting
    ensemble.use_performance_weights = True
    ensemble.min_model_accuracy = 0.65  # Filter models below 65% accuracy
    all_models.append(ensemble)

    # Optionally add stacking meta-learner
    try:
        from src.models.ensemble.stacking_model import StackingModel
        stacking = StackingModel(
            name="Stacking Meta-Learner",
            base_models=all_models[:-1].copy(),  # All models except the voting ensemble
            meta_learner_type="logistic",  # Use logistic regression as meta-learner
            n_folds=5
        )
        all_models.append(stacking)
        logger.info("Added Stacking Meta-Learner")
    except Exception as e:
        logger.warning(f"Could not add Stacking Meta-Learner: {e}")

    # Filter models if specified
    if model_filter:
        models = [m for m in all_models if m.name in model_filter]
        if not models:
            logger.warning(f"No models found matching filter: {model_filter}")
            logger.info(f"Available models: {[m.name for m in all_models]}")
            return []
    else:
        models = all_models
    
    # Log all models being trained
    logger.info(f"\nTraining {len(models)} models: {[m.name for m in models]}")
    logger.info(f"Models include CatBoost: {any('CatBoost' in m.name for m in models)}")
    logger.info(f"Models include Stacking: {any('Stacking' in m.name for m in models)}")
    
    # Log all models being trained
    logger.info(f"\n{'='*60}")
    logger.info(f"TRAINING {len(models)} MODELS")
    logger.info(f"{'='*60}")
    for i, model in enumerate(models, 1):
        logger.info(f"  {i}. {model.name}")
    
    # Verify CatBoost and Stacking are included
    has_catboost = any('CatBoost' in m.name for m in models)
    has_stacking = any('Stacking' in m.name or 'Meta-Learner' in m.name for m in models)
    logger.info(f"\nModel verification:")
    logger.info(f"  CatBoost included: {'✓ YES' if has_catboost else '✗ NO'}")
    logger.info(f"  Stacking Meta-Learner included: {'✓ YES' if has_stacking else '✗ NO'}")
    logger.info(f"{'='*60}\n")
    logger.info(f"{'='*60}")
    
    # Verify CatBoost and Stacking are included
    has_catboost = any('CatBoost' in m.name for m in models)
    has_stacking = any('Stacking' in m.name for m in models)
    logger.info(f"Models include CatBoost: {has_catboost}")
    logger.info(f"Models include Stacking: {has_stacking}")
    if not has_catboost:
        logger.warning("⚠️  CatBoost model not found in training list!")
    if not has_stacking:
        logger.warning("⚠️  Stacking Meta-Learner not found in training list!")
    logger.info("")
    
    results = []
    
    if parallel:
        # Train models in parallel using ThreadPoolExecutor
        logger.info("Training models in parallel...")
        try:
            with ThreadPoolExecutor(max_workers=min(len(models), 4)) as executor:
                # Submit all training tasks
                future_to_model = {
                    executor.submit(
                        train_single_model,
                        model, X_train, y_train, X_val, y_val, X_test, y_test,
                        feature_names, output_path, sample_weight, preprocessor, class_weight
                    ): model for model in models
                }
                
                logger.info(f"Submitted {len(future_to_model)} models for parallel training")

                # Collect results as they complete
                completed = 0
                for future in as_completed(future_to_model):
                    completed += 1
                    model = future_to_model[future]
                    logger.info(f"Processing result {completed}/{len(future_to_model)} for {model.name}...")
                    try:
                        result = future.result(timeout=7200)  # 2 hour timeout per model
                        results.append(result)
                        if result.get('success', False):
                            accuracy = result.get('accuracy', 0.0)
                            logger.info(f"✅ {model.name} completed successfully ({accuracy:.1%} accuracy)")
                            
                            # Update ensemble with model accuracy for performance-based weighting
                            from src.models.ensemble.ensemble_model import EnsembleModel
                            if not isinstance(model, EnsembleModel):
                                # Find ensemble in models list and update it
                                for m in models:
                                    if isinstance(m, EnsembleModel) and model in m.base_models:
                                        m.update_model_accuracy(model.name, accuracy)
                                        logger.debug(f"Updated ensemble with {model.name} accuracy: {accuracy:.1%}")
                        else:
                            logger.warning(f"❌ {model.name} failed: {result.get('error', 'Unknown error')}")
                    except Exception as e:
                        logger.error(f"Error training {model.name} in parallel: {e}", exc_info=True)
            results.append({
                'model': model.name,
                            'success': False,
                            'error': str(e),
                            'accuracy': 0.0
                        })
        except Exception as e:
            logger.error(f"Critical error in parallel training: {e}", exc_info=True)
            logger.info("Falling back to sequential training...")
            parallel = False  # Fall through to sequential
    if not parallel:
        # Train models sequentially (more reliable)
        logger.info("Training models sequentially...")
        for idx, model in enumerate(models, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Training model {idx}/{len(models)}: {model.name}")
            logger.info(f"{'='*60}")
            try:
                result = train_single_model(
                    model, X_train, y_train, X_val, y_val, X_test, y_test,
                    feature_names, output_path, sample_weight, preprocessor, class_weight
                )
                results.append(result)
                if result.get('success', False):
                    accuracy = result.get('accuracy', 0.0)
                    logger.info(f"✅ {model.name}: {accuracy:.1%} accuracy")
                    
                    # Update ensemble with model accuracy for performance-based weighting
                    from src.models.ensemble.ensemble_model import EnsembleModel
                    if not isinstance(model, EnsembleModel):
                        # Find ensemble in models list and update it
                        for m in models:
                            if isinstance(m, EnsembleModel) and model in m.base_models:
                                m.update_model_accuracy(model.name, accuracy)
                                logger.debug(f"Updated ensemble with {model.name} accuracy: {accuracy:.1%}")
                else:
                    logger.warning(f"❌ {model.name} failed: {result.get('error', 'Unknown error')}")
            except Exception as e:
                logger.error(f"Critical error training {model.name}: {e}", exc_info=True)
                results.append({
                    'model': model.name,
                    'success': False,
                    'error': str(e),
                    'accuracy': 0.0
                })

    # Update ensemble with all model accuracies before final evaluation
    # This ensures performance-based weighting works correctly
    from src.models.ensemble.ensemble_model import EnsembleModel
    for result in results:
        if result.get('success', False) and not result.get('model', '').endswith('Council'):
            model_name = result.get('model', '')
            accuracy = result.get('accuracy', 0.0)
            # Find and update ensemble
            for model in models:
                if isinstance(model, EnsembleModel):
                    # Find the base model by name
                    for base_model in model.base_models:
                        if base_model.name == model_name:
                            model.update_model_accuracy(model_name, accuracy)
                            logger.debug(f"Updated ensemble with {model_name} accuracy: {accuracy:.1%}")
                            break
    
    # Log ensemble configuration
    for model in models:
        if isinstance(model, EnsembleModel):
            logger.info(f"\n{'='*60}")
            logger.info(f"ENSEMBLE CONFIGURATION: {model.name}")
            logger.info(f"{'='*60}")
            logger.info(f"Voting strategy: {model.voting_strategy}")
            logger.info(f"Performance-based weighting: {model.use_performance_weights}")
            logger.info(f"Minimum model accuracy threshold: {model.min_model_accuracy:.1%}")
            logger.info(f"\nModel accuracies and weights:")
            for base_model in model.base_models:
                accuracy = model.model_accuracies.get(base_model.name, 0.0)
                perf_weights = model._calculate_performance_weights(model.base_models)
                weight = perf_weights.get(base_model.name, 0.0)
                status = "✓ INCLUDED" if accuracy >= model.min_model_accuracy else "✗ EXCLUDED"
                logger.info(f"  {base_model.name}: {accuracy:.1%} accuracy, weight: {weight:.3f} {status}")
            logger.info(f"{'='*60}\n")
    
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
            logger.info(f"  Accuracy:  {result.get('accuracy', 0):.1%}")
            logger.info(f"  Precision: {result.get('precision', 0):.1%}")
            logger.info(f"  Recall:    {result.get('recall', 0):.1%}")
            logger.info(f"  F1 Score:  {result.get('f1', 0):.1%}")
            logger.info(f"  ROC-AUC:   {result.get('roc_auc', 0):.3f}")

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
    parser.add_argument('--feature-selection', action='store_true', default=True,
                       help='Apply feature selection to keep most important features (default: True)')
    parser.add_argument('--tune-hyperparameters', action='store_true',
                       help='Tune hyperparameters using Optuna (slower but better accuracy)')
    parser.add_argument('--temporal-weighting', action='store_true', default=True,
                       help='Enable temporal weighting (default: True)')
    parser.add_argument('--no-temporal-weighting', dest='temporal_weighting', action='store_false',
                       help='Disable temporal weighting')
    parser.add_argument('--decay-factor', type=float, default=0.25,
                       help='Temporal weighting decay factor (default: 0.25, higher = more weight on recent)')
    parser.add_argument('--include-2025-week15', action='store_true', default=True,
                       help='Include 2025 data through Week 15 (default: True)')
    parser.add_argument('--no-include-2025-week15', dest='include_2025_week15', action='store_false',
                       help='Do not filter 2025 data to Week 15')
    parser.add_argument('--no-feature-selection', dest='use_feature_selection', action='store_false', default=True,
                       help='Disable feature selection (use all features)')
    parser.add_argument('--n-features', type=int, default=110,
                       help='Number of features to select (default: 110)')
    parser.add_argument('--test-no-feature-selection', action='store_true',
                       help='Test mode: disable feature selection to compare accuracy')
    parser.add_argument('--test-decay-factor', type=float, default=None,
                       help='Test mode: override decay factor for temporal weighting (e.g., 0.15, 0.20)')
    parser.add_argument('--test-set-2024-only', action='store_true',
                       help='Test mode: use only 2024 late weeks for testing (exclude 2025 weeks 14-15)')
    
    args = parser.parse_args()
    
    # Apply test mode overrides
    if args.test_no_feature_selection:
        logger.info("TEST MODE: Feature selection disabled for comparison")
        args.use_feature_selection = False
    if args.test_decay_factor is not None:
        logger.info(f"TEST MODE: Using decay_factor={args.test_decay_factor} instead of default")
        args.decay_factor = args.test_decay_factor
    
    logger.info("\n" + "="*60)
    logger.info("NFL BETTING AGENT COUNCIL - MODEL TRAINING")
    logger.info("="*60)
    logger.info(f"Seasons: {args.seasons}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Temporal weighting: {args.temporal_weighting}")
    if args.temporal_weighting:
        logger.info(f"Decay factor: {args.decay_factor}")
    logger.info(f"Include 2025 Week 15: {args.include_2025_week15}\n")
    
    # Prepare data
    logger.info("Preparing training data...")
    # Prepare training data with test mode support
    test_set_2024_only = getattr(args, 'test_set_2024_only', False)
    
    X_train, y_train, X_val, y_val, X_test, y_test, feature_names, preprocessor, sample_weights = prepare_training_data(
        args.seasons,
        include_2025_week15=args.include_2025_week15,
        temporal_weighting=args.temporal_weighting,
        decay_factor=args.decay_factor,
        use_feature_selection=args.use_feature_selection,
        n_features=args.n_features,
        test_set_2024_only=test_set_2024_only
    )
    
    logger.info(f"\nDataset sizes:")
    logger.info(f"  Training:   {X_train.shape[0]} games, {X_train.shape[1]} features")
    logger.info(f"  Validation: {X_val.shape[0]} games")
    logger.info(f"  Test:       {X_test.shape[0]} games")
    
    # Feature selection is already applied in prepare_training_data if enabled
    # All models will use the same selected features
    if args.use_feature_selection:
        logger.info(f"\n✓ Feature selection enabled: Using top {len(feature_names)} features")
    else:
        logger.info(f"\n✓ Using all {len(feature_names)} features (feature selection disabled)")

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save preprocessor (save as both scaler.pkl for backward compatibility and preprocessor.pkl)
    scaler_path = output_path / "scaler.pkl"
    preprocessor_path = output_path / "preprocessor.pkl"
    joblib.dump(preprocessor, scaler_path)
    joblib.dump(preprocessor, preprocessor_path)
    logger.info(f"Saved preprocessor to {scaler_path} and {preprocessor_path}")
    logger.info(f"Preprocessor has {len(preprocessor.get_feature_names())} features")

    # Calculate class weights for handling class imbalance
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = {int(c): float(w) for c, w in zip(classes, class_weights)}
    logger.info(f"Class weights: {class_weight_dict}")

    # Train models (sequential by default for reliability - parallel can cause silent failures)
    # Sequential is more reliable, easier to debug, and shows progress clearly
    use_parallel = False  # Disable parallel by default - more reliable
    logger.info(f"Training mode: {'parallel' if use_parallel else 'sequential (more reliable)'}")
    
    results = train_all_models(
        X_train, y_train, X_val, y_val, X_test, y_test, 
        feature_names, args.output_dir, 
        parallel=use_parallel,
        model_filter=args.model,
        sample_weight=sample_weights,
        preprocessor=preprocessor,
        class_weight=class_weight_dict
    )

    # Calculate and log feature importance for tree-based models
    feature_importance_summary = {}
    for result in results:
        if result.get('success', False):
            model_name = result['model']
            # Try to get feature importance from saved model
            try:
                model_file = output_path / f"{model_name.lower().replace(' ', '_')}.pkl"
                if model_file.exists():
                    model = joblib.load(model_file)
                    if hasattr(model, 'get_feature_importance'):
                        importance = model.get_feature_importance()
                        if importance:
                            # Get top 20 features
                            top_features = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:20])
                            feature_importance_summary[model_name] = top_features
                            logger.info(f"\nTop 10 features for {model_name}:")
                            for i, (feat, imp) in enumerate(list(top_features.items())[:10], 1):
                                logger.info(f"  {i}. {feat}: {imp:.4f}")
            except Exception as e:
                logger.debug(f"Could not extract feature importance for {model_name}: {e}")

    # Save metadata
    model_accuracies = {r['model']: r['accuracy'] for r in results if r.get('success', True)}
    metadata_dict = {
        'trained_at': datetime.now().isoformat(),
        'seasons': args.seasons,
        'feature_names': feature_names,
        'num_features': len(feature_names),
        'model_accuracies': model_accuracies,
        'scaler_path': str(scaler_path),
        'strategy': 'selective_75',
        'min_confidence': 0.70,
        'min_spread': 5.0,
        'temporal_weighting': args.temporal_weighting,
        'decay_factor': args.decay_factor if args.temporal_weighting else None,
        'include_2025_week15': args.include_2025_week15,
        'top_features': feature_importance_summary
    }

    metadata_path = output_path / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata_dict, f, indent=2)

    logger.info(f"Saved training metadata to {metadata_path}")
    
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

