"""Tests for feature engineering."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from src.data.processors.feature_engineer import FeatureEngineer
from src.utils.data_types import Proposition, GameInfo, BettingLine, BetType


class TestFeatureEngineer:
    """Test suite for FeatureEngineer."""
    
    @pytest.fixture
    def feature_engineer(self):
        """Create a FeatureEngineer instance."""
        return FeatureEngineer()
    
    @pytest.fixture
    def sample_schedule_df(self):
        """Create a sample schedule DataFrame."""
        return pd.DataFrame({
            'game_id': ['2023_01_BUF_KC', '2023_01_SF_DAL'],
            'season': [2023, 2023],
            'week': [1, 1],
            'gameday': ['2023-09-07', '2023-09-10'],
            'home_team': ['KC', 'DAL'],
            'away_team': ['BUF', 'SF'],
            'home_score': [21, 17],
            'away_score': [20, 28]
        })
    
    @pytest.fixture
    def sample_team_stats_df(self):
        """Create a sample team stats DataFrame."""
        return pd.DataFrame({
            'team': ['KC', 'BUF'],
            'season': [2023, 2023]
        })
    
    @pytest.fixture
    def sample_proposition(self):
        """Create a sample proposition."""
        game_info = GameInfo(
            game_id='2023_02_BUF_KC',
            home_team='KC',
            away_team='BUF',
            game_date=datetime(2023, 9, 14),
            season=2023,
            week=2
        )
        
        betting_line = BettingLine(
            spread=-3.5,
            total=52.5,
            home_ml=-165,
            away_ml=145
        )
        
        return Proposition(
            prop_id='test_prop',
            game_info=game_info,
            bet_type=BetType.SPREAD,
            line=betting_line
        )
    
    def test_extract_basic_features(self, feature_engineer, sample_proposition):
        """Test basic feature extraction."""
        features = feature_engineer._extract_basic_features(sample_proposition.game_info)
        
        assert 'season' in features
        assert 'week' in features
        assert features['season'] == 2023
        assert features['week'] == 2
        assert 'day_of_week' in features
        assert 'is_playoffs' in features
    
    def test_extract_betting_line_features(self, feature_engineer, sample_proposition,
                                          sample_schedule_df, sample_team_stats_df):
        """Test extraction of betting line features."""
        features = feature_engineer.extract_features(
            sample_proposition,
            sample_schedule_df,
            sample_team_stats_df
        )
        
        assert 'spread' in features
        assert 'total' in features
        assert features['spread'] == -3.5
        assert features['total'] == 52.5
        assert 'implied_home_prob' in features
        assert 'implied_away_prob' in features
    
    def test_ml_to_prob_conversion(self, feature_engineer):
        """Test moneyline to probability conversion."""
        # Favorite (-150)
        prob = feature_engineer._ml_to_prob(-150)
        assert 0.5 < prob < 1.0
        assert abs(prob - 0.6) < 0.1
        
        # Underdog (+130)
        prob = feature_engineer._ml_to_prob(130)
        assert 0.0 < prob < 0.5
        assert abs(prob - 0.435) < 0.1
    
    def test_is_divisional_game(self, feature_engineer):
        """Test divisional game detection."""
        # AFC West matchup
        assert feature_engineer._is_divisional_game('KC', 'DEN') == True
        
        # Non-divisional matchup
        assert feature_engineer._is_divisional_game('KC', 'BUF') == False
        
        # NFC North matchup
        assert feature_engineer._is_divisional_game('GB', 'CHI') == True
    
    def test_feature_extraction_complete(self, feature_engineer, sample_proposition,
                                        sample_schedule_df, sample_team_stats_df):
        """Test that feature extraction returns a complete feature set."""
        features = feature_engineer.extract_features(
            sample_proposition,
            sample_schedule_df,
            sample_team_stats_df
        )
        
        # Check that we have features
        assert len(features) > 0
        assert isinstance(features, dict)
        
        # Check for key feature categories
        assert 'season' in features
        assert 'week' in features
        
        # All features should be numeric
        for key, value in features.items():
            assert isinstance(value, (int, float, bool)), f"Feature {key} is not numeric: {type(value)}"
    
    def test_recent_form_features(self, feature_engineer):
        """Test recent form calculation."""
        # Create mock schedule with recent games
        recent_games = pd.DataFrame({
            'game_id': ['g1', 'g2', 'g3'],
            'gameday': ['2023-09-01', '2023-09-08', '2023-09-15'],
            'home_team': ['KC', 'BUF', 'KC'],
            'away_team': ['BUF', 'MIA', 'LAC'],
            'home_score': [24, 31, 28],
            'away_score': [20, 17, 21]
        })
        
        # Test win percentage calculation
        win_pct = feature_engineer._calc_win_pct(recent_games, 'KC')
        assert 0.0 <= win_pct <= 1.0

