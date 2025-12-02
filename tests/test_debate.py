"""Tests for debate system (mocked to avoid API calls)."""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime

from src.agents.moderator import DebateModerator
from src.utils.data_types import (
    ModelPrediction,
    Outcome,
    Proposition,
    GameInfo,
    BetType
)


class TestDebateSystem:
    """Test suite for the debate system."""
    
    @pytest.fixture
    def sample_predictions(self):
        """Create sample model predictions."""
        return [
            ModelPrediction(
                model_name="Model A",
                prediction=Outcome.HOME_WIN,
                confidence=0.75,
                key_features={'feature1': 1.0},
                reasoning="Strong home advantage"
            ),
            ModelPrediction(
                model_name="Model B",
                prediction=Outcome.HOME_WIN,
                confidence=0.65,
                key_features={'feature2': 0.5},
                reasoning="Historical trends favor home"
            ),
            ModelPrediction(
                model_name="Model C",
                prediction=Outcome.AWAY_WIN,
                confidence=0.60,
                key_features={'feature3': -0.5},
                reasoning="Away team recent form"
            )
        ]
    
    @pytest.fixture
    def sample_proposition(self):
        """Create a sample proposition."""
        game_info = GameInfo(
            game_id='2023_10_BUF_KC',
            home_team='KC',
            away_team='BUF',
            game_date=datetime(2023, 11, 1),
            season=2023,
            week=10
        )
        
        return Proposition(
            prop_id='test_prop',
            game_info=game_info,
            bet_type=BetType.GAME_OUTCOME
        )
    
    def test_weighted_voting(self, sample_predictions, sample_proposition):
        """Test weighted voting fallback mechanism."""
        moderator = DebateModerator()
        
        model_accuracies = {
            'Model A': 0.70,
            'Model B': 0.65,
            'Model C': 0.60
        }
        
        # Test weighted voting
        prediction, confidence, consensus = moderator._weighted_voting(
            sample_predictions,
            [],  # Empty debate transcript
            model_accuracies
        )
        
        # Should predict HOME_WIN (2 models with higher accuracies)
        assert prediction == Outcome.HOME_WIN
        assert 0.0 <= confidence <= 1.0
        assert 0.0 <= consensus <= 1.0
    
    def test_parse_synthesis(self):
        """Test parsing of moderator synthesis."""
        moderator = DebateModerator()
        
        synthesis = """
        FINAL RECOMMENDATION: HOME_WIN
        CONFIDENCE: 72%
        REASONING: The home team has strong advantages in this matchup.
        CONSENSUS LEVEL: 85%
        """
        
        prediction, confidence, reasoning, consensus = moderator._parse_synthesis(synthesis)
        
        assert prediction == Outcome.HOME_WIN
        assert confidence == 0.72
        assert consensus == 0.85
        assert "advantages" in reasoning.lower()
    
    def test_format_predictions(self, sample_predictions):
        """Test formatting of predictions."""
        moderator = DebateModerator()
        
        model_accuracies = {
            'Model A': 0.70,
            'Model B': 0.65,
            'Model C': 0.60
        }
        
        formatted = moderator._format_predictions(sample_predictions, model_accuracies)
        
        assert 'Model A' in formatted
        assert 'Model B' in formatted
        assert 'Model C' in formatted
        assert '70.0%' in formatted or '70%' in formatted
    
    def test_format_proposition(self, sample_proposition):
        """Test proposition formatting."""
        moderator = DebateModerator()
        
        formatted = moderator._format_proposition(sample_proposition)
        
        assert 'BUF' in formatted
        assert 'KC' in formatted
        assert '2023' in formatted
        assert 'Week 10' in formatted or 'week=10' in formatted
    
    @patch('src.agents.moderator.DebateAgent')
    def test_create_agents(self, mock_agent_class, sample_predictions):
        """Test agent creation."""
        moderator = DebateModerator()
        
        model_accuracies = {
            'Neural Analyst': 0.68,
            'Gradient Strategist': 0.65,
            'Forest Evaluator': 0.62
        }
        
        # Create predictions with matching model names
        predictions = [
            ModelPrediction(
                model_name='Neural Analyst',
                prediction=Outcome.HOME_WIN,
                confidence=0.70,
                reasoning="Test"
            ),
            ModelPrediction(
                model_name='Gradient Strategist',
                prediction=Outcome.HOME_WIN,
                confidence=0.65,
                reasoning="Test"
            )
        ]
        
        agents = moderator._create_agents(predictions, model_accuracies)
        
        assert len(agents) == len(predictions)
    
    def test_consensus_calculation(self):
        """Test consensus level calculation."""
        # High consensus case
        predictions_high = [
            ModelPrediction('M1', Outcome.HOME_WIN, 0.7, reasoning=""),
            ModelPrediction('M2', Outcome.HOME_WIN, 0.75, reasoning=""),
            ModelPrediction('M3', Outcome.HOME_WIN, 0.72, reasoning="")
        ]
        
        moderator = DebateModerator()
        accuracies = {'M1': 0.6, 'M2': 0.65, 'M3': 0.62}
        
        _, _, consensus = moderator._weighted_voting(predictions_high, [], accuracies)
        
        # With all predictions the same, consensus should be high
        assert consensus > 0.8

