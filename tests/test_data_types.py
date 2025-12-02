"""Tests for data types."""

import pytest
from datetime import datetime
from src.utils.data_types import (
    BetType,
    Outcome,
    GameInfo,
    BettingLine,
    Proposition,
    ModelPrediction
)


def test_game_info_creation():
    """Test GameInfo creation."""
    game = GameInfo(
        game_id="2024_10_KC_BUF",
        home_team="KC",
        away_team="BUF",
        game_date=datetime(2024, 11, 17),
        season=2024,
        week=10
    )
    
    assert game.game_id == "2024_10_KC_BUF"
    assert game.home_team == "KC"
    assert game.away_team == "BUF"
    assert game.season == 2024
    assert game.week == 10


def test_betting_line_creation():
    """Test BettingLine creation."""
    line = BettingLine(
        spread=-2.5,
        total=54.5,
        home_ml=-135,
        away_ml=115
    )
    
    assert line.spread == -2.5
    assert line.total == 54.5
    assert line.home_ml == -135
    assert line.away_ml == 115


def test_proposition_creation():
    """Test Proposition creation."""
    game_info = GameInfo(
        game_id="2024_10_KC_BUF",
        home_team="KC",
        away_team="BUF",
        game_date=datetime(2024, 11, 17),
        season=2024,
        week=10
    )
    
    line = BettingLine(
        spread=-2.5,
        total=54.5,
        home_ml=-135,
        away_ml=115
    )
    
    prop = Proposition(
        prop_id="test_prop",
        game_info=game_info,
        bet_type=BetType.SPREAD,
        line=line
    )
    
    assert prop.prop_id == "test_prop"
    assert prop.bet_type == BetType.SPREAD
    assert prop.line.spread == -2.5


def test_model_prediction_creation():
    """Test ModelPrediction creation."""
    pred = ModelPrediction(
        model_name="Test Model",
        prediction=Outcome.HOME_WIN,
        confidence=0.75,
        probability=0.75,
        key_features={"feature1": 1.0},
        reasoning="Test reasoning"
    )
    
    assert pred.model_name == "Test Model"
    assert pred.prediction == Outcome.HOME_WIN
    assert pred.confidence == 0.75
    assert "feature1" in pred.key_features


if __name__ == "__main__":
    pytest.main([__file__])

