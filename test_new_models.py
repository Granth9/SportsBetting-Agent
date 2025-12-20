"""Test script to verify new models (CatBoost and Stacking) work correctly."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline.predictor import BettingCouncil
from src.utils.data_types import Proposition, GameInfo, BetType
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def test_model_integration():
    """Test that new models integrate correctly with existing models."""
    logger.info("="*60)
    logger.info("Testing Model Integration")
    logger.info("="*60)
    
    try:
        # Initialize council (will load all trained models)
        logger.info("Initializing BettingCouncil...")
        council = BettingCouncil()
        
        logger.info(f"Loaded {len(council.models)} models")
        for model in council.models:
            logger.info(f"  - {model.name} ({model.model_type})")
            if hasattr(model, 'base_models'):
                logger.info(f"    Base models: {len(model.base_models)}")
        
        # Check if new models are present
        model_names = [m.name for m in council.models]
        has_catboost = any('CatBoost' in name for name in model_names)
        has_stacking = any('Stacking' in name for name in model_names)
        
        logger.info(f"\nNew Models Status:")
        logger.info(f"  CatBoost: {'✅ Found' if has_catboost else '❌ Not found (optional)'}")
        logger.info(f"  Stacking: {'✅ Found' if has_stacking else '❌ Not found'}")
        
        # Test prediction
        logger.info("\n" + "="*60)
        logger.info("Testing Prediction")
        logger.info("="*60)
        
        prop = Proposition(
            prop_id="test_1",
            game_info=GameInfo(
                home_team="KC",
                away_team="BUF",
                season=2024,
                week=10,
                home_score=None,
                away_score=None
            ),
            bet_type=BetType.SPREAD,
            line=-2.5
        )
        
        logger.info(f"Testing prediction for: {prop.game_info.away_team} @ {prop.game_info.home_team}")
        logger.info(f"Spread: {prop.line}")
        
        # Get predictions from all models
        logger.info("\nGetting predictions from all models...")
        schedule_df, team_stats_df, player_stats_df = council._collect_data(prop)
        features = council.feature_engineer.extract_features(
            prop, schedule_df, team_stats_df, player_stats_df
        )
        
        predictions = council._run_models(prop, features)
        
        logger.info(f"\nReceived {len(predictions)} predictions:")
        for pred in predictions:
            logger.info(f"  {pred.model_name}: {pred.prediction.value} ({pred.confidence:.1%})")
        
        # Test full analysis (includes debate)
        logger.info("\n" + "="*60)
        logger.info("Testing Full Analysis (with Debate)")
        logger.info("="*60)
        
        recommendation = council.analyze(prop)
        
        logger.info(f"\nFinal Recommendation:")
        logger.info(f"  Prediction: {recommendation.prediction.value}")
        logger.info(f"  Confidence: {recommendation.confidence:.1%}")
        logger.info(f"  Action: {recommendation.action.value}")
        logger.info(f"  Bet Size: {recommendation.bet_size:.1%}")
        
        # Check if stacking was used
        if has_stacking:
            stacking_pred = [p for p in predictions if 'Stacking' in p.model_name]
            if stacking_pred:
                logger.info(f"\n✅ Stacking Meta-Learner prediction: {stacking_pred[0].prediction.value} ({stacking_pred[0].confidence:.1%})")
        
        logger.info("\n" + "="*60)
        logger.info("✅ All tests passed!")
        logger.info("="*60)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = test_model_integration()
    sys.exit(0 if success else 1)

