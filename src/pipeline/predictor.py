"""Main prediction pipeline coordinating models and debate."""

from typing import List, Dict, Any, Optional
from pathlib import Path
import asyncio

from src.models.base_model import BaseModel
from src.models.neural_nets.deep_predictor import DeepPredictor
from src.models.traditional.gradient_boost_model import GradientBoostModel
from src.models.traditional.random_forest_model import RandomForestModel
from src.models.traditional.statistical_model import StatisticalModel
from src.models.traditional.lightgbm_model import LightGBMModel
from src.models.traditional.svm_model import SVMModel

# CatBoost is optional
try:
    from src.models.traditional.catboost_model import CatBoostModel
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
from src.models.ensemble.ensemble_model import EnsembleModel
from src.models.ensemble.stacking_model import StackingModel
from src.agents.moderator import DebateModerator
from src.data.collectors.nfl_data_collector import NFLDataCollector
from src.data.processors.feature_engineer import FeatureEngineer
from src.data.processors.data_preprocessor import DataPreprocessor
from src.utils.data_types import (
    Proposition,
    ModelPrediction,
    DebateResult,
    Recommendation,
    BetType,
    Outcome
)
from src.utils.logger import setup_logger
from src.utils.config_loader import get_config


logger = setup_logger(__name__)


class BettingCouncil:
    """Main betting council that coordinates models and debate."""
    
    def __init__(
        self,
        models: Optional[List[BaseModel]] = None,
        debate_rounds: int = 4
    ):
        """Initialize the betting council.
        
        Args:
            models: List of trained models (will create default if None)
            debate_rounds: Number of debate rounds
        """
        self.config = get_config()
        
        # Initialize models
        if models is None:
            self.models = self._create_default_models()
        else:
            self.models = models
        
        # Initialize moderator
        self.moderator = DebateModerator(num_rounds=debate_rounds)
        
        # Initialize data components
        self.data_collector = NFLDataCollector()
        self.feature_engineer = FeatureEngineer()
        self.preprocessor: Optional[DataPreprocessor] = None  # Preprocessor for feature transformation
        
        # Track model performance
        self.model_accuracies: Dict[str, float] = {
            model.name: model.get_recent_accuracy() for model in self.models
        }
        
        # Enable dynamic weighting for ensemble models
        for model in self.models:
            if isinstance(model, EnsembleModel):
                model.enable_dynamic_weights()
                logger.info(f"Enabled dynamic weighting for {model.name}")
        
        logger.info(f"Betting Council initialized with {len(self.models)} models")
    
    def _create_default_models(self) -> List[BaseModel]:
        """Create default set of models.
        
        Returns:
            List of initialized models
        """
        logger.info("Creating default model set")
        
        models = [
            DeepPredictor(
                name="Neural Analyst",
                hidden_layers=self.config.get('models.neural_net.hidden_layers', [256, 128, 64]),
                dropout=self.config.get('models.neural_net.dropout', 0.3),
                learning_rate=self.config.get('models.neural_net.learning_rate', 0.001),
                epochs=self.config.get('models.neural_net.epochs', 100),
                early_stopping_patience=10
            ),
            GradientBoostModel(
                name="Gradient Strategist",
                n_estimators=self.config.get('models.gradient_boosting.n_estimators', 200),
                max_depth=self.config.get('models.gradient_boosting.max_depth', 6),
                learning_rate=self.config.get('models.gradient_boosting.learning_rate', 0.05)
            ),
            RandomForestModel(
                name="Forest Evaluator",
                n_estimators=self.config.get('models.random_forest.n_estimators', 150),
                max_depth=self.config.get('models.random_forest.max_depth', 10),
                min_samples_split=self.config.get('models.random_forest.min_samples_split', 5)
            ),
            StatisticalModel(
                name="Statistical Conservative",
                regularization=self.config.get('models.statistical.regularization', 0.1)
            ),
            LightGBMModel(
                name="LightGBM Optimizer",
                n_estimators=self.config.get('models.lightgbm.n_estimators', 200),
                max_depth=self.config.get('models.lightgbm.max_depth', 7),
                learning_rate=self.config.get('models.lightgbm.learning_rate', 0.05),
                num_leaves=self.config.get('models.lightgbm.num_leaves', 31)
            ),
            SVMModel(
                name="SVM Strategist",
                C=self.config.get('models.svm.C', 1.0),
                kernel=self.config.get('models.svm.kernel', 'rbf'),
                gamma=self.config.get('models.svm.gamma', 'scale')
            )
        ]
        
        # Optionally add CatBoost if available
        if CATBOOST_AVAILABLE:
            try:
                catboost_config = self.config.get('models.catboost', {})
                models.append(CatBoostModel(
                    name="CatBoost Optimizer",
                    n_estimators=catboost_config.get('n_estimators', 200),
                    max_depth=catboost_config.get('max_depth', 7),
                    learning_rate=catboost_config.get('learning_rate', 0.05)
                ))
                logger.info("Added CatBoost model to ensemble")
            except Exception as e:
                logger.warning(f"Could not add CatBoost model: {e}")
        
        # Optionally add ensemble models
        ensemble_config = self.config.get('models.ensemble', {})
        
        # Add voting ensemble if enabled
        if ensemble_config.get('enabled', False):
            ensemble = EnsembleModel(
                name="Ensemble Council",
                base_models=models.copy(),
                voting_strategy=ensemble_config.get('voting_strategy', 'weighted'),
                weights=ensemble_config.get('weights', {})
            )
            models.append(ensemble)
        
        # Add stacking meta-learner if enabled
        stacking_config = self.config.get('models.stacking', {})
        if stacking_config.get('enabled', False):
            stacking = StackingModel(
                name="Stacking Meta-Learner",
                base_models=models.copy(),
                meta_learner_type=stacking_config.get('meta_learner_type', 'logistic'),
                n_folds=stacking_config.get('n_folds', 5)
            )
            models.append(stacking)
            logger.info("Added Stacking Meta-Learner to ensemble")
        
        return models
    
    def analyze(self, proposition: Proposition) -> Recommendation:
        """Analyze a betting proposition through the full pipeline.
        
        Args:
            proposition: The betting proposition to analyze
            
        Returns:
            Final recommendation with debate results
        """
        logger.info(f"Analyzing proposition: {proposition.prop_id}")
        
        # Step 1: Collect required data
        logger.info("Step 1: Collecting data")
        schedule_df, team_stats_df, player_stats_df = self._collect_data(proposition)
        
        # Step 2: Extract features
        logger.info("Step 2: Extracting features")
        features = self.feature_engineer.extract_features(
            proposition,
            schedule_df,
            team_stats_df,
            player_stats_df
        )
        
        # Step 3: Get predictions from all models
        logger.info("Step 3: Running models")
        model_predictions = self._run_models(proposition, features)
        
        # Step 4: Conduct debate
        logger.info("Step 4: Conducting debate")
        debate_result = self.moderator.orchestrate_debate(
            proposition,
            model_predictions,
            self.model_accuracies
        )
        
        # Step 5: Generate final recommendation
        logger.info("Step 5: Generating recommendation")
        recommendation = self._generate_recommendation(proposition, debate_result)
        
        logger.info(f"Analysis complete: {recommendation.recommended_action}")
        
        return recommendation
    
    def _collect_data(self, proposition: Proposition) -> tuple:
        """Collect required data for the proposition.
        
        Args:
            proposition: The betting proposition
            
        Returns:
            Tuple of (schedule_df, team_stats_df, player_stats_df)
        """
        game_info = proposition.game_info
        
        # Get schedule data
        schedule_df = self.data_collector.get_schedule([game_info.season])
        
        # Get team stats
        team_stats_df = self.data_collector.get_team_stats([game_info.season])
        
        # Get player stats if needed
        player_stats_df = None
        if proposition.bet_type == BetType.PLAYER_PROP:
            player_stats_df = self.data_collector.get_player_stats([game_info.season])
        
        return schedule_df, team_stats_df, player_stats_df
    
    def _run_models(
        self,
        proposition: Proposition,
        features: Dict[str, Any]
    ) -> List[ModelPrediction]:
        """Run all models on the proposition.
        
        Args:
            proposition: The betting proposition
            features: Extracted features
            
        Returns:
            List of model predictions
        """
        # Validate features against preprocessor if available
        if self.preprocessor is not None:
            self._validate_features(features)
        
        predictions = []
        
        for model in self.models:
            if not model.is_trained:
                logger.warning(f"{model.name} is not trained, skipping")
                continue
            
            try:
                prediction = model.predict_proposition(proposition, features)
                predictions.append(prediction)
                logger.debug(f"{model.name}: {prediction.prediction.value} ({prediction.confidence:.1%})")
            
            except Exception as e:
                logger.error(f"Error running {model.name}: {e}")
        
        return predictions
    
    def _validate_features(self, features: Dict[str, Any]) -> None:
        """Validate that features match preprocessor expectations.
        
        Args:
            features: Feature dictionary to validate
        """
        if self.preprocessor is None:
            return
        
        preprocessor_features = set(self.preprocessor.get_feature_names())
        provided_features = set(features.keys())
        
        # Check for missing features
        missing_features = preprocessor_features - provided_features
        if missing_features:
            logger.warning(f"Missing {len(missing_features)} features expected by preprocessor: {list(missing_features)[:10]}")
        
        # Check for unexpected features (less critical, but good to know)
        unexpected_features = provided_features - preprocessor_features
        if unexpected_features:
            logger.debug(f"Found {len(unexpected_features)} features not in preprocessor (will be ignored): {list(unexpected_features)[:10]}")
    
    def _generate_recommendation(
        self,
        proposition: Proposition,
        debate_result: DebateResult
    ) -> Recommendation:
        """Generate final recommendation from debate result.
        
        Args:
            proposition: The betting proposition
            debate_result: Result of the debate
            
        Returns:
            Final recommendation
        """
        # Determine recommended action based on confidence
        confidence_threshold = self.config.get('betting.confidence_threshold', 0.65)
        
        if debate_result.final_confidence >= confidence_threshold + 0.15:
            recommended_action = "STRONG_BET"
        elif debate_result.final_confidence >= confidence_threshold:
            recommended_action = "BET"
        else:
            recommended_action = "PASS"
        
        # Calculate bet size using Kelly Criterion (simplified)
        bet_size = None
        expected_value = None
        
        if recommended_action != "PASS" and proposition.line:
            kelly_fraction = self.config.get('betting.kelly_criterion_fraction', 0.25)
            
            # Convert odds to probability
            if debate_result.final_prediction.value in ["HOME_WIN", "home_win"]:
                odds = proposition.line.home_ml
            else:
                odds = proposition.line.away_ml
            
            # Calculate Kelly bet size (simplified)
            if odds > 0:
                decimal_odds = (odds / 100) + 1
            else:
                decimal_odds = (100 / abs(odds)) + 1
            
            win_prob = debate_result.final_confidence
            expected_value = (win_prob * (decimal_odds - 1)) - (1 - win_prob)
            
            if expected_value > 0:
                kelly = (win_prob * decimal_odds - 1) / (decimal_odds - 1)
                bet_size = max(0, min(kelly * kelly_fraction, 0.05))  # Cap at 5% of bankroll
        
        # Risk assessment
        risk_assessment = self._assess_risk(debate_result, proposition)
        
        return Recommendation(
            proposition=proposition,
            debate_result=debate_result,
            recommended_action=recommended_action,
            bet_size=bet_size,
            expected_value=expected_value,
            risk_assessment=risk_assessment
        )
    
    def _assess_risk(
        self,
        debate_result: DebateResult,
        proposition: Proposition
    ) -> str:
        """Assess risk level for the recommendation.
        
        Args:
            debate_result: Debate result
            proposition: The proposition
            
        Returns:
            Risk assessment string
        """
        confidence = debate_result.final_confidence
        consensus = debate_result.consensus_level
        
        if confidence > 0.75 and consensus > 0.7:
            risk = "LOW"
            explanation = "High confidence with strong model consensus."
        elif confidence > 0.65 and consensus > 0.6:
            risk = "MODERATE"
            explanation = "Good confidence with reasonable consensus."
        elif confidence > 0.55:
            risk = "ELEVATED"
            explanation = "Moderate confidence or divided models."
        else:
            risk = "HIGH"
            explanation = "Low confidence or significant model disagreement."
        
        return f"{risk}: {explanation}"
    
    def load_models(self, model_dir: str) -> None:
        """Load trained models from disk.
        
        Args:
            model_dir: Directory containing saved models
        """
        import joblib
        model_dir_path = Path(model_dir)
        
        # Try to load preprocessor first
        preprocessor_path = model_dir_path / 'preprocessor.pkl'
        if preprocessor_path.exists():
            try:
                self.preprocessor = joblib.load(str(preprocessor_path))
                logger.info(f"Loaded preprocessor from {preprocessor_path}")
                logger.info(f"Preprocessor has {len(self.preprocessor.get_feature_names())} features")
            except Exception as e:
                logger.warning(f"Could not load preprocessor: {e}")
        else:
            logger.warning(f"Preprocessor not found at {preprocessor_path}. Models may not have correct feature scaling.")
        
        # Load models
        for model in self.models:
            model_file = model_dir_path / f"{model.name.lower().replace(' ', '_')}.pkl"
            if model_file.exists():
                try:
                    model.load(str(model_file))
                    logger.info(f"Loaded {model.name} from {model_file}")
                    
                    # Set preprocessor on model if available
                    if self.preprocessor is not None:
                        model.set_preprocessor(self.preprocessor)
                        logger.debug(f"Set preprocessor on {model.name}")
                except Exception as e:
                    logger.error(f"Error loading {model.name}: {e}")
            else:
                logger.warning(f"Model file not found: {model_file}")
        
        # Ensure all models have the same preprocessor
        if self.preprocessor is not None:
            for model in self.models:
                if model.preprocessor is None:
                    model.set_preprocessor(self.preprocessor)
                    logger.info(f"Set shared preprocessor on {model.name}")
    
    def save_models(self, model_dir: str) -> None:
        """Save all models to disk.
        
        Args:
            model_dir: Directory to save models
        """
        import joblib
        model_dir_path = Path(model_dir)
        model_dir_path.mkdir(parents=True, exist_ok=True)
        
        # Save preprocessor if available
        if self.preprocessor is not None:
            preprocessor_path = model_dir_path / 'preprocessor.pkl'
            try:
                joblib.dump(self.preprocessor, str(preprocessor_path))
                logger.info(f"Saved preprocessor to {preprocessor_path}")
            except Exception as e:
                logger.error(f"Error saving preprocessor: {e}")
        else:
            logger.warning("No preprocessor to save. Models may not load correctly.")
        
        # Save models
        for model in self.models:
            model_file = model_dir_path / f"{model.name.lower().replace(' ', '_')}.pkl"
            try:
                model.save(str(model_file))
                logger.info(f"Saved {model.name} to {model_file}")
            except Exception as e:
                logger.error(f"Error saving {model.name}: {e}")
    
    def update_model_performance(self, model_name: str, accuracy: float) -> None:
        """Update performance tracking for a model.
        
        Args:
            model_name: Name of the model
            accuracy: New accuracy score
        """
        self.model_accuracies[model_name] = accuracy
        logger.info(f"Updated {model_name} accuracy to {accuracy:.1%}")

