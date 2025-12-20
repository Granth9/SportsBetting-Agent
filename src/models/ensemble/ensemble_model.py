"""Ensemble meta-model that combines multiple models."""

import numpy as np
from typing import Dict, Any, List, Optional
from collections import Counter

from src.models.base_model import BaseModel
from src.utils.data_types import ModelPrediction, Outcome
from src.utils.logger import setup_logger


logger = setup_logger(__name__)


class EnsembleModel(BaseModel):
    """Meta-model that combines predictions from multiple base models."""
    
    def __init__(
        self,
        name: str = "Ensemble Council",
        base_models: Optional[List[BaseModel]] = None,
        voting_strategy: str = "weighted",
        weights: Optional[Dict[str, float]] = None
    ):
        """Initialize the ensemble model.
        
        Args:
            name: Model name
            base_models: List of base models to ensemble
            voting_strategy: Strategy for combining predictions ('weighted', 'majority', 'average')
            weights: Dictionary mapping model names to weights (if None, uses equal weights)
        """
        super().__init__(name, "ensemble")
        
        self.base_models = base_models or []
        self.voting_strategy = voting_strategy
        self.weights = weights or {}
        self.model_performances: Dict[str, List[float]] = {}  # Track recent accuracies
        self.model_accuracies: Dict[str, float] = {}  # Store latest accuracy for each model
        self.use_dynamic_weights = False  # Whether to use dynamic weighting
        self.min_model_accuracy: float = 0.65  # Minimum accuracy threshold for model inclusion
        self.use_performance_weights: bool = True  # Use performance-based weighting
        
        # If no weights provided, use equal weights
        if not self.weights and self.base_models:
            self.weights = {model.name: float(1.0 / len(self.base_models)) for model in self.base_models}
        
        # Ensure all weights are floats (convert any strings) - inline conversion since _safe_float not yet defined
        if self.weights:
            cleaned_weights = {}
            default_weight = 1.0 / len(self.base_models) if self.base_models else 1.0
            for name, weight in self.weights.items():
                if isinstance(weight, (int, float, np.number)):
                    cleaned_weights[name] = float(weight)
                elif isinstance(weight, str):
                    try:
                        cleaned_weights[name] = float(weight)
                    except (ValueError, TypeError):
                        cleaned_weights[name] = default_weight
                else:
                    cleaned_weights[name] = default_weight
            self.weights = cleaned_weights
        
        # Initialize performance tracking
        for model in self.base_models:
            self.model_performances[model.name] = []
            if model.name not in self.model_accuracies:
                self.model_accuracies[model.name] = 0.0  # Initialize accuracy tracking
        
        # Ensure all initial weights are floats (convert any strings)
        if self.weights:
            self.weights = {name: float(w) if isinstance(w, (int, float, np.number)) else (float(w) if isinstance(w, str) and w.replace('.', '').replace('-', '').isdigit() else 1.0 / len(self.base_models) if self.base_models else 1.0)
                            for name, w in self.weights.items()}
    
    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        """Safely convert a value to float.
        
        Args:
            value: Value to convert (can be str, int, float, None, etc.)
            default: Default value if conversion fails
            
        Returns:
            float value
        """
        if value is None:
            return float(default)
        if isinstance(value, (int, float, np.number)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except (ValueError, TypeError):
                return float(default)
        return float(default)
    
    def _safe_confidence(self, pred: ModelPrediction) -> float:
        """Safely extract confidence from ModelPrediction.
        
        Args:
            pred: ModelPrediction object
            
        Returns:
            float confidence value (0.0 to 1.0)
        """
        return max(0.0, min(1.0, self._safe_float(pred.confidence, 0.5)))
    
    def add_model(self, model: BaseModel, weight: Optional[float] = None) -> None:
        """Add a base model to the ensemble.
        
        Args:
            model: Base model to add
            weight: Weight for this model (if None, uses equal weight)
        """
        self.base_models.append(model)
        self.model_performances[model.name] = []
        self.model_accuracies[model.name] = 0.0
        
        if weight is not None:
            self.weights[model.name] = self._safe_float(weight, 1.0 / len(self.base_models))
        else:
            # Rebalance weights to be equal
            n = len(self.base_models)
            self.weights = {m.name: float(1.0 / n) for m in self.base_models}
    
    def update_model_accuracy(self, model_name: str, accuracy: float) -> None:
        """Update the accuracy for a specific model.
        
        Args:
            model_name: Name of the model
            accuracy: Accuracy value (0.0 to 1.0)
        """
        accuracy = max(0.0, min(1.0, float(accuracy)))
        self.model_accuracies[model_name] = accuracy
        if model_name in self.model_performances:
            self.model_performances[model_name].append(accuracy)
            # Keep only last 10 accuracies
            if len(self.model_performances[model_name]) > 10:
                self.model_performances[model_name] = self.model_performances[model_name][-10:]
        else:
            self.model_performances[model_name] = [accuracy]
    
    def _filter_weak_models(self, predictions: List[ModelPrediction]) -> List[ModelPrediction]:
        """Filter out predictions from weak models below accuracy threshold.
        
        Args:
            predictions: List of model predictions
            
        Returns:
            Filtered list of predictions (only from models above threshold)
        """
        filtered = []
        excluded = []
        
        for pred in predictions:
            model_name = pred.model_name
            accuracy = self.model_accuracies.get(model_name, 0.0)
            
            if accuracy >= self.min_model_accuracy:
                filtered.append(pred)
            else:
                excluded.append((model_name, accuracy))
        
        if excluded:
            excluded_str = ", ".join([f"{name} ({acc:.1%})" for name, acc in excluded])
            logger.debug(f"Filtered out {len(excluded)} weak model(s): {excluded_str}")
        
        return filtered
    
    def _calculate_performance_weights(self, models: List[BaseModel], baseline_accuracy: float = 0.65) -> Dict[str, float]:
        """Calculate performance-based weights for models.
        
        Args:
            models: List of base models
            baseline_accuracy: Baseline accuracy for weighting (default 0.65)
            
        Returns:
            Dictionary mapping model names to performance weights
        """
        performance_weights = {}
        
        for model in models:
            model_name = model.name
            accuracy = self.model_accuracies.get(model_name, 0.0)
            
            # Filter out weak models
            if accuracy < self.min_model_accuracy:
                performance_weights[model_name] = 0.0
                logger.debug(f"Excluding {model_name} (accuracy: {accuracy:.1%} < threshold: {self.min_model_accuracy:.1%})")
            else:
                # Exponential weighting: better models get exponentially more weight
                # Formula: weight = exp((accuracy - baseline) * 10)
                # This gives much higher weight to models above baseline
                weight = np.exp((accuracy - baseline_accuracy) * 10)
                performance_weights[model_name] = float(weight)
        
        # Normalize weights
        total_weight = sum(performance_weights.values())
        if total_weight > 0:
            performance_weights = {name: float(w / total_weight) for name, w in performance_weights.items()}
        else:
            # Fallback to equal weights if all models filtered
            logger.warning("All models filtered out, using equal weights")
            performance_weights = {model.name: float(1.0 / len(models)) for model in models}
        
        return performance_weights
    
    def train(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None, **kwargs) -> None:
        """Train all base models.
        
        Args:
            X: Training features
            y: Training targets
            X_val: Validation features (optional, passed to base models)
            y_val: Validation targets (optional, passed to base models)
            **kwargs: Additional training parameters
        """
        if not self.base_models:
            raise ValueError("No base models to train")
        
        logger.info(f"Training {self.name} with {len(self.base_models)} base models")
        
        # Train each base model
        for model in self.base_models:
            if not model.is_trained:
                logger.info(f"Training base model: {model.name}")
                # Pass validation data if available
                if X_val is not None and y_val is not None:
                    model.train(X, y, X_val, y_val, **kwargs)
                else:
                    model.train(X, y, **kwargs)
            else:
                logger.info(f"Base model {model.name} already trained, skipping")
        
        # Set feature names from first model
        if self.base_models and self.base_models[0].feature_names:
            self.feature_names = self.base_models[0].feature_names
        
        self.is_trained = True
        logger.info(f"{self.name} training completed")
    
    def predict(self, X: np.ndarray) -> ModelPrediction:
        """Make a prediction by combining base model predictions.
        
        Args:
            X: Input features
            
        Returns:
            ModelPrediction object
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        if not self.base_models:
            raise ValueError("No base models available")
        
        # Validate that all models have consistent feature expectations
        self._validate_model_feature_consistency()
        
        # Get predictions from all base models
        base_predictions = []
        for model in self.base_models:
            if not model.is_trained:
                logger.warning(f"Base model {model.name} is not trained, skipping")
                continue
            
            try:
                pred = model.predict(X)
                base_predictions.append(pred)
            except Exception as e:
                logger.error(f"Error getting prediction from {model.name}: {e}")
                continue
        
        if not base_predictions:
            raise ValueError("No valid predictions from base models")
        
        # Filter weak models if performance weighting is enabled (before voting)
        if self.use_performance_weights:
            original_count = len(base_predictions)
            base_predictions = self._filter_weak_models(base_predictions)
            if len(base_predictions) < original_count:
                logger.debug(f"Filtered {original_count - len(base_predictions)} weak model(s) from ensemble voting")
            if not base_predictions:
                logger.warning("All models filtered out, using all models for fallback")
                # Recreate predictions without filtering for fallback
                base_predictions = []
                for model in self.base_models:
                    if model.is_trained:
                        try:
                            pred = model.predict(X)
                            base_predictions.append(pred)
                        except Exception:
                            continue
        
        # Combine predictions based on strategy
        if self.voting_strategy == "selective_75":
            # 75%+ accuracy mode: unanimous + 70% confidence (spread checked externally)
            final_prediction, final_confidence, probabilities = self._selective_75_vote(base_predictions)
        elif self.voting_strategy == "super_confident":
            # Strictest mode: all models agree, high avg confidence, all individual confidences > 55%
            final_prediction, final_confidence, probabilities = self._super_confident_vote(base_predictions)
        elif self.voting_strategy == "unanimous_high_confidence":
            # Only predict when all models agree AND confidence is high (70% accuracy)
            final_prediction, final_confidence, probabilities = self._unanimous_high_confidence_vote(base_predictions)
        elif self.voting_strategy == "weighted_confidence":
            # Weight by confidence scores (higher confidence = more weight)
            final_prediction, final_confidence, probabilities = self._weighted_confidence_vote(base_predictions)
        elif self.voting_strategy == "weighted":
            final_prediction, final_confidence, probabilities = self._weighted_vote(base_predictions)
        elif self.voting_strategy == "majority":
            final_prediction, final_confidence, probabilities = self._majority_vote(base_predictions)
        else:  # average
            final_prediction, final_confidence, probabilities = self._average_vote(base_predictions)
        
        # Aggregate key features from all models
        key_features = self._aggregate_features(base_predictions)
        
        # Generate reasoning
        model_names = [p.model_name for p in base_predictions]
        reasoning = f"Ensemble ({self.voting_strategy} voting) combines {len(base_predictions)} models: "
        reasoning += f"{', '.join(model_names)}. "
        
        # Ensure final_confidence is a float
        final_confidence = self._safe_float(final_confidence, 0.5)
        final_confidence = max(0.0, min(1.0, float(final_confidence)))  # Clamp to [0, 1]
        
        # Ensure probability is a float
        prob_value = probabilities.get(final_prediction, final_confidence)
        prob_value = self._safe_float(prob_value, final_confidence)
        prob_value = max(0.0, min(1.0, float(prob_value)))  # Clamp to [0, 1]
        
        reasoning += f"Final prediction: {final_prediction.value} with {final_confidence:.1%} confidence."
        
        return ModelPrediction(
            model_name=self.name,
            prediction=final_prediction,
            confidence=float(final_confidence),
            probability=float(prob_value),
            key_features=key_features,
            reasoning=reasoning
        )
    
    def predict_proba(self, X: np.ndarray) -> Dict[Outcome, float]:
        """Get probability distribution by averaging base model probabilities.
        
        Args:
            X: Input features
            
        Returns:
            Dictionary of outcome probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        if not self.base_models:
            raise ValueError("No base models available")
        
        # Get probabilities from all base models
        all_probs = []
        model_weights = []
        
        for model in self.base_models:
            if not model.is_trained:
                continue
            
            try:
                probs = model.predict_proba(X)
                # Ensure probs is a dictionary
                if not isinstance(probs, dict):
                    logger.warning(f"{model.name} returned non-dict probabilities: {type(probs)}")
                    continue
                
                # Ensure all probability values are floats, not strings
                cleaned_probs = {}
                for outcome in Outcome:
                    prob_value = probs.get(outcome, 0.0)
                    # Comprehensive type checking and conversion
                    if prob_value is None:
                        cleaned_probs[outcome] = 0.0
                    elif isinstance(prob_value, str):
                        try:
                            cleaned_probs[outcome] = float(prob_value)
                        except (ValueError, TypeError) as e:
                            logger.warning(f"{model.name} returned invalid prob string '{prob_value}' for {outcome}: {e}")
                            cleaned_probs[outcome] = 0.0
                    elif isinstance(prob_value, (int, float, np.number)):
                        cleaned_probs[outcome] = float(prob_value)
                    else:
                        logger.warning(f"{model.name} returned unexpected prob type {type(prob_value)} for {outcome}: {prob_value}")
                        cleaned_probs[outcome] = 0.0
                
                # Validate all values are floats
                for outcome, val in cleaned_probs.items():
                    if not isinstance(val, (int, float, np.number)):
                        logger.error(f"{model.name} cleaned prob for {outcome} is still not numeric: {type(val)} = {val}")
                        cleaned_probs[outcome] = 0.0
                
                all_probs.append(cleaned_probs)
                weight = self.weights.get(model.name, 1.0 / len(self.base_models))
                weight = self._safe_float(weight, 1.0 / len(self.base_models))
                # Validate weight is float
                if not isinstance(weight, (int, float, np.number)):
                    logger.error(f"Weight for {model.name} is not numeric: {type(weight)} = {weight}")
                    weight = 1.0 / len(self.base_models)
                model_weights.append(float(weight))
            except Exception as e:
                logger.error(f"Error getting probabilities from {model.name}: {e}", exc_info=True)
                continue
        
        if not all_probs:
            raise ValueError("No valid probabilities from base models")
        
        # Normalize weights
        total_weight = sum(model_weights)
        if total_weight > 0:
            model_weights = [self._safe_float(w / total_weight, 1.0 / len(model_weights)) for w in model_weights]
        else:
            # Fallback to equal weights
            model_weights = [float(1.0 / len(model_weights))] * len(model_weights)
        
        # Weighted average of probabilities
        final_probs = {}
        for outcome in Outcome:
            weighted_sum = 0.0
            for idx, (probs, weight) in enumerate(zip(all_probs, model_weights)):
                try:
                    # Get probability value
                    prob_value = probs.get(outcome, 0.0)
                    
                    # Comprehensive type checking and conversion
                    if prob_value is None:
                        prob_value = 0.0
                    elif isinstance(prob_value, str):
                        try:
                            prob_value = float(prob_value)
                        except (ValueError, TypeError):
                            logger.warning(f"Could not convert prob string '{prob_value}' to float for {outcome}, using 0.0")
                            prob_value = 0.0
                    elif isinstance(prob_value, (int, float, np.number)):
                        prob_value = float(prob_value)
                    else:
                        logger.warning(f"Unexpected prob type {type(prob_value)} for {outcome}, using 0.0")
                        prob_value = 0.0
                    
                    # Ensure weight is float
                    if isinstance(weight, str):
                        try:
                            weight = float(weight)
                        except (ValueError, TypeError):
                            logger.warning(f"Could not convert weight string '{weight}' to float, using 1.0")
                            weight = 1.0
                    elif isinstance(weight, (int, float, np.number)):
                        weight = float(weight)
                    else:
                        logger.warning(f"Unexpected weight type {type(weight)}, using 1.0")
                        weight = 1.0
                    
                    # Final type check before arithmetic
                    if not isinstance(prob_value, (int, float, np.number)):
                        logger.error(f"prob_value still not numeric after conversion: {type(prob_value)} = {prob_value}")
                        prob_value = 0.0
                    if not isinstance(weight, (int, float, np.number)):
                        logger.error(f"weight still not numeric after conversion: {type(weight)} = {weight}")
                        weight = 1.0
                    
                    # Perform arithmetic with explicit float conversion
                    contribution = float(prob_value) * float(weight)
                    weighted_sum = float(weighted_sum) + contribution
                    
                except Exception as e:
                    logger.error(f"Error aggregating probability for {outcome} from model {idx}: {e}, prob_value={prob_value} (type: {type(prob_value)}), weight={weight} (type: {type(weight)})", exc_info=True)
                    # Use default values on error
                    weighted_sum = float(weighted_sum) + 0.0
            
            final_probs[outcome] = float(weighted_sum)
        
        return final_probs
    
    def _selective_75_vote(self, predictions: List[ModelPrediction], 
                           confidence_threshold: float = 0.70) -> tuple:
        """Selective 75%+ accuracy mode.
        
        This strategy achieves 75%+ accuracy by requiring:
        1. All models agree unanimously
        2. Average confidence > 70%
        
        Note: Spread filtering (>= 5 points) should be done externally
        before calling predict() for maximum accuracy.
        
        Args:
            predictions: List of base model predictions
            confidence_threshold: Minimum confidence required (default 0.70)
            
        Returns:
            Tuple of (final_prediction, final_confidence, probabilities)
        """
        # Check if all models agree
        votes = [p.prediction for p in predictions]
        all_home = all(v == Outcome.HOME_WIN for v in votes)
        all_away = all(v == Outcome.AWAY_WIN for v in votes)
        
        # Calculate average confidence - ensure all are floats
        confidences = [float(self._safe_confidence(p)) for p in predictions]
        avg_confidence = float(np.mean(confidences))
        
        # Must be unanimous AND confident
        if (all_home or all_away) and avg_confidence >= confidence_threshold:
            final_prediction = Outcome.HOME_WIN if all_home else Outcome.AWAY_WIN
            final_confidence = float(avg_confidence)
            home_prob = float(avg_confidence if all_home else 1 - avg_confidence)
            away_prob = float(1 - avg_confidence if all_home else avg_confidence)
            probabilities = {
                Outcome.HOME_WIN: home_prob,
                Outcome.AWAY_WIN: away_prob
            }
            return final_prediction, final_confidence, probabilities
        
        # Not confident enough - return low confidence prediction
        # This signals to skip this bet
        return Outcome.HOME_WIN, 0.50, {Outcome.HOME_WIN: 0.5, Outcome.AWAY_WIN: 0.5}
    
    def should_bet(self, X: np.ndarray, spread: float = 0.0, min_spread: float = 5.0) -> bool:
        """Check if we should place a bet based on selective criteria.
        
        Args:
            X: Input features
            spread: The point spread for the game (absolute value)
            min_spread: Minimum spread required (default 5.0)
            
        Returns:
            True if bet meets criteria, False otherwise
        """
        # Check spread requirement
        if abs(spread) < min_spread:
            return False
        
        # Get prediction
        try:
            pred = self.predict(X)
            # Check if confident (not the 0.50 fallback)
            return self._safe_confidence(pred) >= 0.60
        except:
            return False
    
    def _super_confident_vote(self, predictions: List[ModelPrediction], 
                               avg_confidence_threshold: float = 0.70,
                               min_individual_confidence: float = 0.55) -> tuple:
        """Super confident mode - strictest prediction criteria.
        
        This strategy aims for maximum accuracy by only predicting when:
        1. All models agree unanimously
        2. Average confidence > 70%
        3. All individual model confidences > 55%
        
        Args:
            predictions: List of base model predictions
            avg_confidence_threshold: Minimum average confidence (default 0.70)
            min_individual_confidence: Minimum individual confidence (default 0.55)
            
        Returns:
            Tuple of (final_prediction, final_confidence, probabilities)
        """
        # Check if all models agree
        votes = [p.prediction for p in predictions]
        all_home = all(v == Outcome.HOME_WIN for v in votes)
        all_away = all(v == Outcome.AWAY_WIN for v in votes)
        
        # Check individual confidences
        individual_confidences = [self._safe_confidence(p) for p in predictions]
        all_confident = all(c >= min_individual_confidence for c in individual_confidences)
        
        # Calculate average confidence
        avg_confidence = np.mean(individual_confidences)
        
        # Super confident: unanimous + high avg + all individuals confident
        if (all_home or all_away) and avg_confidence >= avg_confidence_threshold and all_confident:
            final_prediction = Outcome.HOME_WIN if all_home else Outcome.AWAY_WIN
            final_confidence = avg_confidence
            probabilities = {
                Outcome.HOME_WIN: avg_confidence if all_home else 1 - avg_confidence,
                Outcome.AWAY_WIN: 1 - avg_confidence if all_home else avg_confidence
            }
            return final_prediction, final_confidence, probabilities
        
        # Not confident enough - fall back to unanimous_high_confidence
        return self._unanimous_high_confidence_vote(predictions)
    
    def _unanimous_high_confidence_vote(self, predictions: List[ModelPrediction], confidence_threshold: float = 0.6) -> tuple:
        """Unanimous agreement with high confidence threshold.
        
        This strategy achieves ~70% accuracy by only predicting when:
        1. All models agree on the prediction
        2. Average confidence is above the threshold
        
        Args:
            predictions: List of base model predictions
            confidence_threshold: Minimum confidence required (default 0.6)
            
        Returns:
            Tuple of (final_prediction, final_confidence, probabilities)
        """
        # Check if all models agree
        votes = [p.prediction for p in predictions]
        all_home = all(v == Outcome.HOME_WIN for v in votes)
        all_away = all(v == Outcome.AWAY_WIN for v in votes)
        
        # Calculate average confidence - ensure all are floats
        confidences = [float(self._safe_confidence(p)) for p in predictions]
        avg_confidence = float(np.mean(confidences))
        
        # Check if unanimous and high confidence
        if (all_home or all_away) and (avg_confidence > confidence_threshold or avg_confidence < (1 - confidence_threshold)):
            final_prediction = Outcome.HOME_WIN if all_home else Outcome.AWAY_WIN
            final_confidence = float(avg_confidence)
            home_prob = float(avg_confidence if all_home else 1 - avg_confidence)
            away_prob = float(1 - avg_confidence if all_home else avg_confidence)
            probabilities = {
                Outcome.HOME_WIN: home_prob,
                Outcome.AWAY_WIN: away_prob
            }
        else:
            # Not confident enough - fall back to weighted confidence vote
            return self._weighted_confidence_vote(predictions)
        
        return final_prediction, final_confidence, probabilities
    
    def _weighted_confidence_vote(self, predictions: List[ModelPrediction]) -> tuple:
        """Weighted voting by confidence scores and model performance.
        
        Enhanced to use performance-based weights combined with confidence scores.
        Formula: final_weight = performance_weight * confidence_score
        
        Args:
            predictions: List of base model predictions
            
        Returns:
            Tuple of (final_prediction, final_confidence, probabilities)
        """
        from collections import defaultdict
        
        # Filter weak models if performance weighting is enabled
        if self.use_performance_weights:
            predictions = self._filter_weak_models(predictions)
            if not predictions:
                logger.warning("All models filtered out, falling back to majority vote")
                # Get all predictions back for fallback
                all_models = [m for m in self.base_models if m.is_trained]
                if all_models:
                    # Recreate predictions without filtering
                    return self._majority_vote([p for p in predictions if p.model_name in [m.name for m in all_models]])
                return self._majority_vote(predictions)
        
        # Get performance weights if enabled
        if self.use_performance_weights and self.model_accuracies:
            # Get models that made predictions
            pred_models = [m for m in self.base_models if any(p.model_name == m.name for p in predictions)]
            performance_weights = self._calculate_performance_weights(pred_models)
        else:
            performance_weights = {}
        
        # Weight each prediction by confidence AND performance
        weighted_votes = {}
        total_weight = 0.0
        
        for pred in predictions:
            confidence = self._safe_confidence(pred)
            confidence = float(confidence)  # Ensure float
            outcome = pred.prediction
            model_name = pred.model_name
            
            # Get performance weight (default to 1.0 if not using performance weights)
            perf_weight = float(performance_weights.get(model_name, 1.0)) if performance_weights else 1.0
            
            # Combined weight: performance * confidence
            combined_weight = float(perf_weight) * float(confidence)
            
            # Initialize if not present
            if outcome not in weighted_votes:
                weighted_votes[outcome] = 0.0
            
            # Ensure current value is float before addition
            current_value = self._safe_float(weighted_votes.get(outcome, 0.0), 0.0)
            weighted_votes[outcome] = float(current_value) + float(combined_weight)
            total_weight = float(total_weight) + float(combined_weight)
        
        # Normalize weights - ensure total_weight is float
        total_weight = float(total_weight)
        if total_weight > 0:
            for outcome in list(weighted_votes.keys()):
                current_val = self._safe_float(weighted_votes[outcome], 0.0)
                weighted_votes[outcome] = float(current_val) / float(total_weight)
        else:
            # Fallback to majority vote
            return self._majority_vote(predictions)
        
        # Find outcome with highest weighted vote - ensure all values are floats
        weighted_votes_float = {k: self._safe_float(v, 0.0) for k, v in weighted_votes.items()}
        
        if weighted_votes_float:
            final_prediction = max(weighted_votes_float.items(), key=lambda x: float(x[1]))[0]
            final_confidence = float(weighted_votes_float[final_prediction])
        else:
            # Fallback to majority vote
            return self._majority_vote(predictions)
        
        # Calculate probabilities - ensure all are floats
        probabilities = {outcome: float(self._safe_float(prob, 0.0)) for outcome, prob in weighted_votes_float.items()}
        
        return final_prediction, final_confidence, probabilities
    
    def _weighted_vote(self, predictions: List[ModelPrediction]) -> tuple:
        """Weighted voting strategy.
        
        Uses dynamic weights if enabled, otherwise uses static weights.
        
        Args:
            predictions: List of base model predictions
            
        Returns:
            Tuple of (final_prediction, final_confidence, probabilities)
        """
        # Get weights (dynamic or static)
        if self.use_dynamic_weights:
            weights = self._calculate_dynamic_weights()
        else:
            weights = self.weights
        
        # Ensure all weights are floats
        weights = {name: self._safe_float(weight, 1.0 / len(predictions)) 
                   for name, weight in weights.items()}
        
        # Get probabilities from all predictions
        outcome_votes = {}
        outcome_weights = {}
        
        for pred in predictions:
            weight = self._safe_float(weights.get(pred.model_name, 1.0 / len(predictions)), 1.0 / len(predictions))
            confidence = self._safe_confidence(pred)
            
            # Ensure weight and confidence are floats
            weight = float(weight)
            confidence = float(confidence)
            
            # Vote for the predicted outcome
            if pred.prediction not in outcome_votes:
                outcome_votes[pred.prediction] = 0.0
                outcome_weights[pred.prediction] = 0.0
            
            outcome_votes[pred.prediction] = float(outcome_votes[pred.prediction]) + weight
            outcome_weights[pred.prediction] = float(outcome_weights[pred.prediction]) + (confidence * weight)
        
        # Find outcome with highest weighted vote - ensure all values are floats
        outcome_votes_float = {k: self._safe_float(v, 0.0) for k, v in outcome_votes.items()}
        outcome_weights_float = {k: self._safe_float(v, 0.0) for k, v in outcome_weights.items()}
        
        final_prediction = max(outcome_votes_float.items(), key=lambda x: float(x[1]))[0]
        votes_val = float(outcome_votes_float.get(final_prediction, 0.0))
        weights_val = float(outcome_weights_float.get(final_prediction, 0.0))
        final_confidence = float(weights_val / votes_val) if votes_val > 0 else 0.5
        
        # Normalize votes to probabilities - ensure all are floats
        total_votes = float(sum(outcome_votes_float.values()))
        probabilities = {outcome: float(self._safe_float(votes, 0.0) / total_votes) if total_votes > 0 else 0.0 
                        for outcome, votes in outcome_votes_float.items()}
        
        return final_prediction, final_confidence, probabilities
    
    def _calculate_dynamic_weights(self, n_recent: int = 50) -> Dict[str, float]:
        """Calculate dynamic weights based on recent model performance.
        
        Args:
            n_recent: Number of recent predictions to consider
            
        Returns:
            Dictionary mapping model names to weights
        """
        weights = {}
        total_performance = 0.0
        
        for model in self.base_models:
            model_name = model.name
            performances = self.model_performances.get(model_name, [])
            
            if performances:
                # Use recent performance (last n_recent predictions)
                recent_perfs = performances[-n_recent:]
                avg_performance = np.mean(recent_perfs)
                # Use model's recent accuracy if available
                recent_accuracy = model.get_recent_accuracy(n=n_recent)
                # Combine both metrics
                performance_score = (avg_performance + recent_accuracy) / 2.0
            else:
                # Fall back to model's recent accuracy
                performance_score = model.get_recent_accuracy(n=n_recent)
            
            # Ensure minimum weight to avoid zero weights
            performance_score = max(performance_score, 0.4)  # Minimum 40% accuracy assumption
            performance_score = self._safe_float(performance_score, 0.4)
            weights[model_name] = performance_score
            total_performance += performance_score
        
        # Normalize weights
        if total_performance > 0:
            weights = {name: self._safe_float(w / total_performance, 1.0 / len(self.base_models)) 
                       for name, w in weights.items()}
        else:
            # Fall back to equal weights
            weights = {model.name: float(1.0 / len(self.base_models)) for model in self.base_models}
        
        return weights
    
    def update_model_performance(self, model_name: str, is_correct: bool) -> None:
        """Update performance tracking for a model.
        
        Args:
            model_name: Name of the model
            is_correct: Whether the prediction was correct
        """
        if model_name not in self.model_performances:
            self.model_performances[model_name] = []
        
        # Add result (1.0 for correct, 0.0 for incorrect)
        self.model_performances[model_name].append(1.0 if is_correct else 0.0)
        
        # Keep only recent performance (last 100 predictions)
        if len(self.model_performances[model_name]) > 100:
            self.model_performances[model_name] = self.model_performances[model_name][-100:]
    
    def enable_dynamic_weights(self) -> None:
        """Enable dynamic weighting based on recent performance."""
        self.use_dynamic_weights = True
        logger.info(f"{self.name} enabled dynamic weighting")
    
    def disable_dynamic_weights(self) -> None:
        """Disable dynamic weighting, use static weights."""
        self.use_dynamic_weights = False
        logger.info(f"{self.name} disabled dynamic weighting, using static weights")
    
    def _majority_vote(self, predictions: List[ModelPrediction]) -> tuple:
        """Majority voting strategy.
        
        Args:
            predictions: List of base model predictions
            
        Returns:
            Tuple of (final_prediction, final_confidence, probabilities)
        """
        # Count votes for each outcome
        outcome_counts = Counter(pred.prediction for pred in predictions)
        final_prediction = outcome_counts.most_common(1)[0][0]
        
        # Average confidence of models that voted for the winning outcome
        winning_predictions = [pred for pred in predictions if pred.prediction == final_prediction]
        final_confidence = np.mean([self._safe_confidence(pred) for pred in winning_predictions]) if winning_predictions else 0.5
        
        # Convert counts to probabilities
        total = len(predictions)
        probabilities = {outcome: count / total for outcome, count in outcome_counts.items()}
        
        return final_prediction, final_confidence, probabilities
    
    def _average_vote(self, predictions: List[ModelPrediction]) -> tuple:
        """Average voting strategy (average probabilities).
        
        Args:
            predictions: List of base model predictions
            
        Returns:
            Tuple of (final_prediction, final_confidence, probabilities)
        """
        # Average probabilities across all models
        all_probs = []
        for pred in predictions:
            # Get probabilities from prediction
            prob_value = self._safe_float(pred.probability, 0.5) if pred.probability is not None else 0.5
            prob_home = prob_value if pred.prediction == Outcome.HOME_WIN else (1 - prob_value)
            all_probs.append({
                Outcome.HOME_WIN: prob_home,
                Outcome.AWAY_WIN: 1 - prob_home
            })
        
        # Average probabilities
        avg_probs = {}
        for outcome in [Outcome.HOME_WIN, Outcome.AWAY_WIN]:
            avg_probs[outcome] = np.mean([probs.get(outcome, 0.0) for probs in all_probs])
        
        # Normalize
        total = sum(avg_probs.values())
        if total > 0:
            avg_probs = {k: v / total for k, v in avg_probs.items()}
        
        # Find outcome with highest probability
        final_prediction = max(avg_probs.items(), key=lambda x: x[1])[0]
        final_confidence = avg_probs[final_prediction]
        
        return final_prediction, final_confidence, avg_probs
    
    def _aggregate_features(self, predictions: List[ModelPrediction]) -> Dict[str, Any]:
        """Aggregate key features from all model predictions.
        
        Args:
            predictions: List of base model predictions
            
        Returns:
            Dictionary of aggregated features
        """
        # Collect all key features
        all_features = {}
        feature_counts = {}
        
        for pred in predictions:
            if pred.key_features:
                for feature, value in pred.key_features.items():
                    if feature not in all_features:
                        all_features[feature] = 0.0
                        feature_counts[feature] = 0
                    
                    # Ensure value is numeric before adding
                    numeric_value = self._safe_float(value, 0.0)
                    all_features[feature] = float(all_features[feature]) + float(numeric_value)
                    feature_counts[feature] = int(feature_counts[feature]) + 1
        
        # Average feature importance - ensure all values are floats
        aggregated = {}
        for feature, total_value in all_features.items():
            count = feature_counts[feature]
            if count > 0:
                total_val = self._safe_float(total_value, 0.0)
                count_val = self._safe_float(count, 1.0)
                aggregated[feature] = float(total_val) / float(count_val)
        
        # Return top 5 features - ensure sort key is float
        top_features = dict(sorted(aggregated.items(), key=lambda x: float(self._safe_float(x[1], 0.0)), reverse=True)[:5])
        
        return {
            'top_features': top_features,
            'n_models': len(predictions),
            'voting_strategy': self.voting_strategy
        }
    
    def _validate_model_feature_consistency(self) -> None:
        """Validate that all base models have consistent feature expectations.
        
        Logs warnings if models have different feature names or preprocessors.
        """
        if not self.base_models:
            return
        
        # Collect feature names from all models
        feature_sets = []
        preprocessor_types = []
        
        for model in self.base_models:
            if not model.is_trained:
                continue
            
            # Check feature names
            if model.feature_names:
                feature_sets.append(set(model.feature_names))
            
            # Check preprocessor
            if model.preprocessor is not None:
                preprocessor_type = type(model.preprocessor).__name__
                preprocessor_types.append(preprocessor_type)
        
        # Check for feature name mismatches
        if len(feature_sets) > 1:
            # Find common features
            common_features = set.intersection(*feature_sets) if feature_sets else set()
            
            # Find unique features per model
            all_features = set.union(*feature_sets) if feature_sets else set()
            unique_features = all_features - common_features
            
            if unique_features:
                logger.warning(f"Models have {len(unique_features)} inconsistent feature names. "
                             f"This may cause prediction errors.")
                logger.debug(f"Unique features: {list(unique_features)[:10]}")
        
        # Check for preprocessor consistency
        if len(set(preprocessor_types)) > 1:
            logger.warning(f"Models have different preprocessor types: {set(preprocessor_types)}. "
                          f"This may cause feature scaling mismatches.")
        elif len(preprocessor_types) > 0 and len(preprocessor_types) < len([m for m in self.base_models if m.is_trained]):
            logger.warning(f"Some models have preprocessors while others don't. "
                          f"This may cause feature scaling mismatches.")
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get aggregated feature importance from base models.
        
        Returns:
            Dictionary of feature importances
        """
        if not self.base_models:
            return None
        
        # Collect feature importances from all models
        all_importances = {}
        
        for model in self.base_models:
            if not model.is_trained:
                continue
            
            importance = model.get_feature_importance()
            if importance:
                weight = self._safe_float(self.weights.get(model.name, 1.0 / len(self.base_models)), 1.0 / len(self.base_models))
                
                for feature, value in importance.items():
                    if feature not in all_importances:
                        all_importances[feature] = 0.0
                    all_importances[feature] += value * weight
        
        # Normalize
        total = sum(all_importances.values())
        if total > 0:
            all_importances = {k: v / total for k, v in all_importances.items()}
        
        return all_importances if all_importances else None

