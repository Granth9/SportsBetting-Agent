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
        
        # If no weights provided, use equal weights
        if not self.weights and self.base_models:
            self.weights = {model.name: 1.0 / len(self.base_models) for model in self.base_models}
    
    def add_model(self, model: BaseModel, weight: Optional[float] = None) -> None:
        """Add a base model to the ensemble.
        
        Args:
            model: Base model to add
            weight: Weight for this model (if None, uses equal weight)
        """
        self.base_models.append(model)
        
        if weight is not None:
            self.weights[model.name] = weight
        else:
            # Rebalance weights to be equal
            n = len(self.base_models)
            self.weights = {m.name: 1.0 / n for m in self.base_models}
    
    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """Train all base models.
        
        Args:
            X: Training features
            y: Training targets
            **kwargs: Additional training parameters
        """
        if not self.base_models:
            raise ValueError("No base models to train")
        
        logger.info(f"Training {self.name} with {len(self.base_models)} base models")
        
        # Train each base model
        for model in self.base_models:
            if not model.is_trained:
                logger.info(f"Training base model: {model.name}")
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
        
        # Combine predictions based on strategy
        if self.voting_strategy == "weighted":
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
        reasoning += f"Final prediction: {final_prediction.value} with {final_confidence:.1%} confidence."
        
        return ModelPrediction(
            model_name=self.name,
            prediction=final_prediction,
            confidence=final_confidence,
            probability=probabilities.get(final_prediction, final_confidence),
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
                all_probs.append(probs)
                weight = self.weights.get(model.name, 1.0 / len(self.base_models))
                model_weights.append(weight)
            except Exception as e:
                logger.error(f"Error getting probabilities from {model.name}: {e}")
                continue
        
        if not all_probs:
            raise ValueError("No valid probabilities from base models")
        
        # Normalize weights
        total_weight = sum(model_weights)
        if total_weight > 0:
            model_weights = [w / total_weight for w in model_weights]
        
        # Weighted average of probabilities
        final_probs = {}
        for outcome in Outcome:
            weighted_sum = sum(probs.get(outcome, 0.0) * weight 
                             for probs, weight in zip(all_probs, model_weights))
            final_probs[outcome] = weighted_sum
        
        return final_probs
    
    def _weighted_vote(self, predictions: List[ModelPrediction]) -> tuple:
        """Weighted voting strategy.
        
        Args:
            predictions: List of base model predictions
            
        Returns:
            Tuple of (final_prediction, final_confidence, probabilities)
        """
        # Get probabilities from all predictions
        outcome_votes = {}
        outcome_weights = {}
        
        for pred in predictions:
            weight = self.weights.get(pred.model_name, 1.0 / len(predictions))
            
            # Vote for the predicted outcome
            if pred.prediction not in outcome_votes:
                outcome_votes[pred.prediction] = 0.0
                outcome_weights[pred.prediction] = 0.0
            
            outcome_votes[pred.prediction] += weight
            outcome_weights[pred.prediction] += pred.confidence * weight
        
        # Find outcome with highest weighted vote
        final_prediction = max(outcome_votes.items(), key=lambda x: x[1])[0]
        final_confidence = outcome_weights[final_prediction] / outcome_votes[final_prediction] if outcome_votes[final_prediction] > 0 else 0.5
        
        # Normalize votes to probabilities
        total_votes = sum(outcome_votes.values())
        probabilities = {outcome: votes / total_votes if total_votes > 0 else 0.0 
                        for outcome, votes in outcome_votes.items()}
        
        return final_prediction, final_confidence, probabilities
    
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
        final_confidence = np.mean([pred.confidence for pred in winning_predictions]) if winning_predictions else 0.5
        
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
            prob_home = pred.probability if pred.prediction == Outcome.HOME_WIN else (1 - pred.probability)
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
                    
                    all_features[feature] += value
                    feature_counts[feature] += 1
        
        # Average feature importance
        aggregated = {}
        for feature, total_value in all_features.items():
            count = feature_counts[feature]
            if count > 0:
                aggregated[feature] = total_value / count
        
        # Return top 5 features
        top_features = dict(sorted(aggregated.items(), key=lambda x: x[1], reverse=True)[:5])
        
        return {
            'top_features': top_features,
            'n_models': len(predictions),
            'voting_strategy': self.voting_strategy
        }
    
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
                weight = self.weights.get(model.name, 1.0 / len(self.base_models))
                
                for feature, value in importance.items():
                    if feature not in all_importances:
                        all_importances[feature] = 0.0
                    all_importances[feature] += value * weight
        
        # Normalize
        total = sum(all_importances.values())
        if total > 0:
            all_importances = {k: v / total for k, v in all_importances.items()}
        
        return all_importances if all_importances else None

