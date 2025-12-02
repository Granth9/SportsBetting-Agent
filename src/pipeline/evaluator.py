"""Backtesting and evaluation framework."""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from src.pipeline.predictor import BettingCouncil
from src.utils.data_types import (
    Proposition,
    GameInfo,
    BettingLine,
    BetType,
    Outcome,
    Recommendation
)
from src.utils.logger import setup_logger


logger = setup_logger(__name__)


class PerformanceEvaluator:
    """Evaluate and backtest betting council performance."""
    
    def __init__(self, council: BettingCouncil):
        """Initialize the evaluator.
        
        Args:
            council: The betting council to evaluate
        """
        self.council = council
        self.results: List[Dict[str, Any]] = []
    
    def backtest(
        self,
        propositions: List[Proposition],
        initial_bankroll: float = 1000.0
    ) -> Dict[str, Any]:
        """Run backtest on historical propositions.
        
        Args:
            propositions: List of historical propositions with known outcomes
            initial_bankroll: Starting bankroll amount
            
        Returns:
            Dictionary of performance metrics
        """
        logger.info(f"Starting backtest on {len(propositions)} propositions")
        
        bankroll = initial_bankroll
        bet_history = []
        
        correct_predictions = 0
        total_predictions = 0
        
        for i, prop in enumerate(propositions):
            if (i + 1) % 10 == 0:
                logger.info(f"Progress: {i+1}/{len(propositions)}")
            
            # Get recommendation
            try:
                recommendation = self.council.analyze(prop)
            except Exception as e:
                logger.error(f"Error analyzing proposition {prop.prop_id}: {e}")
                continue
            
            # Check if we should bet
            if recommendation.recommended_action == "PASS":
                continue
            
            # Calculate bet amount
            bet_amount = 0
            if recommendation.bet_size:
                bet_amount = bankroll * recommendation.bet_size
            else:
                # Default to 2% of bankroll
                bet_amount = bankroll * 0.02
            
            # Evaluate result (need actual outcome from proposition)
            actual_outcome = self._get_actual_outcome(prop)
            predicted_outcome = recommendation.debate_result.final_prediction
            
            is_correct = (actual_outcome == predicted_outcome)
            
            # Calculate profit/loss
            if is_correct:
                # Win the bet
                if prop.line:
                    if predicted_outcome.value in ["HOME_WIN", "home_win"]:
                        odds = prop.line.home_ml
                    else:
                        odds = prop.line.away_ml
                    
                    if odds > 0:
                        profit = bet_amount * (odds / 100)
                    else:
                        profit = bet_amount * (100 / abs(odds))
                else:
                    profit = bet_amount  # Assume even money if no odds
                
                bankroll += profit
                correct_predictions += 1
            else:
                # Lose the bet
                bankroll -= bet_amount
            
            total_predictions += 1
            
            # Record result
            bet_history.append({
                'proposition_id': prop.prop_id,
                'bet_amount': bet_amount,
                'predicted': predicted_outcome.value,
                'actual': actual_outcome.value if actual_outcome else None,
                'correct': is_correct,
                'profit_loss': profit if is_correct else -bet_amount,
                'bankroll': bankroll,
                'confidence': recommendation.debate_result.final_confidence,
                'consensus': recommendation.debate_result.consensus_level
            })
        
        # Calculate metrics
        metrics = self._calculate_metrics(
            bet_history,
            initial_bankroll,
            bankroll,
            correct_predictions,
            total_predictions
        )
        
        # Store results
        self.results = bet_history
        
        logger.info(f"Backtest complete. Final bankroll: ${bankroll:.2f}")
        logger.info(f"Accuracy: {metrics['accuracy']:.1%}")
        logger.info(f"ROI: {metrics['roi']:.1%}")
        
        return metrics
    
    def _get_actual_outcome(self, proposition: Proposition) -> Optional[Outcome]:
        """Get the actual outcome of a proposition.
        
        Args:
            proposition: The proposition
            
        Returns:
            Actual outcome or None if not available
        """
        game_info = proposition.game_info
        
        # Check if we have actual scores
        if game_info.home_score is None or game_info.away_score is None:
            return None
        
        # Determine outcome based on bet type
        if proposition.bet_type == BetType.GAME_OUTCOME:
            if game_info.home_score > game_info.away_score:
                return Outcome.HOME_WIN
            else:
                return Outcome.AWAY_WIN
        
        elif proposition.bet_type == BetType.SPREAD:
            if proposition.line:
                adjusted_home = game_info.home_score + proposition.line.spread
                if adjusted_home > game_info.away_score:
                    return Outcome.COVER
                else:
                    return Outcome.NO_COVER
        
        elif proposition.bet_type == BetType.TOTAL:
            if proposition.line:
                total_points = game_info.home_score + game_info.away_score
                if total_points > proposition.line.total:
                    return Outcome.OVER
                else:
                    return Outcome.UNDER
        
        return None
    
    def _calculate_metrics(
        self,
        bet_history: List[Dict[str, Any]],
        initial_bankroll: float,
        final_bankroll: float,
        correct_predictions: int,
        total_predictions: int
    ) -> Dict[str, Any]:
        """Calculate performance metrics.
        
        Args:
            bet_history: History of bets
            initial_bankroll: Starting bankroll
            final_bankroll: Ending bankroll
            correct_predictions: Number of correct predictions
            total_predictions: Total number of predictions
            
        Returns:
            Dictionary of metrics
        """
        if not bet_history:
            return {
                'accuracy': 0.0,
                'roi': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'total_bets': 0,
                'profit_loss': 0.0
            }
        
        # Basic metrics
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        profit_loss = final_bankroll - initial_bankroll
        roi = profit_loss / initial_bankroll
        
        # Calculate Sharpe ratio
        returns = [bet['profit_loss'] / bet['bet_amount'] for bet in bet_history]
        if returns:
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            sharpe_ratio = (avg_return / std_return) * np.sqrt(252) if std_return > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Calculate max drawdown
        bankrolls = [bet['bankroll'] for bet in bet_history]
        peak = initial_bankroll
        max_drawdown = 0
        
        for bankroll in bankrolls:
            if bankroll > peak:
                peak = bankroll
            drawdown = (peak - bankroll) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        return {
            'accuracy': accuracy,
            'roi': roi,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_bets': len(bet_history),
            'profit_loss': profit_loss,
            'final_bankroll': final_bankroll,
            'win_rate': accuracy
        }
    
    def export_results(self, output_path: str) -> None:
        """Export backtest results to CSV.
        
        Args:
            output_path: Path to save results
        """
        if not self.results:
            logger.warning("No results to export")
            return
        
        df = pd.DataFrame(self.results)
        df.to_csv(output_path, index=False)
        logger.info(f"Exported results to {output_path}")
    
    def evaluate_model_performance(
        self,
        propositions: List[Proposition]
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate individual model performance.
        
        Args:
            propositions: List of propositions
            
        Returns:
            Dictionary mapping model names to performance metrics
        """
        logger.info("Evaluating individual model performance")
        
        model_results = {model.name: [] for model in self.council.models}
        
        for prop in propositions:
            actual_outcome = self._get_actual_outcome(prop)
            if actual_outcome is None:
                continue
            
            # Get features and run each model
            try:
                schedule_df, team_stats_df, player_stats_df = self.council._collect_data(prop)
                features = self.council.feature_engineer.extract_features(
                    prop, schedule_df, team_stats_df, player_stats_df
                )
                
                for model in self.council.models:
                    if not model.is_trained:
                        continue
                    
                    prediction = model.predict_proposition(prop, features)
                    is_correct = (prediction.prediction == actual_outcome)
                    
                    model_results[model.name].append({
                        'correct': is_correct,
                        'confidence': prediction.confidence
                    })
            
            except Exception as e:
                logger.error(f"Error evaluating models on {prop.prop_id}: {e}")
        
        # Calculate metrics for each model
        model_metrics = {}
        for model_name, results in model_results.items():
            if results:
                accuracy = sum(r['correct'] for r in results) / len(results)
                avg_confidence = sum(r['confidence'] for r in results) / len(results)
                
                model_metrics[model_name] = {
                    'accuracy': accuracy,
                    'avg_confidence': avg_confidence,
                    'total_predictions': len(results)
                }
            else:
                model_metrics[model_name] = {
                    'accuracy': 0.0,
                    'avg_confidence': 0.0,
                    'total_predictions': 0
                }
        
        return model_metrics

