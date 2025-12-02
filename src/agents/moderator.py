"""Debate moderator for agent council."""

import anthropic
import re
from typing import List, Dict, Any
from datetime import datetime

from src.utils.data_types import (
    ModelPrediction,
    AgentArgument,
    DebateResult,
    Proposition,
    Outcome
)
from src.utils.config_loader import get_config
from src.utils.logger import setup_logger
from src.agents.debate_agent import DebateAgent
from src.agents.prompts.system_prompts import (
    get_agent_prompt,
    MODERATOR_SYNTHESIS_TEMPLATE
)


logger = setup_logger(__name__)


class DebateModerator:
    """Moderator for multi-agent debate on betting propositions."""
    
    def __init__(
        self,
        num_rounds: int = 4,
        temperature: float = 0.7
    ):
        """Initialize the debate moderator.
        
        Args:
            num_rounds: Number of debate rounds
            temperature: LLM temperature
        """
        self.num_rounds = num_rounds
        self.temperature = temperature
        
        # Get configuration
        config = get_config()
        self.api_key = config.anthropic_api_key
        self.model_name = config.get('anthropic.model', 'claude-3-5-sonnet-20241022')
        
        # Initialize Anthropic client
        self.client = anthropic.Anthropic(api_key=self.api_key)
        
        # Get system prompt
        self.system_prompt = get_agent_prompt('moderator')
    
    def orchestrate_debate(
        self,
        proposition: Proposition,
        model_predictions: List[ModelPrediction],
        model_accuracies: Dict[str, float]
    ) -> DebateResult:
        """Orchestrate a multi-round debate between model agents.
        
        Args:
            proposition: The betting proposition
            model_predictions: Predictions from all models
            model_accuracies: Historical accuracy for each model
            
        Returns:
            DebateResult with final recommendation
        """
        logger.info(f"Starting debate for {proposition.prop_id}")
        
        # Create agents for each model
        agents = self._create_agents(model_predictions, model_accuracies)
        
        # Track all arguments
        all_arguments: List[AgentArgument] = []
        
        # Round 1: Initial statements
        logger.info("Round 1: Initial statements")
        for agent in agents:
            argument = agent.initial_statement(proposition)
            all_arguments.append(argument)
        
        # Rounds 2-3: Rebuttals
        for round_num in range(2, min(4, self.num_rounds)):
            logger.info(f"Round {round_num}: Rebuttals")
            
            # Get arguments from this round
            round_arguments = []
            
            for agent in agents:
                # Show each agent the previous round's arguments
                prev_round_args = [arg for arg in all_arguments if arg.round_number == round_num - 1]
                
                argument = agent.rebuttal(proposition, prev_round_args, round_num)
                round_arguments.append(argument)
            
            all_arguments.extend(round_arguments)
        
        # Final round: Final statements
        logger.info(f"Round {self.num_rounds}: Final statements")
        for agent in agents:
            argument = agent.final_statement(proposition, all_arguments, self.num_rounds)
            all_arguments.append(argument)
        
        # Synthesize final recommendation
        debate_result = self._synthesize_debate(
            proposition,
            model_predictions,
            all_arguments,
            model_accuracies
        )
        
        logger.info(f"Debate completed. Final: {debate_result.final_prediction.value} ({debate_result.final_confidence:.1%})")
        
        return debate_result
    
    def _create_agents(
        self,
        model_predictions: List[ModelPrediction],
        model_accuracies: Dict[str, float]
    ) -> List[DebateAgent]:
        """Create debate agents for each model.
        
        Args:
            model_predictions: Model predictions
            model_accuracies: Historical accuracies
            
        Returns:
            List of DebateAgent instances
        """
        # Map model names to agent types
        model_to_agent_type = {
            "Neural Analyst": "neural_analyst",
            "Gradient Strategist": "gradient_strategist",
            "Forest Evaluator": "forest_evaluator",
            "Statistical Conservative": "statistical_conservative"
        }
        
        agents = []
        for prediction in model_predictions:
            agent_type = model_to_agent_type.get(prediction.model_name, "neural_analyst")
            accuracy = model_accuracies.get(prediction.model_name, 0.5)
            
            agent = DebateAgent(
                agent_type=agent_type,
                model_prediction=prediction,
                historical_accuracy=accuracy,
                temperature=self.temperature
            )
            agents.append(agent)
        
        return agents
    
    def _synthesize_debate(
        self,
        proposition: Proposition,
        model_predictions: List[ModelPrediction],
        debate_transcript: List[AgentArgument],
        model_accuracies: Dict[str, float]
    ) -> DebateResult:
        """Synthesize the debate into a final recommendation.
        
        Args:
            proposition: The betting proposition
            model_predictions: Original model predictions
            debate_transcript: All arguments from the debate
            model_accuracies: Historical accuracies
            
        Returns:
            DebateResult
        """
        logger.info("Synthesizing debate into final recommendation")
        
        # Format model predictions
        predictions_str = self._format_predictions(model_predictions, model_accuracies)
        
        # Format debate transcript
        transcript_str = self._format_transcript(debate_transcript)
        
        # Format proposition
        prop_desc = self._format_proposition(proposition)
        
        # Create synthesis prompt
        user_prompt = MODERATOR_SYNTHESIS_TEMPLATE.format(
            proposition_description=prop_desc,
            model_predictions=predictions_str,
            debate_transcript=transcript_str
        )
        
        # Generate synthesis
        synthesis = self._generate_synthesis(user_prompt)
        
        # Parse the synthesis
        final_prediction, final_confidence, reasoning, consensus = self._parse_synthesis(synthesis)
        
        # If parsing fails, fall back to weighted voting
        if final_prediction is None:
            final_prediction, final_confidence, consensus = self._weighted_voting(
                model_predictions,
                debate_transcript,
                model_accuracies
            )
            reasoning = "Final recommendation based on weighted model predictions."
        
        return DebateResult(
            final_prediction=final_prediction,
            final_confidence=final_confidence,
            consensus_level=consensus,
            debate_transcript=debate_transcript,
            model_predictions=model_predictions,
            reasoning_summary=reasoning,
            timestamp=datetime.now()
        )
    
    def _generate_synthesis(self, user_prompt: str) -> str:
        """Generate synthesis using Claude.
        
        Args:
            user_prompt: The synthesis prompt
            
        Returns:
            Generated synthesis
        """
        try:
            message = self.client.messages.create(
                model=self.model_name,
                max_tokens=1000,
                temperature=self.temperature,
                system=self.system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            return message.content[0].text
        
        except Exception as e:
            logger.error(f"Error generating synthesis: {e}")
            return ""
    
    def _parse_synthesis(self, synthesis: str) -> tuple:
        """Parse the moderator's synthesis.
        
        Args:
            synthesis: The synthesis text
            
        Returns:
            Tuple of (prediction, confidence, reasoning, consensus)
        """
        # Extract final recommendation
        prediction = None
        confidence = 0.5
        consensus = 0.5
        reasoning = synthesis
        
        # Look for FINAL RECOMMENDATION
        match = re.search(r'FINAL RECOMMENDATION:\s*(\w+)', synthesis, re.IGNORECASE)
        if match:
            pred_str = match.group(1).upper()
            for outcome in Outcome:
                if outcome.value.upper().replace('_', '') in pred_str or pred_str in outcome.value.upper():
                    prediction = outcome
                    break
        
        # Look for CONFIDENCE
        match = re.search(r'CONFIDENCE:\s*(\d+)', synthesis, re.IGNORECASE)
        if match:
            confidence = int(match.group(1)) / 100
        
        # Look for CONSENSUS LEVEL
        match = re.search(r'CONSENSUS LEVEL:\s*(\d+)', synthesis, re.IGNORECASE)
        if match:
            consensus = int(match.group(1)) / 100
        
        # Extract reasoning
        match = re.search(r'REASONING:\s*(.+?)(?:CONSENSUS|$)', synthesis, re.IGNORECASE | re.DOTALL)
        if match:
            reasoning = match.group(1).strip()
        
        return prediction, confidence, reasoning, consensus
    
    def _weighted_voting(
        self,
        model_predictions: List[ModelPrediction],
        debate_transcript: List[AgentArgument],
        model_accuracies: Dict[str, float]
    ) -> tuple:
        """Fall back to weighted voting if synthesis fails.
        
        Args:
            model_predictions: Model predictions
            debate_transcript: Debate arguments
            model_accuracies: Historical accuracies
            
        Returns:
            Tuple of (prediction, confidence, consensus)
        """
        # Get final confidences from debate
        final_confidences = {}
        for arg in reversed(debate_transcript):
            if arg.agent_name not in final_confidences:
                final_confidences[arg.agent_name] = arg.confidence
        
        # Weight votes by accuracy and confidence
        votes: Dict[Outcome, float] = {}
        
        for pred in model_predictions:
            accuracy = model_accuracies.get(pred.model_name, 0.5)
            final_conf = final_confidences.get(pred.model_name, pred.confidence)
            
            weight = accuracy * final_conf
            
            if pred.prediction not in votes:
                votes[pred.prediction] = 0
            votes[pred.prediction] += weight
        
        # Get winner
        if votes:
            final_prediction = max(votes, key=votes.get)
            total_weight = sum(votes.values())
            final_confidence = votes[final_prediction] / total_weight if total_weight > 0 else 0.5
            
            # Calculate consensus (how concentrated the votes are)
            consensus = votes[final_prediction] / total_weight if total_weight > 0 else 0.5
        else:
            final_prediction = Outcome.HOME_WIN
            final_confidence = 0.5
            consensus = 0.5
        
        return final_prediction, final_confidence, consensus
    
    def _format_predictions(
        self,
        predictions: List[ModelPrediction],
        accuracies: Dict[str, float]
    ) -> str:
        """Format model predictions for display.
        
        Args:
            predictions: Model predictions
            accuracies: Historical accuracies
            
        Returns:
            Formatted string
        """
        lines = []
        for pred in predictions:
            accuracy = accuracies.get(pred.model_name, 0.5)
            line = f"{pred.model_name} (accuracy: {accuracy:.1%}): "
            line += f"{pred.prediction.value} ({pred.confidence:.1%} confident)"
            lines.append(line)
        
        return "\n".join(lines)
    
    def _format_transcript(self, transcript: List[AgentArgument]) -> str:
        """Format debate transcript.
        
        Args:
            transcript: All arguments
            
        Returns:
            Formatted string
        """
        lines = []
        current_round = 0
        
        for arg in transcript:
            if arg.round_number != current_round:
                lines.append(f"\n--- Round {arg.round_number} ---")
                current_round = arg.round_number
            
            line = f"{arg.agent_name}: {arg.statement}"
            lines.append(line)
        
        return "\n".join(lines)
    
    def _format_proposition(self, proposition: Proposition) -> str:
        """Format proposition for display.
        
        Args:
            proposition: The proposition
            
        Returns:
            Formatted string
        """
        game = proposition.game_info
        desc = f"{game.away_team} @ {game.home_team} (Week {game.week}, {game.season})\n"
        desc += f"Bet Type: {proposition.bet_type.value}\n"
        
        if proposition.line:
            desc += f"Spread: {proposition.line.spread}, Total: {proposition.line.total}"
        
        return desc

