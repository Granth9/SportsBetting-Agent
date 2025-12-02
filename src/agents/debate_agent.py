"""LLM agents for model debate."""

import anthropic
from typing import Dict, Any, Optional
from datetime import datetime

from src.utils.data_types import ModelPrediction, AgentArgument, Proposition
from src.utils.config_loader import get_config
from src.utils.logger import setup_logger
from src.agents.prompts.system_prompts import (
    get_agent_prompt,
    INITIAL_STATEMENT_TEMPLATE,
    REBUTTAL_TEMPLATE,
    FINAL_STATEMENT_TEMPLATE
)


logger = setup_logger(__name__)


class DebateAgent:
    """LLM agent representing a machine learning model in debate."""
    
    def __init__(
        self,
        agent_type: str,
        model_prediction: ModelPrediction,
        historical_accuracy: float = 0.5,
        temperature: float = 0.7
    ):
        """Initialize a debate agent.
        
        Args:
            agent_type: Type of agent (neural_analyst, gradient_strategist, etc.)
            model_prediction: The model's prediction for this proposition
            historical_accuracy: Historical accuracy of the model
            temperature: LLM temperature for generation
        """
        self.agent_type = agent_type
        self.model_prediction = model_prediction
        self.historical_accuracy = historical_accuracy
        self.temperature = temperature
        
        # Get configuration
        config = get_config()
        self.api_key = config.anthropic_api_key
        self.model_name = config.get('anthropic.model', 'claude-3-5-sonnet-20241022')
        
        # Initialize Anthropic client
        self.client = anthropic.Anthropic(api_key=self.api_key)
        
        # Get system prompt
        self.system_prompt = get_agent_prompt(agent_type, historical_accuracy)
        
        # Track confidence over debate
        self.current_confidence = model_prediction.confidence
    
    def initial_statement(self, proposition: Proposition) -> AgentArgument:
        """Generate initial statement for the proposition.
        
        Args:
            proposition: The betting proposition
            
        Returns:
            AgentArgument with the statement
        """
        logger.debug(f"{self.agent_type} generating initial statement")
        
        # Format proposition description
        prop_desc = self._format_proposition(proposition)
        
        # Format key features
        key_features_str = ", ".join([f"{k}: {v}" for k, v in self.model_prediction.key_features.items()])
        
        # Create prompt
        user_prompt = INITIAL_STATEMENT_TEMPLATE.format(
            proposition_description=prop_desc,
            prediction=self.model_prediction.prediction.value,
            confidence=self.model_prediction.confidence,
            probability=self.model_prediction.probability or 0.5,
            key_features=key_features_str
        )
        
        # Generate response
        statement = self._generate_response(user_prompt, max_tokens=500)
        
        return AgentArgument(
            agent_name=self.model_prediction.model_name,
            round_number=1,
            statement=statement,
            confidence=self.current_confidence
        )
    
    def rebuttal(
        self,
        proposition: Proposition,
        other_arguments: list[AgentArgument],
        round_number: int
    ) -> AgentArgument:
        """Generate a rebuttal to other agents' arguments.
        
        Args:
            proposition: The betting proposition
            other_arguments: Arguments from other agents
            round_number: Current debate round
            
        Returns:
            AgentArgument with the rebuttal
        """
        logger.debug(f"{self.agent_type} generating rebuttal for round {round_number}")
        
        # Format proposition
        prop_desc = self._format_proposition(proposition)
        
        # Format other arguments
        other_args_str = "\n\n".join([
            f"{arg.agent_name} ({arg.confidence:.1%} confident): {arg.statement}"
            for arg in other_arguments
            if arg.agent_name != self.model_prediction.model_name
        ])
        
        # Create prompt
        user_prompt = REBUTTAL_TEMPLATE.format(
            proposition_description=prop_desc,
            prediction=self.model_prediction.prediction.value,
            confidence=self.current_confidence,
            other_arguments=other_args_str
        )
        
        # Generate response
        statement = self._generate_response(user_prompt, max_tokens=400)
        
        # Extract any confidence adjustment from the statement
        self._update_confidence_from_statement(statement)
        
        return AgentArgument(
            agent_name=self.model_prediction.model_name,
            round_number=round_number,
            statement=statement,
            confidence=self.current_confidence
        )
    
    def final_statement(
        self,
        proposition: Proposition,
        debate_history: list[AgentArgument],
        round_number: int
    ) -> AgentArgument:
        """Generate final statement after debate.
        
        Args:
            proposition: The betting proposition
            debate_history: All previous arguments
            round_number: Final round number
            
        Returns:
            AgentArgument with final statement
        """
        logger.debug(f"{self.agent_type} generating final statement")
        
        # Format proposition
        prop_desc = self._format_proposition(proposition)
        
        # Summarize debate
        debate_summary = self._summarize_debate(debate_history)
        
        # Create prompt
        user_prompt = FINAL_STATEMENT_TEMPLATE.format(
            proposition_description=prop_desc,
            prediction=self.model_prediction.prediction.value,
            confidence=self.model_prediction.confidence,
            debate_summary=debate_summary
        )
        
        # Generate response
        statement = self._generate_response(user_prompt, max_tokens=300)
        
        # Extract final confidence
        self._update_confidence_from_statement(statement)
        
        return AgentArgument(
            agent_name=self.model_prediction.model_name,
            round_number=round_number,
            statement=statement,
            confidence=self.current_confidence
        )
    
    def _generate_response(self, user_prompt: str, max_tokens: int = 500) -> str:
        """Generate a response using Claude.
        
        Args:
            user_prompt: User prompt
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text
        """
        try:
            message = self.client.messages.create(
                model=self.model_name,
                max_tokens=max_tokens,
                temperature=self.temperature,
                system=self.system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            return message.content[0].text
        
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            # Fallback to a simple statement
            return f"I predict {self.model_prediction.prediction.value} based on my model's analysis."
    
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
            desc += f"Spread: {proposition.line.spread}, Total: {proposition.line.total}\n"
        
        return desc
    
    def _summarize_debate(self, debate_history: list[AgentArgument]) -> str:
        """Summarize debate history.
        
        Args:
            debate_history: All arguments
            
        Returns:
            Summary string
        """
        summary = []
        
        # Group by agent
        by_agent: Dict[str, list[AgentArgument]] = {}
        for arg in debate_history:
            if arg.agent_name not in by_agent:
                by_agent[arg.agent_name] = []
            by_agent[arg.agent_name].append(arg)
        
        # Summarize each agent's position
        for agent_name, args in by_agent.items():
            latest = args[-1]
            summary.append(f"{agent_name}: {latest.statement[:100]}... ({latest.confidence:.1%} confident)")
        
        return "\n".join(summary)
    
    def _update_confidence_from_statement(self, statement: str) -> None:
        """Extract and update confidence from statement text.
        
        Args:
            statement: The generated statement
        """
        # Look for confidence indicators in the text
        import re
        
        # Look for explicit percentage
        match = re.search(r'(\d+)%', statement)
        if match:
            new_confidence = int(match.group(1)) / 100
            # Only update if it's a reasonable value
            if 0.3 <= new_confidence <= 0.99:
                self.current_confidence = new_confidence
        
        # Look for confidence keywords
        if "very confident" in statement.lower() or "highly confident" in statement.lower():
            self.current_confidence = min(0.95, self.current_confidence + 0.05)
        elif "less confident" in statement.lower() or "uncertain" in statement.lower():
            self.current_confidence = max(0.5, self.current_confidence - 0.1)

