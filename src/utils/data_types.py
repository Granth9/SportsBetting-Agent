"""Common data types and structures."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any


class BetType(Enum):
    """Types of betting propositions."""
    GAME_OUTCOME = "game_outcome"
    SPREAD = "spread"
    TOTAL = "total"
    PLAYER_PROP = "player_prop"


class Outcome(Enum):
    """Prediction outcomes."""
    HOME_WIN = "home_win"
    AWAY_WIN = "away_win"
    OVER = "over"
    UNDER = "under"
    COVER = "cover"
    NO_COVER = "no_cover"


@dataclass
class GameInfo:
    """Information about an NFL game."""
    game_id: str
    home_team: str
    away_team: str
    game_date: datetime
    season: int
    week: int
    home_score: Optional[int] = None
    away_score: Optional[int] = None
    weather: Optional[Dict[str, Any]] = None


@dataclass
class BettingLine:
    """Betting line information."""
    spread: float
    total: float
    home_ml: float
    away_ml: float
    source: str = "unknown"


@dataclass
class BettingOption:
    """A single betting option from a sportsbook."""
    option_id: str
    title: str  # e.g., "Patrick Mahomes Over 275.5 Passing Yards"
    description: str  # Full description of the bet
    bet_type: BetType
    odds: float  # American odds format (e.g., -110, +150)
    line_value: Optional[float] = None  # For spreads, totals, props
    player_name: Optional[str] = None  # For player props
    stat_type: Optional[str] = None  # e.g., "passing_yards", "touchdowns"
    source: str = "unknown"  # sleeper, underdog, espn
    popularity_rank: Optional[int] = None  # Rank by popularity
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_proposition(self, game_info: GameInfo) -> Proposition:
        """Convert this betting option to a Proposition for analysis.
        
        Args:
            game_info: Game information for this bet
            
        Returns:
            Proposition object
        """
        # Create betting line if applicable
        betting_line = None
        if self.bet_type in [BetType.SPREAD, BetType.TOTAL, BetType.GAME_OUTCOME]:
            # Extract line info from metadata or line_value
            spread = self.metadata.get('spread', 0.0) if self.bet_type == BetType.SPREAD else 0.0
            total = self.line_value if self.bet_type == BetType.TOTAL else 0.0
            
            # Convert odds to moneyline format
            home_ml = self.metadata.get('home_ml', -110)
            away_ml = self.metadata.get('away_ml', -110)
            
            betting_line = BettingLine(
                spread=spread,
                total=total,
                home_ml=home_ml,
                away_ml=away_ml,
                source=self.source
            )
        
        return Proposition(
            prop_id=self.option_id,
            game_info=game_info,
            bet_type=self.bet_type,
            line=betting_line,
            line_value=self.line_value,
            player_name=self.player_name,
            stat_type=self.stat_type,
            metadata={
                **self.metadata,
                'source': self.source,
                'original_title': self.title,
                'odds': self.odds
            }
        )


@dataclass
class Proposition:
    """A betting proposition to evaluate."""
    prop_id: str
    game_info: GameInfo
    bet_type: BetType
    line: Optional[BettingLine] = None
    player_name: Optional[str] = None
    stat_type: Optional[str] = None
    line_value: Optional[float] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ModelPrediction:
    """Prediction from a single model."""
    model_name: str
    prediction: Outcome
    confidence: float  # 0.0 to 1.0
    probability: Optional[float] = None  # For outcome probabilities
    key_features: Dict[str, Any] = None
    reasoning: str = ""
    
    def __post_init__(self):
        if self.key_features is None:
            self.key_features = {}


@dataclass
class AgentArgument:
    """An argument made by an agent during debate."""
    agent_name: str
    round_number: int
    statement: str
    confidence: float
    responding_to: Optional[str] = None


@dataclass
class DebateResult:
    """Result of the agent debate."""
    final_prediction: Outcome
    final_confidence: float
    consensus_level: float  # How much agents agreed (0.0 to 1.0)
    debate_transcript: List[AgentArgument]
    model_predictions: List[ModelPrediction]
    reasoning_summary: str
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class Recommendation:
    """Final betting recommendation."""
    proposition: Proposition
    debate_result: DebateResult
    recommended_action: str  # "BET", "PASS", "STRONG_BET"
    bet_size: Optional[float] = None
    expected_value: Optional[float] = None
    risk_assessment: str = ""

