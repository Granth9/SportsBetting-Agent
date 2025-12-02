"""Common data types and structures."""

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

