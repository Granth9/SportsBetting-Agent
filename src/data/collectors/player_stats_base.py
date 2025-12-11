"""Base abstract class for player stats collectors."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class PlayerStatsCollector(ABC):
    """Base class for all player stats collectors.
    
    All player stats collectors must implement these methods to ensure
    compatibility with the prediction pipeline.
    """
    
    @abstractmethod
    def get_player_weekly_stats(self, player_name: str, stat_type: str = 'receiving_yards') -> Dict[str, Any]:
        """Get weekly stats for a player in the current season.
        
        Args:
            player_name: Player name (e.g., "Puka Nacua")
            stat_type: Type of stat ('receiving_yards', 'rushing_yards', 'passing_yards', 
                       'receiving_td', 'rushing_td', 'passing_td', 'anytime_td', 'total_yards')
            
        Returns:
            Dictionary with player stats in standardized format:
            {
                'success': bool,
                'player_name': str,
                'player_id': str,
                'season': int,
                'stat_type': str,
                'games_played': int,
                'avg_yards': float (or avg_tds_per_game for TD stats),
                'median_yards': float,
                'std_yards': float,
                'max_yards': float,
                'min_yards': float,
                'game_yards': List[float] (or game_tds for TD stats),
                'last_5_games': List[str] (formatted as "W14: 167"),
                'source': str,
                'is_td_stat': bool (optional),
                'td_rate': float (for TD stats),
                'games_with_td': int (for TD stats),
                'total_tds': int (for TD stats),
                'error': str (if success=False)
            }
        """
        pass
    
    @abstractmethod
    def _find_player_id(self, player_name: str) -> Optional[str]:
        """Find player ID from name.
        
        Args:
            player_name: Player name (e.g., "Puka Nacua")
            
        Returns:
            Player ID string or None if not found
        """
        pass
    
    @abstractmethod
    def _get_current_season(self) -> int:
        """Get current NFL season year.
        
        Returns:
            Current season year (e.g., 2025)
        """
        pass

