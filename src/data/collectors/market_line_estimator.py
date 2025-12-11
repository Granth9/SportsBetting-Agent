"""Market-consistent betting line estimator for player props."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class MarketLineEstimator:
    """Estimate market-consistent betting lines for player props.
    
    Uses statistical methods to generate realistic betting lines similar to what
    sportsbooks would set, without requiring paid API access.
    """
    
    def __init__(self):
        """Initialize the market line estimator."""
        self.nfl_collector = None
        self._defensive_stats_cache = {}
    
    def _get_nfl_collector(self):
        """Lazy load NFL data collector."""
        if self.nfl_collector is None:
            from src.data.collectors.nfl_data_collector import NFLDataCollector
            self.nfl_collector = NFLDataCollector()
        return self.nfl_collector
    
    def estimate_player_line(
        self,
        player_name: str,
        stat_type: str,
        opponent_team: str,
        season: int = 2025,
        player_stats: Optional[Dict[str, Any]] = None
    ) -> float:
        """Estimate a market-consistent betting line for a player prop.
        
        Args:
            player_name: Player name
            stat_type: Type of stat ('rushing_yards', 'receiving_yards', 'passing_yards', 'total_yards')
            opponent_team: Opponent team abbreviation
            season: Season year
            player_stats: Optional pre-fetched player stats dict
            
        Returns:
            Estimated betting line (e.g., 65.5 for rushing yards)
        """
        # Get base line from player's historical performance
        base_line = self._get_base_line(player_name, stat_type, season, player_stats)
        
        if base_line <= 0:
            logger.warning(f"Invalid base line for {player_name}, using default")
            return self._get_default_line(stat_type)
        
        # Adjust for opponent matchup
        adjusted_line = self._adjust_for_opponent(base_line, stat_type, opponent_team, season)
        
        # Apply market efficiency rules (rounding, standard increments)
        final_line = self._apply_market_efficiency(adjusted_line, stat_type, player_name, player_stats)
        
        return final_line
    
    def _get_base_line(
        self,
        player_name: str,
        stat_type: str,
        season: int,
        player_stats: Optional[Dict[str, Any]] = None
    ) -> float:
        """Calculate base line from player's historical average.
        
        Args:
            player_name: Player name
            stat_type: Type of stat
            season: Season year
            player_stats: Optional pre-fetched stats
            
        Returns:
            Base line value
        """
        if player_stats is None:
            # Get player stats from quick predictor
            try:
                from src.pipeline.quick_predict import PlayerPropPredictor
                pp = PlayerPropPredictor()
                player_stats = pp.get_player_history(player_name, stat_type)
            except Exception as e:
                logger.debug(f"Error getting player stats: {e}")
                return self._get_default_line(stat_type)
        
        if not player_stats.get('found'):
            return self._get_default_line(stat_type)
        
        # Use average as base (more representative of typical performance)
        # Median can be skewed by outlier games
        avg = player_stats.get('avg_yards', 0)
        median = player_stats.get('median_yards', 0)
        
        # Use average as primary, but consider median for validation
        base = avg if avg > 0 else median
        
        # If we have game-level data, use recent form (last 5 games)
        last_5_games = player_stats.get('last_5_games', [])
        if last_5_games and len(last_5_games) >= 3:
            # Convert to numeric, filter out invalid values
            recent_values = [float(x) for x in last_5_games if isinstance(x, (int, float)) and not pd.isna(x)]
            if len(recent_values) >= 3:
                # Use trimmed mean (remove highest and lowest) for more stable estimate
                # This reduces impact of outlier games
                sorted_values = sorted(recent_values)
                if len(sorted_values) >= 5:
                    # Remove highest and lowest
                    trimmed = sorted_values[1:-1]
                    recent_avg = np.mean(trimmed)
                else:
                    recent_avg = np.mean(recent_values)
                
                # Weight recent form moderately (55% recent, 45% season average)
                # This balances current form with season-long consistency
                # Slightly favor season average to be more conservative (closer to market lines)
                if recent_avg > 0:
                    base = (recent_avg * 0.55) + (base * 0.45)
        
        # Don't add premium - market lines are typically set conservatively
        # Sportsbooks set lines to balance action, not to reflect peak performance
        
        return max(0, base)
    
    def _adjust_for_opponent(
        self,
        base_line: float,
        stat_type: str,
        opponent_team: str,
        season: int
    ) -> float:
        """Adjust line based on opponent's defensive performance.
        
        Args:
            base_line: Base line value
            stat_type: Type of stat
            opponent_team: Opponent team abbreviation
            season: Season year
            
        Returns:
            Adjusted line value
        """
        try:
            nfl = self._get_nfl_collector()
            
            # Try to use current season data first, fall back to previous year
            seasons_to_try = [season, season - 1]
            
            # Get weekly stats to calculate opponent defensive rankings
            # Try current season first, then previous year
            weekly_stats = pd.DataFrame()
            schedule = pd.DataFrame()
            data_season = None
            
            try:
                import nfl_data_py as nfl_data
                for try_season in seasons_to_try:
                    try:
                        weekly_stats = nfl_data.import_weekly_data([try_season])
                        schedule = nfl_data.import_schedules([try_season])
                        if len(weekly_stats) > 0 and len(schedule) > 0:
                            data_season = try_season
                            break
                    except Exception as e:
                        logger.debug(f"Error loading {try_season} data: {e}")
                        continue
            except Exception as e:
                logger.debug(f"Error loading weekly stats: {e}")
                return base_line
            
            if len(weekly_stats) == 0 or len(schedule) == 0:
                return base_line
            
            # Find games where opponent_team was the defense
            # Handle both 'LA' and 'LAR' for Rams
            opponent_variants = [opponent_team]
            if opponent_team == 'LAR':
                opponent_variants.append('LA')
            elif opponent_team == 'LA':
                opponent_variants.append('LAR')
            
            opponent_games = pd.DataFrame()
            for opp_variant in opponent_variants:
                games = schedule[
                    ((schedule['home_team'] == opp_variant) | (schedule['away_team'] == opp_variant)) &
                    pd.notna(schedule.get('home_score'))
                ]
                if len(games) > 0:
                    opponent_games = games
                    break
            
            if len(opponent_games) == 0:
                return base_line
            
            # Calculate opponent's average yards allowed for this stat type
            total_allowed = 0
            games_counted = 0
            
            for _, game in opponent_games.iterrows():
                # Determine which team scored against opponent
                if game['home_team'] == opponent_team:
                    scoring_team = game['away_team']
                else:
                    scoring_team = game['home_team']
                
                # Get stats for that team in this game
                game_stats = weekly_stats[
                    (weekly_stats['season'] == game['season']) &
                    (weekly_stats['week'] == game['week']) &
                    (weekly_stats['recent_team'] == scoring_team)
                ]
                
                if len(game_stats) == 0:
                    continue
                
                # Sum up the relevant stat type
                if stat_type == 'rushing_yards':
                    yards_allowed = game_stats['rushing_yards'].fillna(0).sum()
                elif stat_type == 'receiving_yards':
                    yards_allowed = game_stats['receiving_yards'].fillna(0).sum()
                elif stat_type == 'passing_yards':
                    yards_allowed = game_stats['passing_yards'].fillna(0).sum()
                elif stat_type == 'total_yards':
                    yards_allowed = (game_stats['rushing_yards'].fillna(0) + 
                                   game_stats['receiving_yards'].fillna(0)).sum()
                else:
                    yards_allowed = 0
                
                total_allowed += yards_allowed
                games_counted += 1
            
            if games_counted == 0:
                return base_line
            
            avg_allowed = total_allowed / games_counted
            
            # Calculate league average for comparison
            try:
                all_teams_stats = weekly_stats.groupby(['season', 'week', 'recent_team']).agg({
                    'rushing_yards': 'sum',
                    'receiving_yards': 'sum',
                    'passing_yards': 'sum'
                }).reset_index()
            except Exception as e:
                logger.debug(f"Error calculating league averages: {e}")
                return base_line
            
            if stat_type == 'rushing_yards':
                league_avg = all_teams_stats['rushing_yards'].mean()
            elif stat_type == 'receiving_yards':
                league_avg = all_teams_stats['receiving_yards'].mean()
            elif stat_type == 'passing_yards':
                league_avg = all_teams_stats['passing_yards'].mean()
            elif stat_type == 'total_yards':
                league_avg = (all_teams_stats['rushing_yards'] + all_teams_stats['receiving_yards']).mean()
            else:
                league_avg = avg_allowed
            
            # Adjust base line based on opponent strength
            # If opponent allows more than league average, increase line
            # If opponent allows less, decrease line
            if league_avg > 0:
                adjustment_factor = avg_allowed / league_avg
                # For current season data, use slightly wider range (0.75x to 1.25x)
                # For previous year data, use tighter range (0.8x to 1.2x)
                if data_season == season:
                    adjustment_factor = max(0.75, min(1.25, adjustment_factor))
                else:
                    adjustment_factor = max(0.8, min(1.2, adjustment_factor))
                adjusted_line = base_line * adjustment_factor
            else:
                adjusted_line = base_line
            
            return adjusted_line
            
        except Exception as e:
            logger.debug(f"Error adjusting for opponent: {e}")
            return base_line
    
    def _apply_market_efficiency(
        self,
        line: float,
        stat_type: str,
        player_name: str,
        player_stats: Optional[Dict[str, Any]] = None
    ) -> float:
        """Apply market efficiency rules to round line to standard values.
        
        Args:
            line: Raw line value
            stat_type: Type of stat
            player_stats: Optional player stats for volatility calculation
            
        Returns:
            Market-standard rounded line
        """
        # Get standard increment based on stat type
        if stat_type == 'passing_yards':
            increment = 10  # QBs: 10-yard increments (250, 260, 270, etc.)
        else:
            increment = 5  # RBs/WRs: 5-yard increments (55, 60, 65, etc.)
        
        # Round to nearest increment
        rounded = round(line / increment) * increment
        
        # For yards props, add 0.5 (standard sportsbook format)
        # e.g., 65 becomes 65.5, 250 becomes 250.5
        base_line = rounded + 0.5
        
        # Apply minimal "vig" adjustment - sportsbooks typically set lines very close to true value
        # Adjust by 1-2% to account for house edge (sportsbooks set lines slightly lower for overs)
        vig_adjustment = 0.015  # 1.5% adjustment (sportsbooks typically set lines 1-2% below true value)
        adjusted = base_line * (1 - vig_adjustment)
        
        # Round to nearest 0.5
        rounded_to_half = round(adjusted * 2) / 2
        
        # Ensure it ends in .5 (standard sportsbook format)
        if rounded_to_half % 1 == 0:
            # If it's a whole number, round to nearest .5
            # Round up if closer to next .5, down if closer to previous .5
            if adjusted - rounded_to_half >= 0.25:
                final_line = rounded_to_half + 0.5
            else:
                final_line = rounded_to_half - 0.5
        else:
            final_line = rounded_to_half
        
        # Ensure minimum values
        min_lines = {
            'rushing_yards': 15.5,
            'receiving_yards': 10.5,
            'passing_yards': 150.5,
            'total_yards': 20.5
        }
        min_line = min_lines.get(stat_type, 10.5)
        final_line = max(min_line, final_line)
        
        return final_line
    
    def _get_default_line(self, stat_type: str) -> float:
        """Get default line if player stats unavailable.
        
        Args:
            stat_type: Type of stat
            
        Returns:
            Default line value
        """
        defaults = {
            'rushing_yards': 60.5,
            'receiving_yards': 55.5,
            'passing_yards': 250.5,
            'total_yards': 70.5
        }
        return defaults.get(stat_type, 50.5)

