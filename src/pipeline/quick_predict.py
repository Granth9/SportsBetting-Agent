"""Quick prediction module for natural language game queries and player props."""

import re
import ssl
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import pandas as pd

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# SSL fix for nfl_data_py (required for nfl_data_py to work)
ssl._create_default_https_context = ssl._create_unverified_context


class PlayerPropPredictor:
    """Predictor for player prop bets (yards, touchdowns, etc.).
    
    Uses MultiSourceStatsCollector for 2025 season data (API-SPORTS primary, Sleeper fallback),
    with fallback to nfl_data_py for historical data.
    """
    
    def __init__(self):
        """Initialize the player prop predictor."""
        self.player_stats = None
        self.loaded_years = []
        self.stats_collector = None
        self._use_multi_source = True  # Try multi-source collector first for 2025 data
        
        # Import team key players and injured players from parlay builder
        try:
            from src.pipeline.parlay_builder import ParlayBuilder
            self.TEAM_KEY_PLAYERS = ParlayBuilder.TEAM_KEY_PLAYERS
            self.INJURED_PLAYERS = ParlayBuilder.INJURED_PLAYERS
        except:
            # Fallback if import fails
            self.TEAM_KEY_PLAYERS = {}
            self.INJURED_PLAYERS = {}
    
    def _get_stats_collector(self):
        """Lazy load multi-source stats collector."""
        if self.stats_collector is None:
            try:
                from src.data.collectors.multi_source_stats import MultiSourceStatsCollector
                self.stats_collector = MultiSourceStatsCollector()
                logger.info("Multi-source player stats collector initialized (API-SPORTS primary, Sleeper fallback)")
            except ImportError as e:
                logger.warning(f"Could not import multi-source collector: {e}")
                self._use_multi_source = False
        return self.stats_collector
    
    def load_player_stats(self, years: List[int] = None) -> pd.DataFrame:
        """Load player weekly stats.
        
        Args:
            years: Years to load (defaults to recent available years)
            
        Returns:
            DataFrame with player stats
        """
        # Only load if not already loaded
        if self.player_stats is not None and len(self.player_stats) > 0:
            return self.player_stats
        
        import nfl_data_py as nfl
        
        # Try different year combinations
        if years is None:
            current_year = datetime.now().year
            year_options = [
                [current_year - 1],  # Just last year
                [current_year - 2, current_year - 1],  # Last 2 years
                [current_year - 1, current_year],  # Try with current
            ]
        else:
            year_options = [years]
        
        for years_to_try in year_options:
            try:
                logger.info(f"Loading player stats for {years_to_try}...")
                self.player_stats = nfl.import_weekly_data(years_to_try)
                self.loaded_years = years_to_try
                logger.info(f"Loaded {len(self.player_stats)} player-game records")
                return self.player_stats
            except Exception as e:
                logger.debug(f"Could not load {years_to_try}: {e}")
                continue
        
        logger.warning("Could not load any player stats")
        self.player_stats = pd.DataFrame()
        return self.player_stats
    
    def find_player(self, query: str) -> Optional[str]:
        """Find player name from query.
        
        Args:
            query: Natural language query
            
        Returns:
            Player display name or None
        """
        query_lower = query.lower()
        
        # Common player name patterns to extract
        name_patterns = [
            # "R.J. Harvey" with periods (initials) - match first
            r"([A-Z]\.?\s*[A-Z]\.?\s+[A-Z][a-zA-Z]+)",
            # "De'Von Achane" or "Ja'Marr Chase" with apostrophe in first name
            r"([A-Z][a-zA-Z]*['\'][A-Z][a-zA-Z]+\s+[A-Z][a-zA-Z]+)",
            # "Ja'Marr Chase" with apostrophe (more flexible)
            r"([A-Za-z]+['\'][A-Za-z]+\s+[A-Za-z]+)",
            # Names like "Ladd McConkey" with mixed case (Mc, Mac, O', etc.)
            r"([A-Z][a-z]+\s+(?:Mc|Mac|O\')?[A-Z][a-zA-Z]+)",
            # "will Devonta Smith get" -> "Devonta Smith"  
            r"(?:will|can|does)\s+([A-Z][a-z]+(?:\'[A-Z][a-z]+)?\s+[A-Z][a-zA-Z]+)\s+(?:get|have|hit)",
            # "Devonta Smith 40 yards" -> "Devonta Smith"
            r"([A-Z][a-z]+(?:\'[A-Z][a-z]+)?\s+[A-Z][a-zA-Z]+)\s+\d+",
            # Two capitalized words (name pattern) - allow mixed case in last name
            r"([A-Z][a-z]+(?:\'[A-Z][a-z]+)?\s+[A-Z][a-zA-Z]+)",
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, query)
            if match:
                potential_name = match.group(1).strip()
                # Filter out common non-name words
                skip_words = ['the', 'and', 'over', 'under', 'will', 'get', 'have', 'can', 'does']
                words = potential_name.split()
                if len(words) >= 2 and words[0].lower() not in skip_words and words[1].lower() not in skip_words:
                    return potential_name
        
        # Fallback: Try to match known players from stats collector
        # Common player names to match (top players)
        known_players = [
            "Ja'Marr Chase", "Devonta Smith", "CeeDee Lamb", "Tyreek Hill",
            "Justin Jefferson", "Amon-Ra St. Brown", "A.J. Brown", "Davante Adams",
            "Saquon Barkley", "Derrick Henry", "Josh Jacobs", "Jonathan Taylor",
            "Patrick Mahomes", "Josh Allen", "Lamar Jackson", "Jalen Hurts",
            "Travis Kelce", "George Kittle", "Mark Andrews", "T.J. Hockenson",
            "Ladd McConkey", "Malik Nabers", "Brian Thomas Jr", "Puka Nacua",
            "Brock Bowers", "Bijan Robinson", "Jahmyr Gibbs", "De'Von Achane",
            "Omarion Hampton", "Marvin Harrison Jr", "Caleb Williams", "Jayden Daniels",
            "R.J. Harvey", "RJ Harvey",
        ]
        
        for player in known_players:
            # Check if player name (with or without apostrophe) is in query
            player_normalized = player.lower().replace("'", "").replace("-", " ")
            query_normalized = query_lower.replace("'", "").replace("-", " ")
            
            # Check if full name matches (all parts)
            player_parts = player_normalized.split()
            if len(player_parts) >= 2:
                # Check if all significant parts (len > 3) are in query
                significant_parts = [p for p in player_parts if len(p) > 3]
                if significant_parts:
                    matches = sum(1 for part in significant_parts if part in query_normalized)
                    if matches >= len(significant_parts):
                        return player
                # Also check if last name is in query (for cases like "Achane")
                if len(player_parts) >= 2:
                    last_name = player_parts[-1]
                    if len(last_name) > 3 and last_name in query_normalized:
                        # Verify with first name or part of first name
                        first_name = player_parts[0]
                        if first_name[:3] in query_normalized or first_name in query_normalized:
                            return player
        
        # Fallback: Load stats and search for matches
        if self.player_stats is None or len(self.player_stats) == 0:
            self.load_player_stats()
        
        if self.player_stats is None or len(self.player_stats) == 0:
            return None
        
        # Get unique player names
        players = self.player_stats['player_display_name'].dropna().unique()
        
        # Try exact match first
        for player in players:
            if player.lower() in query_lower:
                return player
        
        # Try partial match (last name)
        for player in players:
            parts = player.split()
            if len(parts) >= 2:
                last_name = parts[-1].lower()
                first_name = parts[0].lower()
                # Match "Smith" or "Devonta" etc.
                if last_name in query_lower and len(last_name) > 3:
                    # Verify with first name or initial
                    if first_name in query_lower or first_name[0] in query_lower:
                        return player
        
        return None
    
    def extract_yards_line(self, query: str) -> Optional[float]:
        """Extract yards line from query.
        
        Args:
            query: Natural language query like "will he get 40 yards"
            
        Returns:
            Yards line or None
        """
        # Pattern: "40 yards", "40 yds", "40yrds", "over 40", "under 40", "100 total yards"
        patterns = [
            r'(\d+(?:\.\d+)?)\s*(?:total\s+)?(?:yards?|yds?|yrds?)',
            r'(?:over|under|get|hit)\s*(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)\s*(?:receiving|rushing|passing|total|scrimmage)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query.lower())
            if match:
                return float(match.group(1))
        
        return None
    
    def extract_stat_type(self, query: str) -> str:
        """Extract stat type from query.
        
        Args:
            query: Natural language query
            
        Returns:
            Stat type: yards ('receiving_yards', etc.) or TDs ('anytime_td', etc.)
        """
        query_lower = query.lower()
        
        # Check for touchdown queries first
        if any(word in query_lower for word in ['touchdown', ' td', 'score a td', 'get a td']):
            # Check for specific TD type
            if any(word in query_lower for word in ['rushing td', 'rush td', 'rushing touchdown']):
                return 'rushing_td'
            elif any(word in query_lower for word in ['receiving td', 'rec td', 'receiving touchdown']):
                return 'receiving_td'
            elif any(word in query_lower for word in ['passing td', 'pass td', 'passing touchdown']):
                return 'passing_td'
            else:
                # Default to anytime TD (rushing or receiving)
                return 'anytime_td'
        
        # Check for total/combined yards (receiving + rushing)
        if any(word in query_lower for word in ['total', 'scrimmage', 'combined', 'all purpose', 'all-purpose']):
            return 'total_yards'
        elif any(word in query_lower for word in ['rushing', 'rush', 'run', 'carry', 'carries']):
            return 'rushing_yards'
        elif any(word in query_lower for word in ['passing', 'pass', 'throw', 'qb']):
            return 'passing_yards'
        elif any(word in query_lower for word in ['receiving', 'rec', 'catch', 'catches', 'reception']):
            return 'receiving_yards'
        else:
            # Default to receiving yards for WRs/TEs
            return 'receiving_yards'
    
    def is_td_query(self, query: str) -> bool:
        """Check if query is about touchdowns.
        
        Args:
            query: Natural language query
            
        Returns:
            True if this is a touchdown query
        """
        query_lower = query.lower()
        td_keywords = ['touchdown', ' td', 'score', 'anytime scorer', 'first td', 'last td']
        return any(kw in query_lower for kw in td_keywords)
    
    def get_player_history_multi_source(self, player_name: str, stat_type: str = 'receiving_yards') -> Dict[str, Any]:
        """Get player's 2025 season stats from multi-source collector.
        
        Args:
            player_name: Player display name
            stat_type: Type of stat to analyze
            
        Returns:
            Dictionary with player stats
        """
        collector = self._get_stats_collector()
        if not collector:
            return {'found': False, 'error': 'Stats collector not available'}
        
        try:
            stats = collector.get_player_weekly_stats(player_name, stat_type)
            
            if not stats['success']:
                return {'found': False, 'error': stats.get('error', 'Unknown error')}
            
            # Handle TD stats differently
            if stats.get('is_td_stat'):
                return {
                    'found': True,
                    'is_td_stat': True,
                    'player_name': stats['player_name'],
                    'position': 'WR/RB/QB',
                    'team': 'NFL',
                    'games_played': stats['games_played'],
                    'total_tds': stats['total_tds'],
                    'games_with_td': stats['games_with_td'],
                    'td_rate': stats['td_rate'],
                    'avg_tds_per_game': stats['avg_tds_per_game'],
                    'game_tds': stats['game_tds'],
                    'last_5_games': stats['last_5_games'],
                    'season': str(stats['season']),
                    'source': stats.get('source', 'Multi-Source (2025)'),
                }
            
            # Handle yards stats
            return {
                'found': True,
                'is_td_stat': False,
                'player_name': stats['player_name'],
                'position': 'WR/RB/QB',
                'team': 'NFL',
                'games_played': stats['games_played'],
                'avg_yards': stats['avg_yards'],
                'median_yards': stats['median_yards'],
                'max_yards': stats.get('max_yards', max(stats['game_yards']) if stats['game_yards'] else 0),
                'min_yards': stats.get('min_yards', min(stats['game_yards']) if stats['game_yards'] else 0),
                'std_yards': stats['std_yards'],
                'last_5_games': stats['last_5_games'],
                'game_yards': stats['game_yards'],  # All game yards for hit rate calc
                'season': str(stats['season']),
                'source': stats.get('source', 'Multi-Source (2025)'),
            }
        except Exception as e:
            logger.warning(f"Stats lookup failed for {player_name}: {e}")
            return {'found': False, 'error': str(e)}
    
    def get_player_history(self, player_name: str, stat_type: str = 'receiving_yards', 
                          n_games: int = 10) -> Dict[str, Any]:
        """Get player's recent performance history.
        
        Args:
            player_name: Player display name
            stat_type: Type of stat to analyze
            n_games: Number of recent games to consider
            
        Returns:
            Dictionary with player stats
        """
        # Try multi-source collector first for 2025 data
        if self._use_multi_source:
            multi_source_result = self.get_player_history_multi_source(player_name, stat_type)
            if multi_source_result.get('found'):
                source = multi_source_result.get('source', 'Multi-Source')
                logger.info(f"Found 2025 stats for {player_name} via {source}")
                return multi_source_result
        
        # Fallback to nfl_data_py for historical data
        if self.player_stats is None:
            self.load_player_stats()
        
        if self.player_stats is None or len(self.player_stats) == 0:
            return {'found': False, 'error': 'No player data available'}
        
        player_data = self.player_stats[
            self.player_stats['player_display_name'] == player_name
        ].sort_values(['season', 'week'], ascending=False).head(n_games)
        
        if len(player_data) == 0:
            return {'found': False, 'error': f'No data found for {player_name}'}
        
        yards = player_data[stat_type].dropna()
        
        if len(yards) == 0:
            return {'found': False, 'error': f'No {stat_type} data for {player_name}'}
        
        return {
            'found': True,
            'player_name': player_name,
            'position': player_data['position'].iloc[0] if 'position' in player_data.columns else 'Unknown',
            'team': player_data['recent_team'].iloc[0] if 'recent_team' in player_data.columns else 'Unknown',
            'games_played': len(yards),
            'avg_yards': yards.mean(),
            'median_yards': yards.median(),
            'max_yards': yards.max(),
            'min_yards': yards.min(),
            'std_yards': yards.std(),
            'last_5_games': yards.head(5).tolist(),
            'game_yards': yards.tolist(),  # For hit rate calculation
            'season': str(self.loaded_years[-1]) if self.loaded_years else '2024',
            'source': 'nfl_data_py',
        }
    
    def predict_over_under(self, player_name: str, yards_line: float, 
                           stat_type: str = 'receiving_yards') -> Dict[str, Any]:
        """Predict if player will go over/under yards line, or score a TD.
        
        Args:
            player_name: Player display name
            yards_line: Yards line to predict against (ignored for TD props)
            stat_type: Type of stat
            
        Returns:
            Prediction dictionary
        """
        history = self.get_player_history(player_name, stat_type)
        
        if not history.get('found'):
            return {
                'success': False,
                'error': history.get('error', f"No stats found for {player_name}")
            }
        
        # Handle TD predictions
        if history.get('is_td_stat'):
            return self._predict_td(player_name, stat_type, history)
        
        # Calculate hit rate for this line using game_yards from history
        game_yards = history.get('game_yards', [])
        if game_yards:
            times_over = sum(1 for y in game_yards if y >= yards_line)
            total_games = len(game_yards)
            hit_rate = times_over / total_games if total_games > 0 else 0.5
        else:
            hit_rate = 0.5
            times_over = 0
            total_games = 0
        
        # Calculate weighted average with exponential decay favoring most recent games
        # Most recent game gets highest weight, older games get progressively less weight
        season_avg = history['avg_yards']
        last_5_games = history.get('last_5_games', [])
        
        # Convert last_5_games to numeric values
        recent_values = []
        for val in last_5_games:
            if isinstance(val, str):
                # Handle "W14: 121" format
                try:
                    num_val = float(val.split(':')[1].strip())
                    recent_values.append(num_val)
                except:
                    try:
                        num_val = float(val)
                        recent_values.append(num_val)
                    except:
                        continue
            elif isinstance(val, (int, float)) and not pd.isna(val):
                recent_values.append(float(val))
        
        if len(recent_values) >= 3:
            # Use exponential decay weighting: most recent game = highest weight
            # Weights: [0.35, 0.25, 0.20, 0.12, 0.08] for last 5 games
            n_recent = len(recent_values)
            weights = []
            total_weight = 0
            
            # Generate exponential decay weights (most recent = highest)
            for i in range(n_recent):
                # Exponential decay: weight decreases by ~30% per game going back
                weight = 0.4 * (0.7 ** i)  # Most recent gets 0.4, then 0.28, 0.196, etc.
                weights.append(weight)
                total_weight += weight
            
            # Normalize weights to sum to 1
            weights = [w / total_weight for w in weights]
            
            # Calculate weighted average of recent games
            recent_avg = sum(val * weight for val, weight in zip(recent_values, weights))
            
            # Combine with season average: 75% recent (exponentially weighted), 25% season
            # This gives even more weight to recent form than before
            weighted_avg = (recent_avg * 0.75) + (season_avg * 0.25)
            
            # Calculate recent trend (improving vs declining) using weighted recent games
            if len(recent_values) >= 4:
                # Compare most recent 2 games vs previous 2 games
                most_recent = np.mean(recent_values[:2]) if len(recent_values) >= 2 else recent_values[0]
                previous = np.mean(recent_values[2:4]) if len(recent_values) >= 4 else np.mean(recent_values[2:])
                trend_factor = (most_recent - previous) / max(season_avg, 1)  # Normalize by season avg
            else:
                trend_factor = 0
        else:
            # Not enough recent data, use season average
            recent_avg = season_avg
            weighted_avg = season_avg
            trend_factor = 0
        
        # Use weighted average for prediction
        avg = weighted_avg
        std = history.get('std_yards', 10)
        if std == 0 or pd.isna(std):
            std = 10
        
        # Adjust std based on recent volatility if we have recent data
        # Use exponential weighting for recent volatility too
        if len(recent_values) >= 3:
            # Calculate weighted standard deviation using same exponential weights
            n_recent = len(recent_values)
            weights = []
            total_weight = 0
            for i in range(n_recent):
                weight = 0.4 * (0.7 ** i)
                weights.append(weight)
                total_weight += weight
            weights = [w / total_weight for w in weights]
            
            # Weighted mean for std calculation
            weighted_mean = sum(val * weight for val, weight in zip(recent_values, weights))
            # Weighted variance
            weighted_variance = sum(weight * (val - weighted_mean) ** 2 for val, weight in zip(recent_values, weights))
            recent_std = np.sqrt(weighted_variance) if weighted_variance > 0 else np.std(recent_values)
            
            # Blend recent std with season std (70% recent, 30% season) - more weight to recent
            std = (recent_std * 0.7) + (std * 0.3)
        
        # Calculate z-score using weighted average
        z_score = (yards_line - avg) / std
        
        # Adjust confidence based on recent trend
        # If player is improving (positive trend), boost confidence for OVER
        # If player is declining (negative trend), reduce confidence for OVER
        trend_adjustment = 0.0
        if trend_factor > 0.1:  # Improving significantly
            if z_score < 0:  # OVER prediction
                trend_adjustment = min(0.1, trend_factor * 0.5)
            else:  # UNDER prediction
                trend_adjustment = -min(0.1, trend_factor * 0.5)
        elif trend_factor < -0.1:  # Declining significantly
            if z_score < 0:  # OVER prediction
                trend_adjustment = -min(0.1, abs(trend_factor) * 0.5)
            else:  # UNDER prediction
                trend_adjustment = min(0.1, abs(trend_factor) * 0.5)
        
        # Confidence based on how far line is from weighted average
        if z_score < -1:
            prediction = 'OVER'
            confidence = min(0.85, 0.6 + abs(z_score) * 0.1)
        elif z_score > 1:
            prediction = 'UNDER'
            confidence = min(0.85, 0.6 + abs(z_score) * 0.1)
        elif z_score < 0:
            prediction = 'OVER'
            confidence = 0.5 + abs(z_score) * 0.15
        else:
            prediction = 'UNDER'
            confidence = 0.5 + abs(z_score) * 0.15
        
        # Apply trend adjustment to confidence
        confidence = confidence + trend_adjustment
        confidence = max(0.3, min(0.9, confidence))  # Clamp to reasonable range
        
        return {
            'success': True,
            'player_name': player_name,
            'position': history.get('position', 'Unknown'),
            'team': history.get('team', 'Unknown'),
            'stat_type': stat_type.replace('_', ' ').title(),
            'line': yards_line,
            'prediction': prediction,
            'confidence': confidence,
            'avg_yards': history['avg_yards'],
            'weighted_avg_yards': weighted_avg,  # Recent-weighted average
            'recent_avg_yards': recent_avg if len(recent_values) >= 3 else season_avg,
            'median_yards': history['median_yards'],
            'games_analyzed': history['games_played'],
            'hit_rate': hit_rate,
            'times_over': times_over,
            'last_5_games': history['last_5_games'],
            'recent_trend': 'improving' if trend_factor > 0.1 else ('declining' if trend_factor < -0.1 else 'stable'),
            'season': history.get('season', '2024'),
            'data_source': history.get('source', 'unknown'),
        }
    
    def _get_defensive_td_rate(self, opponent_team: str, stat_type: str, season: int = 2025) -> float:
        """Get defensive TD rate for opponent team.
        
        Args:
            opponent_team: Opponent team abbreviation
            stat_type: Type of TD ('rushing_td', 'receiving_td', 'anytime_td')
            season: Season year
            
        Returns:
            TD rate (0.0-1.0) - percentage of games where opponent allows this type of TD
        """
        try:
            import nfl_data_py as nfl
            
            # Get weekly stats for the season
            weekly_stats = nfl.import_weekly_data([season])
            
            if len(weekly_stats) == 0:
                return 0.5  # Default neutral rate
            
            # Get schedule to find games
            schedule = nfl.import_schedules([season])
            
            # Find completed games involving opponent_team
            opponent_games = schedule[
                ((schedule['home_team'] == opponent_team) | (schedule['away_team'] == opponent_team)) &
                pd.notna(schedule.get('home_score'))
            ]
            
            if len(opponent_games) == 0:
                return 0.5
            
            # For each game, calculate TDs scored against opponent_team
            total_tds_allowed = 0
            games_with_td = 0
            games_played = 0
            
            for _, game in opponent_games.iterrows():
                # Determine which team scored against opponent
                if game['home_team'] == opponent_team:
                    scoring_team = game['away_team']
                else:
                    scoring_team = game['home_team']
                
                # Get all TDs scored by that team in this game
                game_stats = weekly_stats[
                    (weekly_stats['season'] == game['season']) &
                    (weekly_stats['week'] == game['week']) &
                    (weekly_stats['recent_team'] == scoring_team)
                ]
                
                if len(game_stats) == 0:
                    continue
                
                # Calculate TDs of the relevant type
                if stat_type == 'rushing_td':
                    tds_in_game = game_stats['rushing_tds'].fillna(0).sum()
                elif stat_type == 'receiving_td':
                    tds_in_game = game_stats['receiving_tds'].fillna(0).sum()
                else:  # anytime_td
                    tds_in_game = (game_stats['rushing_tds'].fillna(0) + 
                                  game_stats['receiving_tds'].fillna(0)).sum()
                
                total_tds_allowed += tds_in_game
                if tds_in_game > 0:
                    games_with_td += 1
                games_played += 1
            
            if games_played == 0:
                return 0.5
            
            # Calculate rate: percentage of games where at least 1 TD was allowed
            td_rate = games_with_td / games_played
            
            # Also factor in average TDs per game for more nuance
            avg_tds_per_game = total_tds_allowed / games_played
            
            # Combine: base rate + adjustment for frequency
            # If they allow 2+ TDs per game on average, boost the rate
            if avg_tds_per_game >= 2.0:
                td_rate = min(0.9, td_rate + 0.1)
            elif avg_tds_per_game >= 1.5:
                td_rate = min(0.85, td_rate + 0.05)
            elif avg_tds_per_game <= 0.5:
                td_rate = max(0.2, td_rate - 0.1)
            
            # Normalize to reasonable range
            return max(0.2, min(0.9, td_rate))
            
        except Exception as e:
            logger.debug(f"Error calculating defensive TD rate: {e}")
            return 0.5  # Default neutral rate
    
    def _get_team_redzone_efficiency(self, team: str, season: int = 2025) -> float:
        """Get team's redzone efficiency (TD conversion rate).
        
        Args:
            team: Team abbreviation
            season: Season year
            
        Returns:
            Redzone efficiency (0.0-1.0) - percentage of redzone trips that result in TDs
        """
        try:
            import nfl_data_py as nfl
            
            # Get weekly stats
            weekly_stats = nfl.import_weekly_data([season - 1])  # Use previous year for historical data
            
            if len(weekly_stats) == 0:
                return 0.6  # Default moderate efficiency
            
            # Get team's stats
            team_stats = weekly_stats[weekly_stats['recent_team'] == team]
            
            if len(team_stats) == 0:
                return 0.6
            
            # Calculate total TDs scored
            total_tds = (team_stats['rushing_tds'].fillna(0) + 
                        team_stats['receiving_tds'].fillna(0)).sum()
            
            # Estimate redzone trips from TDs + field goals
            # Teams typically score TDs on ~60% of redzone trips
            # So redzone trips ≈ TDs / 0.6
            # But we can also use total points as proxy
            # Average: ~3.5 redzone trips per game, ~2.1 TDs per game
            games_played = team_stats['week'].nunique()
            if games_played == 0:
                return 0.6
            
            avg_tds_per_game = total_tds / games_played
            
            # Redzone efficiency = TDs per redzone trip
            # Estimate: teams average ~3.5 redzone trips per game
            # Efficiency = avg_tds_per_game / 3.5
            estimated_rz_trips_per_game = 3.5
            efficiency = avg_tds_per_game / estimated_rz_trips_per_game
            
            # Normalize to reasonable range (0.4 to 0.8)
            return max(0.4, min(0.8, efficiency))
            
        except Exception as e:
            logger.debug(f"Error calculating redzone efficiency: {e}")
            return 0.6  # Default moderate efficiency
    
    def _get_team_redzone_play_type(self, team: str, stat_type: str, season: int = 2025) -> float:
        """Get team's redzone play type distribution (rush vs pass).
        
        Args:
            team: Team abbreviation
            stat_type: Type of TD ('rushing_td', 'receiving_td', 'anytime_td')
            season: Season year
            
        Returns:
            Rush ratio (0.0-1.0) - percentage of redzone TDs that are rushing
            For receiving_td, returns pass ratio (1 - rush ratio)
        """
        try:
            import nfl_data_py as nfl
            
            # Get weekly stats
            weekly_stats = nfl.import_weekly_data([season - 1])  # Use previous year
            
            if len(weekly_stats) == 0:
                return 0.5  # Default balanced
            
            # Get team's stats
            team_stats = weekly_stats[weekly_stats['recent_team'] == team]
            
            if len(team_stats) == 0:
                return 0.5
            
            # Calculate rushing vs receiving TDs
            rushing_tds = team_stats['rushing_tds'].fillna(0).sum()
            receiving_tds = team_stats['receiving_tds'].fillna(0).sum()
            total_tds = rushing_tds + receiving_tds
            
            if total_tds == 0:
                return 0.5
            
            # Calculate rush ratio
            rush_ratio = rushing_tds / total_tds
            
            # For receiving TD queries, return pass ratio
            if stat_type == 'receiving_td':
                return 1 - rush_ratio
            
            # For rushing TD or anytime TD, return rush ratio
            return rush_ratio
            
        except Exception as e:
            logger.debug(f"Error calculating redzone play type: {e}")
            return 0.5  # Default balanced
    
    def _get_redzone_usage(self, player_name: str, stat_type: str, season: int = 2025) -> float:
        """Get player's redzone usage percentage.
        
        Args:
            player_name: Player display name
            stat_type: Type of stat ('rushing_td', 'receiving_td', 'anytime_td')
            season: Season year
            
        Returns:
            Redzone usage rate (0.0-1.0) - percentage of redzone plays where player is involved
        """
        try:
            import nfl_data_py as nfl
            
            # Get weekly stats
            weekly_stats = nfl.import_weekly_data([season])
            
            if len(weekly_stats) == 0:
                return 0.3  # Default moderate usage
            
            # Get player's stats
            player_stats = weekly_stats[
                weekly_stats['player_display_name'] == player_name
            ]
            
            if len(player_stats) == 0:
                return 0.3
            
            # Get player's team
            player_team = player_stats['recent_team'].iloc[0] if 'recent_team' in player_stats.columns else None
            if not player_team:
                return 0.3
            
            # Calculate redzone usage
            # Since nfl_data_py doesn't have explicit redzone snap data, we use:
            # - Player's share of team's TDs of this type (proxy for redzone involvement)
            # - Also factor in player's total touches (carries + targets) vs team total
            
            # Get team's total stats
            team_stats = weekly_stats[weekly_stats['recent_team'] == player_team]
            
            if stat_type == 'receiving_td':
                player_tds = player_stats['receiving_tds'].fillna(0).sum()
                team_tds = team_stats['receiving_tds'].fillna(0).sum()
                # Also factor in targets as proxy for redzone involvement
                player_targets = player_stats['targets'].fillna(0).sum()
                team_targets = team_stats['targets'].fillna(0).sum()
            elif stat_type == 'rushing_td':
                player_tds = player_stats['rushing_tds'].fillna(0).sum()
                team_tds = team_stats['rushing_tds'].fillna(0).sum()
                # Factor in carries
                player_carries = player_stats['carries'].fillna(0).sum()
                team_carries = team_stats['carries'].fillna(0).sum()
            else:  # anytime_td
                player_tds = (player_stats['rushing_tds'].fillna(0) + player_stats['receiving_tds'].fillna(0)).sum()
                team_tds = (team_stats['rushing_tds'].fillna(0) + team_stats['receiving_tds'].fillna(0)).sum()
                # Combine touches
                player_touches = (player_stats['carries'].fillna(0) + player_stats['targets'].fillna(0)).sum()
                team_touches = (team_stats['carries'].fillna(0) + team_stats['targets'].fillna(0)).sum()
            
            if team_tds == 0:
                return 0.3
            
            # Primary metric: player's share of team's TDs (most direct redzone indicator)
            td_share = player_tds / team_tds
            
            # Secondary metric: player's share of touches (usage indicator)
            if stat_type == 'receiving_td':
                touch_share = player_targets / team_targets if team_targets > 0 else 0.3
            elif stat_type == 'rushing_td':
                touch_share = player_carries / team_carries if team_carries > 0 else 0.3
            else:
                touch_share = player_touches / team_touches if team_touches > 0 else 0.3
            
            # Weighted average: 70% TD share, 30% touch share
            usage_rate = (td_share * 0.7) + (touch_share * 0.3)
            
            # Normalize to reasonable range (0.1 to 0.9)
            return max(0.1, min(0.9, usage_rate))
            
        except Exception as e:
            logger.debug(f"Error calculating redzone usage: {e}")
            return 0.3  # Default moderate usage
    
    def _get_player_team_from_schedule(self, player_name: str, season: int = 2025) -> Optional[str]:
        """Get player's team from historical stats or schedule lookup.
        
        Args:
            player_name: Player name
            season: Season year
            
        Returns:
            Team abbreviation or None
        """
        try:
            # Try to get from historical stats first
            if self.player_stats is None:
                self.load_player_stats([season - 1])  # Try previous year
            
            if self.player_stats is not None and len(self.player_stats) > 0:
                player_data = self.player_stats[
                    self.player_stats['player_display_name'] == player_name
                ]
                if len(player_data) > 0 and 'recent_team' in player_data.columns:
                    team = player_data['recent_team'].iloc[0]
                    if pd.notna(team) and team != 'NFL':
                        return team
            
            # Fallback: Use a known player-to-team mapping for 2025
            # This is a simplified approach - in production, would use roster data
            player_team_map_2025 = {
                'Saquon Barkley': 'PHI',
                'A.J. Brown': 'PHI',
                'Devonta Smith': 'PHI',
                'Jalen Hurts': 'PHI',
                'Ladd McConkey': 'LAC',
                'Quentin Johnston': 'LAC',
                'Justin Herbert': 'LAC',
                'Gus Edwards': 'LAC',
                'Omarion Hampton': 'DEN',
                'J.K. Dobbins': 'DEN',
                'Courtland Sutton': 'DEN',
            }
            
            return player_team_map_2025.get(player_name)
                
        except Exception as e:
            logger.debug(f"Error getting player team: {e}")
            return None
    
    def _find_player_opponent(self, player_name: str, player_team: str) -> Optional[str]:
        """Find the opponent team for a player's next game.
        
        Args:
            player_name: Player name
            player_team: Player's team abbreviation (may be 'NFL' or 'Unknown')
            
        Returns:
            Opponent team abbreviation or None
        """
        try:
            import nfl_data_py as nfl
            from datetime import datetime
            
            # If team is not valid, try to get it
            if player_team in ['NFL', 'Unknown', None]:
                player_team = self._get_player_team_from_schedule(player_name)
                if not player_team:
                    return None
            
            # Get current season
            current_year = datetime.now().year
            current_month = datetime.now().month
            if current_month < 3:
                season = current_year - 1
            else:
                season = current_year
            
            # Get schedule
            schedule = nfl.import_schedules([season])
            
            # Normalize team abbreviation (handle LA vs LAR)
            team_variants = [player_team]
            if player_team == 'LAR':
                team_variants.append('LA')
            elif player_team == 'LA':
                team_variants.append('LAR')
            
            # Find next game for player's team
            today = datetime.now().strftime('%Y-%m-%d')
            upcoming_games = pd.DataFrame()
            
            for variant in team_variants:
                games = schedule[
                    ((schedule['home_team'] == variant) | (schedule['away_team'] == variant)) &
                    (schedule['gameday'] >= today)
                ]
                if len(games) > 0:
                    upcoming_games = games
                    break
            
            if len(upcoming_games) == 0:
                return None
            
            next_game = upcoming_games.sort_values('gameday').iloc[0]
            
            # Return opponent (normalize LA to LAR)
            if next_game['home_team'] in team_variants:
                opp = next_game['away_team']
            else:
                opp = next_game['home_team']
            
            # Normalize opponent abbreviation
            if opp == 'LA':
                return 'LAR'
            return opp
                
        except Exception as e:
            logger.debug(f"Error finding opponent: {e}")
            return None
    
    def _predict_td(self, player_name: str, stat_type: str, history: Dict[str, Any]) -> Dict[str, Any]:
        """Predict if player will score a touchdown.
        
        Args:
            player_name: Player display name
            stat_type: Type of TD stat (anytime_td, rushing_td, receiving_td)
            history: Historical TD data
            
        Returns:
            Prediction dictionary
        """
        td_rate = history.get('td_rate', 0)
        games_played = history.get('games_played', 0)
        games_with_td = history.get('games_with_td', 0)
        total_tds = history.get('total_tds', 0)
        avg_tds = history.get('avg_tds_per_game', 0)
        player_team = history.get('team', 'Unknown')
        season = int(history.get('season', 2025))
        
        # Get actual player team (may be 'NFL' from stats collector)
        actual_team = self._get_player_team_from_schedule(player_name, season) if player_team in ['NFL', 'Unknown'] else player_team
        
        # Get opponent defensive TD rate
        opponent_team = self._find_player_opponent(player_name, actual_team or player_team)
        defensive_td_rate = 0.5  # Default neutral
        if opponent_team:
            defensive_td_rate = self._get_defensive_td_rate(opponent_team, stat_type, season)
        
        # Get player redzone usage
        redzone_usage = self._get_redzone_usage(player_name, stat_type, season)
        
        # Get team redzone efficiency
        team_rz_efficiency = 0.6  # Default moderate
        if actual_team:
            team_rz_efficiency = self._get_team_redzone_efficiency(actual_team, season)
        
        # Get team redzone play type distribution
        team_rz_play_type = 0.5  # Default balanced
        if actual_team:
            team_rz_play_type = self._get_team_redzone_play_type(actual_team, stat_type, season)
        
        # Calculate recent TD rate with exponential decay weighting (most recent games weighted more)
        game_tds = history.get('game_tds', [])
        last_5_games = history.get('last_5_games', [])
        
        # Extract TD counts from last 5 games
        recent_td_counts = []
        if game_tds and len(game_tds) >= 5:
            recent_td_counts = game_tds[-5:]  # Last 5 games
        elif last_5_games:
            # Try to extract from last_5_games format "W14: 1 TD"
            for val in last_5_games[-5:]:
                if isinstance(val, str) and 'TD' in val.upper():
                    try:
                        # Extract number before "TD"
                        td_count = float(val.split('TD')[0].split()[-1])
                        recent_td_counts.append(td_count)
                    except:
                        continue
        
        # Calculate recent TD rate with exponential weighting
        if len(recent_td_counts) >= 3:
            # Use exponential decay weights: most recent games weighted more heavily
            n_recent = len(recent_td_counts)
            weights = []
            total_weight = 0
            
            # Generate exponential decay weights
            for i in range(n_recent):
                weight = 0.4 * (0.7 ** i)  # Most recent gets 0.4, then 0.28, 0.196, etc.
                weights.append(weight)
                total_weight += weight
            
            # Normalize weights
            weights = [w / total_weight for w in weights]
            
            # Calculate weighted TD rate (games with TD weighted by recency)
            weighted_td_rate_recent = sum((1 if tds > 0 else 0) * weight for tds, weight in zip(recent_td_counts, weights))
            
            # Combine with season rate: 75% recent (exponentially weighted), 25% season
            weighted_td_rate = (weighted_td_rate_recent * 0.75) + (td_rate * 0.25)
            
            # Calculate trend (most recent 2 games vs previous 2 games)
            if len(recent_td_counts) >= 4:
                most_recent_rate = sum(1 for tds in recent_td_counts[:2] if tds > 0) / 2
                previous_rate = sum(1 for tds in recent_td_counts[2:4] if tds > 0) / 2
                td_trend = most_recent_rate - previous_rate
            else:
                td_trend = 0
        else:
            # Not enough recent data, use season rate
            recent_td_rate = td_rate
            weighted_td_rate_recent = td_rate
            weighted_td_rate = td_rate
            td_trend = 0
        
        # Base prediction from weighted TD rate (recent-weighted)
        base_confidence = 0.5
        if weighted_td_rate >= 0.5:
            base_prediction = 'YES'
            base_confidence = min(0.85, 0.5 + (weighted_td_rate - 0.5) * 0.7)
        elif weighted_td_rate >= 0.35:
            base_prediction = 'YES'
            base_confidence = 0.5 + (weighted_td_rate - 0.35) * 0.5
        elif weighted_td_rate >= 0.25:
            base_prediction = 'NO'
            base_confidence = 0.5 + (0.35 - weighted_td_rate) * 0.5
        else:
            base_prediction = 'NO'
            base_confidence = min(0.85, 0.5 + (0.25 - weighted_td_rate) * 0.7)
        
        # Adjust confidence based on recent TD trend
        # If player is scoring more TDs recently, boost confidence for YES
        trend_adjustment = 0.0
        if td_trend > 0.2:  # Significantly improving
            if base_prediction == 'YES':
                trend_adjustment = min(0.1, td_trend * 0.3)
            else:
                trend_adjustment = -min(0.1, td_trend * 0.3)
        elif td_trend < -0.2:  # Significantly declining
            if base_prediction == 'YES':
                trend_adjustment = -min(0.1, abs(td_trend) * 0.3)
            else:
                trend_adjustment = min(0.1, abs(td_trend) * 0.3)
        
        # Adjust confidence based on defensive TD rate
        # If defense allows TDs frequently (high rate), boost confidence for YES
        # If defense is stingy (low rate), reduce confidence for YES
        defensive_adjustment = 0.0
        if base_prediction == 'YES':
            # High defensive TD rate = good for player (boost confidence)
            # Low defensive TD rate = bad for player (reduce confidence)
            defensive_adjustment = (defensive_td_rate - 0.5) * 0.15
        else:
            # For NO predictions, reverse the logic
            defensive_adjustment = (0.5 - defensive_td_rate) * 0.15
        
        # Adjust confidence based on redzone usage
        # Higher redzone usage = player is more involved in scoring situations
        redzone_adjustment = (redzone_usage - 0.3) * 0.2  # Scale from -0.04 to +0.12
        
        # Adjust confidence based on team redzone efficiency
        # Higher efficiency = team converts more redzone trips to TDs
        efficiency_adjustment = (team_rz_efficiency - 0.6) * 0.15  # Scale from -0.03 to +0.03
        
        # Adjust confidence based on team redzone play type
        # If rushing TD and team rushes a lot in redzone = boost
        # If receiving TD and team passes a lot in redzone = boost
        play_type_adjustment = 0.0
        if base_prediction == 'YES':
            # For rushing TD: high rush ratio is good
            # For receiving TD: high pass ratio (low rush ratio) is good
            if stat_type == 'rushing_td':
                # Rush ratio > 0.5 = good for rushing TD
                play_type_adjustment = (team_rz_play_type - 0.5) * 0.1
            elif stat_type == 'receiving_td':
                # Pass ratio > 0.5 = good for receiving TD (team_rz_play_type is already pass ratio)
                play_type_adjustment = (team_rz_play_type - 0.5) * 0.1
            # For anytime TD, play type doesn't matter as much
        
        # Combine adjustments (including recent trend)
        final_confidence = base_confidence + defensive_adjustment + redzone_adjustment + efficiency_adjustment + play_type_adjustment + trend_adjustment
        final_confidence = max(0.3, min(0.9, final_confidence))  # Clamp to reasonable range
        
        prediction = base_prediction
        
        # Format stat type for display
        td_type_display = {
            'anytime_td': 'Anytime Touchdown',
            'rushing_td': 'Rushing Touchdown',
            'receiving_td': 'Receiving Touchdown',
            'passing_td': 'Passing Touchdown',
        }.get(stat_type, 'Touchdown')
        
        return {
            'success': True,
            'is_td_prop': True,
            'player_name': player_name,
            'position': history.get('position', 'Unknown'),
            'team': history.get('team', 'Unknown'),
            'stat_type': td_type_display,
            'prediction': prediction,
            'confidence': final_confidence,
            'td_rate': td_rate,
            'weighted_td_rate': weighted_td_rate,  # Recent-weighted TD rate
            'recent_td_rate': weighted_td_rate_recent if len(recent_td_counts) >= 3 else td_rate,
            'games_with_td': games_with_td,
            'recent_td_trend': 'improving' if td_trend > 0.2 else ('declining' if td_trend < -0.2 else 'stable'),
            'total_tds': total_tds,
            'avg_tds_per_game': avg_tds,
            'games_analyzed': games_played,
            'last_5_games': history['last_5_games'],
            'season': history.get('season', '2024'),
            'data_source': history.get('source', 'unknown'),
            # New features
            'defensive_td_rate': defensive_td_rate,
            'redzone_usage': redzone_usage,
            'opponent_team': opponent_team if opponent_team else 'Unknown',
            'team_redzone_efficiency': team_rz_efficiency,
            'team_redzone_play_type': team_rz_play_type,
            'player_team': actual_team if actual_team else player_team,
        }


class QuickPredictor:
    """Quick prediction system for NFL games using natural language queries."""
    
    # Team name mapping (comprehensive)
    TEAM_MAPPING = {
        # Full names
        'chiefs': 'KC', 'kansas city': 'KC', 'kansas city chiefs': 'KC',
        'raiders': 'LV', 'las vegas': 'LV', 'las vegas raiders': 'LV',
        'broncos': 'DEN', 'denver': 'DEN', 'denver broncos': 'DEN',
        'chargers': 'LAC', 'los angeles chargers': 'LAC', 'la chargers': 'LAC',
        'bills': 'BUF', 'buffalo': 'BUF', 'buffalo bills': 'BUF',
        'dolphins': 'MIA', 'miami': 'MIA', 'miami dolphins': 'MIA',
        'patriots': 'NE', 'new england': 'NE', 'new england patriots': 'NE',
        'jets': 'NYJ', 'new york jets': 'NYJ', 'ny jets': 'NYJ',
        'ravens': 'BAL', 'baltimore': 'BAL', 'baltimore ravens': 'BAL',
        'bengals': 'CIN', 'cincinnati': 'CIN', 'cincinnati bengals': 'CIN',
        'browns': 'CLE', 'cleveland': 'CLE', 'cleveland browns': 'CLE',
        'steelers': 'PIT', 'pittsburgh': 'PIT', 'pittsburgh steelers': 'PIT',
        'texans': 'HOU', 'houston': 'HOU', 'houston texans': 'HOU',
        'colts': 'IND', 'indianapolis': 'IND', 'indianapolis colts': 'IND',
        'jaguars': 'JAX', 'jacksonville': 'JAX', 'jacksonville jaguars': 'JAX',
        'titans': 'TEN', 'tennessee': 'TEN', 'tennessee titans': 'TEN',
        'cowboys': 'DAL', 'dallas': 'DAL', 'dallas cowboys': 'DAL',
        'giants': 'NYG', 'new york giants': 'NYG', 'ny giants': 'NYG',
        'eagles': 'PHI', 'philadelphia': 'PHI', 'philadelphia eagles': 'PHI',
        'commanders': 'WAS', 'washington': 'WAS', 'washington commanders': 'WAS',
        'bears': 'CHI', 'chicago': 'CHI', 'chicago bears': 'CHI',
        'lions': 'DET', 'detroit': 'DET', 'detroit lions': 'DET',
        'packers': 'GB', 'green bay': 'GB', 'green bay packers': 'GB',
        'vikings': 'MIN', 'minnesota': 'MIN', 'minnesota vikings': 'MIN',
        'falcons': 'ATL', 'atlanta': 'ATL', 'atlanta falcons': 'ATL',
        'panthers': 'CAR', 'carolina': 'CAR', 'carolina panthers': 'CAR',
        'saints': 'NO', 'new orleans': 'NO', 'new orleans saints': 'NO',
        'buccaneers': 'TB', 'tampa bay': 'TB', 'tampa bay buccaneers': 'TB', 'bucs': 'TB',
        'cardinals': 'ARI', 'arizona': 'ARI', 'arizona cardinals': 'ARI',
        'rams': 'LAR', 'los angeles rams': 'LAR', 'la rams': 'LAR',
        '49ers': 'SF', 'san francisco': 'SF', 'san francisco 49ers': 'SF', 'niners': 'SF',
        'seahawks': 'SEA', 'seattle': 'SEA', 'seattle seahawks': 'SEA',
    }
    
    TEAM_FULL_NAMES = {
        'KC': 'Kansas City Chiefs', 'LV': 'Las Vegas Raiders', 'DEN': 'Denver Broncos',
        'LAC': 'Los Angeles Chargers', 'BUF': 'Buffalo Bills', 'MIA': 'Miami Dolphins',
        'NE': 'New England Patriots', 'NYJ': 'New York Jets', 'BAL': 'Baltimore Ravens',
        'CIN': 'Cincinnati Bengals', 'CLE': 'Cleveland Browns', 'PIT': 'Pittsburgh Steelers',
        'HOU': 'Houston Texans', 'IND': 'Indianapolis Colts', 'JAX': 'Jacksonville Jaguars',
        'TEN': 'Tennessee Titans', 'DAL': 'Dallas Cowboys', 'NYG': 'New York Giants',
        'PHI': 'Philadelphia Eagles', 'WAS': 'Washington Commanders', 'CHI': 'Chicago Bears',
        'DET': 'Detroit Lions', 'GB': 'Green Bay Packers', 'MIN': 'Minnesota Vikings',
        'ATL': 'Atlanta Falcons', 'CAR': 'Carolina Panthers', 'NO': 'New Orleans Saints',
        'TB': 'Tampa Bay Buccaneers', 'ARI': 'Arizona Cardinals', 
        'LAR': 'Los Angeles Rams', 'LA': 'Los Angeles Rams',  # Handle both abbreviations
        'SF': 'San Francisco 49ers', 'SEA': 'Seattle Seahawks',
    }
    
    def __init__(self, model_dir: str = 'models/trained'):
        """Initialize the quick predictor.
        
        Args:
            model_dir: Directory containing trained models (kept for API compatibility, not currently used)
        """
        # Import team key players and injured players from parlay builder
        try:
            from src.pipeline.parlay_builder import ParlayBuilder
            self.TEAM_KEY_PLAYERS = ParlayBuilder.TEAM_KEY_PLAYERS
            self.INJURED_PLAYERS = ParlayBuilder.INJURED_PLAYERS
        except:
            # Fallback if import fails
            self.TEAM_KEY_PLAYERS = {}
            self.INJURED_PLAYERS = {}
        
        # Initialize stats collector for team performance analysis
        self.stats_collector = None
        self._use_multi_source = True
        self.schedule = None
        self.player_prop_predictor = PlayerPropPredictor()
    
    def is_player_prop_query(self, query: str) -> bool:
        """Check if query is about a player prop (yards, touchdowns, etc.).
        
        Args:
            query: Natural language query
            
        Returns:
            True if this is a player prop query
        """
        query_lower = query.lower()
        
        # TD-specific keywords (don't require a number)
        td_keywords = ['touchdown', ' td', 'score', 'scorer', 'anytime td']
        is_td_query = any(keyword in query_lower for keyword in td_keywords)
        
        # Player prop indicators (for yards, etc.)
        prop_keywords = [
            'yards', 'yds', 'yrds', 'receiving', 'rushing', 'passing',
            'total yards', 'total yds', 'scrimmage', 'combined',
            'receptions', 'catches', 'targets',
            'will he get', 'will she get', 'over', 'under', 'hit'
        ]
        
        # Check for prop keywords
        has_prop_keyword = any(keyword in query_lower for keyword in prop_keywords)
        
        # Check for numbers (likely a line)
        has_number = bool(re.search(r'\d+', query))
        
        # Check if it's NOT a team vs team query
        team1, team2 = self.extract_teams(query)
        is_team_query = team1 is not None and team2 is not None
        
        # TD queries don't need a number; yards queries do
        if is_td_query and not is_team_query:
            return True
        
        return has_prop_keyword and has_number and not is_team_query
        
    def extract_teams(self, query: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract team names from a natural language query.
        
        Args:
            query: Natural language query like "Chiefs vs Raiders"
            
        Returns:
            Tuple of (team1_abbr, team2_abbr) or (None, None) if not found
        """
        query_lower = query.lower()
        
        # Find all team mentions
        found_teams = []
        
        # Sort by length (longest first) to match full names before partial
        sorted_teams = sorted(self.TEAM_MAPPING.keys(), key=len, reverse=True)
        
        for team_name in sorted_teams:
            if team_name in query_lower:
                abbr = self.TEAM_MAPPING[team_name]
                if abbr not in found_teams:
                    found_teams.append(abbr)
                    # Remove to avoid double counting
                    query_lower = query_lower.replace(team_name, '', 1)
        
        if len(found_teams) >= 2:
            return found_teams[0], found_teams[1]
        elif len(found_teams) == 1:
            return found_teams[0], None
        
        return None, None
    
    def get_current_schedule(self) -> pd.DataFrame:
        """Get the current NFL schedule with betting lines.
        
        Returns:
            DataFrame with current season schedule
        """
        if self.schedule is not None:
            return self.schedule
        
        import nfl_data_py as nfl
        
        # Get current season
        current_year = datetime.now().year
        current_month = datetime.now().month
        
        # NFL season spans two calendar years
        if current_month < 3:
            season = current_year - 1
        else:
            season = current_year
        
        logger.info(f"Loading {season} NFL schedule...")
        self.schedule = nfl.import_schedules([season])
        
        return self.schedule
    
    def _convert_team_abbr_for_schedule(self, team_abbr: str) -> List[str]:
        """Convert team abbreviation to schedule format.
        
        The schedule data uses 'LA' for Rams, but our system uses 'LAR'.
        This function returns both possible abbreviations to check.
        
        Args:
            team_abbr: Team abbreviation from our system
            
        Returns:
            List of possible abbreviations to check in schedule
        """
        # Map our abbreviations to schedule abbreviations
        conversion_map = {
            'LAR': ['LA', 'LAR'],  # Schedule uses 'LA' for Rams
        }
        return conversion_map.get(team_abbr, [team_abbr])
    
    def _get_stats_collector(self):
        """Lazy load multi-source stats collector."""
        if self.stats_collector is None:
            try:
                from src.data.collectors.multi_source_stats import MultiSourceStatsCollector
                self.stats_collector = MultiSourceStatsCollector()
                logger.debug("Multi-source stats collector initialized for team performance")
            except ImportError as e:
                logger.warning(f"Could not import multi-source collector: {e}")
                self._use_multi_source = False
        return self.stats_collector
    
    def _get_team_performance_metrics(self, team: str) -> Dict[str, Any]:
        """Calculate team performance metrics based on recent player stats.
        
        Args:
            team: Team abbreviation (e.g., 'KC', 'PHI')
            
        Returns:
            Dictionary with performance metrics:
            {
                'performance_score': float (-1.0 to 1.0, higher is better),
                'recent_offense_score': float,
                'key_injuries': int,
                'players_analyzed': int,
                'trend': str ('improving', 'declining', 'stable')
            }
        """
        try:
            # Get key players for the team
            key_players = getattr(self, 'TEAM_KEY_PLAYERS', {}).get(team, [])
            if not key_players:
                # No key players data, return neutral
                return {
                    'performance_score': 0.0,
                    'recent_offense_score': 0.0,
                    'key_injuries': 0,
                    'players_analyzed': 0,
                    'trend': 'unknown'
                }
            
            collector = self._get_stats_collector()
            if not collector:
                return {
                    'performance_score': 0.0,
                    'recent_offense_score': 0.0,
                    'key_injuries': 0,
                    'players_analyzed': 0,
                    'trend': 'unknown'
                }
            
            # Analyze key offensive players (QB, RB, WR, TE)
            player_scores = []
            key_injuries = 0
            players_analyzed = 0
            
            # Common QB names to identify quarterbacks
            qb_keywords = ['mahomes', 'allen', 'hurts', 'burrow', 'stroud', 'herbert', 'goff', 
                          'jackson', 'prescott', 'purdy', 'love', 'rodgers', 'carr', 'cousins',
                          'mayfield', 'watson', 'wilson', 'murray', 'stafford', 'howell', 'dart',
                          'williams', 'daniels', 'maye', 'nix', 'richardson', 'jones']
            
            for player_name in key_players[:6]:  # Top 6 key players
                # Check if injured
                injured_players = getattr(self, 'INJURED_PLAYERS', {})
                if player_name in injured_players:
                    status = injured_players[player_name].upper()
                    if status in ['IR', 'OUT']:
                        key_injuries += 1
                        continue  # Skip injured players
                
                # Determine stat type based on player position
                # Try to identify QB by name (common QB names)
                player_lower = player_name.lower()
                is_qb = any(qb in player_lower for qb in qb_keywords)
                
                # Try multiple stat types in order of preference
                stat_types_to_try = []
                if is_qb:
                    stat_types_to_try = ['passing_yards', 'total_yards']
                else:
                    # For skill players, try receiving first, then rushing, then total
                    stat_types_to_try = ['receiving_yards', 'rushing_yards', 'total_yards']
                
                player_score = None
                for stat_type in stat_types_to_try:
                    try:
                        stats = collector.get_player_weekly_stats(player_name, stat_type)
                        if stats.get('success'):
                            players_analyzed += 1
                            
                            # Get recent performance (last 3 games average vs season average)
                            game_yards = stats.get('game_yards', [])
                            if len(game_yards) >= 2:  # Need at least 2 games
                                recent_avg = sum(game_yards[:min(3, len(game_yards))]) / min(3, len(game_yards))
                                season_avg = stats.get('avg_yards', recent_avg)
                                
                                if season_avg > 0:
                                    # Performance ratio: >1.0 means playing above average
                                    perf_ratio = recent_avg / season_avg
                                    # Normalize to -1 to 1 scale
                                    # 1.2x = +0.4, 0.8x = -0.4
                                    player_score = min(1.0, max(-1.0, (perf_ratio - 1.0) * 2))
                                    break  # Found stats, move to next player
                    except Exception as e:
                        logger.debug(f"Error getting stats for {player_name} ({stat_type}): {e}")
                        continue
                
                if player_score is not None:
                    player_scores.append(player_score)
            
            # Calculate team performance score
            if players_analyzed == 0:
                performance_score = 0.0
                recent_offense_score = 0.0
                trend = 'unknown'
            else:
                # Average of player scores
                performance_score = sum(player_scores) / len(player_scores) if player_scores else 0.0
                recent_offense_score = performance_score
                
                # Determine trend
                if performance_score > 0.2:
                    trend = 'improving'
                elif performance_score < -0.2:
                    trend = 'declining'
                else:
                    trend = 'stable'
            
            # Penalize for key injuries
            if key_injuries > 0:
                injury_penalty = min(0.3, key_injuries * 0.1)  # Max 0.3 penalty
                performance_score = max(-1.0, performance_score - injury_penalty)
            
            return {
                'performance_score': round(performance_score, 3),
                'recent_offense_score': round(recent_offense_score, 3),
                'key_injuries': key_injuries,
                'players_analyzed': players_analyzed,
                'trend': trend
            }
        
        except Exception as e:
            logger.debug(f"Error calculating team performance for {team}: {e}")
            return {
                'performance_score': 0.0,
                'recent_offense_score': 0.0,
                'key_injuries': 0,
                'players_analyzed': 0,
                'trend': 'unknown'
            }
    
    def find_game(self, team1: str, team2: str) -> Optional[Dict[str, Any]]:
        """Find an upcoming or recent game between two teams.
        
        Args:
            team1: First team abbreviation
            team2: Second team abbreviation
            
        Returns:
            Game info dict or None if not found
        """
        schedule = self.get_current_schedule()
        
        # Convert team abbreviations to schedule format (handle LA vs LAR)
        team1_variants = self._convert_team_abbr_for_schedule(team1)
        team2_variants = self._convert_team_abbr_for_schedule(team2)
        
        # Find games between these teams (check all variants)
        games = pd.DataFrame()
        for t1 in team1_variants:
            for t2 in team2_variants:
                matches = schedule[
                    ((schedule['home_team'] == t1) & (schedule['away_team'] == t2)) |
                    ((schedule['home_team'] == t2) & (schedule['away_team'] == t1))
                ]
                if len(matches) > 0:
                    games = pd.concat([games, matches])
        
        if len(games) == 0:
            return None
        
        # Remove duplicates and sort
        games = games.drop_duplicates().sort_values('gameday', ascending=False)
        
        if len(games) == 0:
            return None
        
        # Get the most recent/upcoming game
        game = games.iloc[0]
        
        # Normalize team abbreviations (convert schedule format to our format)
        def normalize_team_abbr(abbr: str) -> str:
            """Convert schedule abbreviations to our standard format."""
            if abbr == 'LA':  # Schedule uses 'LA' for Rams
                return 'LAR'
            return abbr
        
        home_team = normalize_team_abbr(game['home_team'])
        away_team = normalize_team_abbr(game['away_team'])
        week = int(game['week'])
        
        # Check for manual betting lines override (for accurate current lines)
        from src.pipeline.parlay_builder import ParlayBuilder
        override_key = (week, away_team, home_team)
        if override_key in ParlayBuilder.BETTING_LINES_OVERRIDE:
            override = ParlayBuilder.BETTING_LINES_OVERRIDE[override_key]
            logger.debug(f"Using manual betting lines override for {away_team} @ {home_team} Week {week}")
            return {
                'home_team': home_team,
                'away_team': away_team,
                'week': week,
                'gameday': game['gameday'],
                'spread': override.get('spread', 0),  # From home team perspective: negative = home favorite
                'total': override.get('total', 45),
                'home_ml': override.get('home_ml', -110),
                'away_ml': override.get('away_ml', -110),
                'completed': pd.notna(game.get('home_score')),
                'home_score': game.get('home_score'),
                'away_score': game.get('away_score'),
            }
        
        # Use data from nfl_data_py (may be outdated)
        return {
            'home_team': home_team,
            'away_team': away_team,
            'week': week,
            'gameday': game['gameday'],
            'spread': game.get('spread_line', 0) if pd.notna(game.get('spread_line')) else 0,
            'total': game.get('total_line', 45) if pd.notna(game.get('total_line')) else 45,
            'home_ml': game.get('home_moneyline', -110) if pd.notna(game.get('home_moneyline')) else -110,
            'away_ml': game.get('away_moneyline', -110) if pd.notna(game.get('away_moneyline')) else -110,
            'completed': pd.notna(game.get('home_score')),
            'home_score': game.get('home_score'),
            'away_score': game.get('away_score'),
        }
    
    def predict_player_prop(self, query: str) -> Dict[str, Any]:
        """Make a player prop prediction.
        
        Args:
            query: Query like "Will Devonta Smith get 40 yards?" or "Will Hampton score a touchdown?"
            
        Returns:
            Prediction dictionary
        """
        # Find player
        player_name = self.player_prop_predictor.find_player(query)
        if not player_name:
            return {
                'success': False,
                'error': "Could not find player name in query",
                'hint': "Try: 'Will Devonta Smith get 40 yards?' or 'Omarion Hampton touchdown'"
            }
        
        # Extract stat type first to check if it's a TD query
        stat_type = self.player_prop_predictor.extract_stat_type(query)
        
        # For TD queries, we don't need a yards line
        if stat_type in ['anytime_td', 'rushing_td', 'receiving_td', 'passing_td']:
            # Make TD prediction
            return self.player_prop_predictor.predict_over_under(player_name, 0.5, stat_type)
        
        # Extract yards line for yards-based queries
        yards_line = self.player_prop_predictor.extract_yards_line(query)
        if not yards_line:
            return {
                'success': False,
                'error': "Could not find yards line in query",
                'hint': "Include a number like '40 yards' or 'over 50'"
            }
        
        # Make yards prediction
        return self.player_prop_predictor.predict_over_under(player_name, yards_line, stat_type)
    
    def predict(self, query: str) -> Dict[str, Any]:
        """Make a prediction based on a natural language query.
        
        Args:
            query: Query like "Chiefs vs Raiders" or "Will Devonta Smith get 40 yards?"
            
        Returns:
            Dictionary with prediction results
        """
        # Check if this is a player prop query
        if self.is_player_prop_query(query):
            result = self.predict_player_prop(query)
            if result['success']:
                result['query_type'] = 'player_prop'
                return result
        
        # Extract teams for game outcome prediction
        team1, team2 = self.extract_teams(query)
        
        if not team1 or not team2:
            # Could be a player prop query that didn't match
            result = self.predict_player_prop(query)
            if result['success']:
                result['query_type'] = 'player_prop'
                return result
            
            return {
                'success': False,
                'error': f"Could not find two teams or a player prop in query: '{query}'",
                'hint': "Try: 'Chiefs vs Raiders' or 'Will Devonta Smith get 40 yards?'"
            }
        
        # Find the game
        game = self.find_game(team1, team2)
        
        if not game:
            return {
                'success': False,
                'error': f"No game found between {team1} and {team2} this season"
            }
        
        # Check if game already completed
        if game['completed']:
            winner = game['home_team'] if game['home_score'] > game['away_score'] else game['away_team']
            return {
                'success': True,
                'completed': True,
                'home_team': game['home_team'],
                'away_team': game['away_team'],
                'home_team_full': self.TEAM_FULL_NAMES.get(game['home_team'], game['home_team']),
                'away_team_full': self.TEAM_FULL_NAMES.get(game['away_team'], game['away_team']),
                'week': game['week'],
                'gameday': str(game['gameday']),
                'result': f"{self.TEAM_FULL_NAMES.get(winner, winner)} won",
                'score': f"{game['away_team']} {int(game['away_score'])} - {game['home_team']} {int(game['home_score'])}"
            }
        
        # Get team performance metrics using player stats
        home_performance = self._get_team_performance_metrics(game['home_team'])
        away_performance = self._get_team_performance_metrics(game['away_team'])
        
        # Make prediction using betting lines + player performance data
        spread = game['spread']
        home_ml = game['home_ml']
        away_ml = game.get('away_ml', -110)
        
        # Calculate implied probability from moneyline (more reliable than spread)
        if home_ml < 0:
            impl_home_prob = abs(home_ml) / (abs(home_ml) + 100)
        else:
            impl_home_prob = 100 / (home_ml + 100)
        
        if away_ml < 0:
            impl_away_prob = abs(away_ml) / (abs(away_ml) + 100)
        else:
            impl_away_prob = 100 / (away_ml + 100)
        
        # Adjust probabilities based on recent team performance
        # Performance score ranges from -0.15 to +0.15 (15% swing)
        home_perf_adjustment = home_performance['performance_score'] * 0.15
        away_perf_adjustment = away_performance['performance_score'] * 0.15
        
        # Apply adjustments (normalize to keep probabilities valid)
        adj_home_prob = impl_home_prob + home_perf_adjustment - away_perf_adjustment
        adj_away_prob = impl_away_prob + away_perf_adjustment - home_perf_adjustment
        
        # Normalize to ensure probabilities sum to 1
        total_prob = adj_home_prob + adj_away_prob
        if total_prob > 0:
            adj_home_prob = adj_home_prob / total_prob
            adj_away_prob = adj_away_prob / total_prob
        
        # Determine who's favored (use adjusted probabilities)
        home_favored = adj_home_prob > adj_away_prob
        
        # Confidence based on probability difference
        prob_diff = abs(adj_home_prob - adj_away_prob)
        confidence = 0.5 + (prob_diff * 0.7)  # Scale to 0.5-0.85 range
        confidence = min(confidence, 0.85)
        
        # Boost confidence if performance metrics strongly favor one team
        perf_diff = abs(home_performance['performance_score'] - away_performance['performance_score'])
        if perf_diff > 0.3:  # Significant performance gap
            confidence = min(confidence + 0.05, 0.90)
        
        # Reduce confidence if key players are injured
        if home_performance['key_injuries'] > 0:
            confidence = max(confidence - (home_performance['key_injuries'] * 0.03), 0.50)
        if away_performance['key_injuries'] > 0:
            confidence = max(confidence - (away_performance['key_injuries'] * 0.03), 0.50)
        
        # Also factor in spread magnitude for additional confidence
        if abs(spread) >= 7:
            confidence = min(confidence + 0.05, 0.90)
        
        # Determine prediction
        if home_favored:
            predicted_winner = game['home_team']
            predicted_confidence = confidence
        else:
            predicted_winner = game['away_team']
            predicted_confidence = confidence
        
        # Apply selective strategy check
        meets_criteria = abs(spread) >= 5.0 and predicted_confidence >= 0.70
        
        # Build performance summary
        performance_summary = []
        if home_performance['performance_score'] > away_performance['performance_score']:
            performance_summary.append(f"{game['home_team']} has better recent form")
        elif away_performance['performance_score'] > home_performance['performance_score']:
            performance_summary.append(f"{game['away_team']} has better recent form")
        
        if home_performance['key_injuries'] > 0:
            performance_summary.append(f"{game['home_team']} has {home_performance['key_injuries']} key player(s) injured")
        if away_performance['key_injuries'] > 0:
            performance_summary.append(f"{game['away_team']} has {away_performance['key_injuries']} key player(s) injured")
        
        return {
            'success': True,
            'completed': False,
            'home_team': game['home_team'],
            'away_team': game['away_team'],
            'home_team_full': self.TEAM_FULL_NAMES.get(game['home_team'], game['home_team']),
            'away_team_full': self.TEAM_FULL_NAMES.get(game['away_team'], game['away_team']),
            'week': game['week'],
            'gameday': str(game['gameday']),
            'prediction': predicted_winner,
            'prediction_full': self.TEAM_FULL_NAMES.get(predicted_winner, predicted_winner),
            'confidence': predicted_confidence,
            'spread': spread,
            'meets_criteria': meets_criteria,
            'strategy_note': "High confidence pick" if meets_criteria else "Lower confidence - consider skipping",
            'performance_analysis': {
                'home': home_performance,
                'away': away_performance,
                'summary': '; '.join(performance_summary) if performance_summary else 'Teams have similar recent form'
            }
        }


def quick_predict(query: str, model_dir: str = 'models/trained') -> Dict[str, Any]:
    """Convenience function for quick predictions.
    
    Args:
        query: Natural language query
        model_dir: Directory containing trained models
        
    Returns:
        Prediction results dictionary
    """
    predictor = QuickPredictor(model_dir)
    return predictor.predict(query)

