"""Quick prediction module for natural language game queries and player props."""

import re
import ssl
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import pandas as pd
import joblib

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# SSL fix for nfl_data_py
ssl._create_default_https_context = ssl._create_unverified_context


class PlayerPropPredictor:
    """Predictor for player prop bets (yards, touchdowns, etc.).
    
    Uses Sleeper API for 2025 season data, with fallback to nfl_data_py for historical data.
    """
    
    def __init__(self):
        """Initialize the player prop predictor."""
        self.player_stats = None
        self.loaded_years = []
        self.sleeper_collector = None
        self._use_sleeper = True  # Try Sleeper first for 2025 data
    
    def _get_sleeper_collector(self):
        """Lazy load Sleeper collector."""
        if self.sleeper_collector is None:
            try:
                from src.data.collectors.sleeper_stats import SleeperPlayerStats
                self.sleeper_collector = SleeperPlayerStats()
                logger.info("Sleeper player stats collector initialized (2025 data)")
            except ImportError as e:
                logger.warning(f"Could not import Sleeper collector: {e}")
                self._use_sleeper = False
        return self.sleeper_collector
    
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
            # "Ja'Marr Chase" with apostrophe - match first
            r"([A-Za-z]+['\'][A-Za-z]+\s+[A-Za-z]+)",
            # Names like "Ladd McConkey" with mixed case (Mc, Mac, O', etc.)
            r"([A-Z][a-z]+\s+(?:Mc|Mac|O\')?[A-Z][a-zA-Z]+)",
            # "will Devonta Smith get" -> "Devonta Smith"  
            r"(?:will|can|does)\s+([A-Z][a-z]+\s+[A-Za-z]+)\s+(?:get|have|hit)",
            # "Devonta Smith 40 yards" -> "Devonta Smith"
            r"([A-Z][a-z]+\s+[A-Za-z]+)\s+\d+",
            # Two capitalized words (name pattern) - allow mixed case in last name
            r"([A-Z][a-z]+\s+[A-Z][a-zA-Z]+)",
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
        
        # Fallback: Try to match known players from Sleeper
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
        ]
        
        for player in known_players:
            # Check if any part of the player name is in the query
            player_parts = player.lower().replace("'", "").split()
            if any(part in query_lower.replace("'", "") for part in player_parts if len(part) > 3):
                # Verify with at least another part
                matches = sum(1 for part in player_parts if part in query_lower.replace("'", ""))
                if matches >= 1:
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
    
    def get_player_history_sleeper(self, player_name: str, stat_type: str = 'receiving_yards') -> Dict[str, Any]:
        """Get player's 2025 season stats from Sleeper API.
        
        Args:
            player_name: Player display name
            stat_type: Type of stat to analyze
            
        Returns:
            Dictionary with player stats
        """
        sleeper = self._get_sleeper_collector()
        if not sleeper:
            return {'found': False, 'error': 'Sleeper collector not available'}
        
        try:
            stats = sleeper.get_player_weekly_stats(player_name, stat_type)
            
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
                    'source': 'Sleeper (2025)',
                }
            
            # Handle yards stats
            return {
                'found': True,
                'is_td_stat': False,
                'player_name': stats['player_name'],
                'position': 'WR/RB/QB',  # Sleeper has position in player data
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
                'source': 'Sleeper (2025)',
            }
        except Exception as e:
            logger.warning(f"Sleeper lookup failed for {player_name}: {e}")
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
        # Try Sleeper first for 2025 data
        if self._use_sleeper:
            sleeper_result = self.get_player_history_sleeper(player_name, stat_type)
            if sleeper_result.get('found'):
                logger.info(f"Found 2025 stats for {player_name} via Sleeper API")
                return sleeper_result
        
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
        
        # Prediction based on average vs line
        avg = history['avg_yards']
        std = history.get('std_yards', 10)
        if std == 0 or pd.isna(std):
            std = 10
        
        # Calculate z-score
        z_score = (yards_line - avg) / std
        
        # Confidence based on how far line is from average
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
            'median_yards': history['median_yards'],
            'games_analyzed': history['games_played'],
            'hit_rate': hit_rate,
            'times_over': times_over,
            'last_5_games': history['last_5_games'],
            'season': history.get('season', '2024'),
            'data_source': history.get('source', 'unknown'),
        }
    
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
        
        # Prediction: YES if TD rate > 40%, NO otherwise
        if td_rate >= 0.5:
            prediction = 'YES'
            confidence = min(0.85, 0.5 + (td_rate - 0.5) * 0.7)
        elif td_rate >= 0.35:
            prediction = 'YES'
            confidence = 0.5 + (td_rate - 0.35) * 0.5
        elif td_rate >= 0.25:
            prediction = 'NO'
            confidence = 0.5 + (0.35 - td_rate) * 0.5
        else:
            prediction = 'NO'
            confidence = min(0.85, 0.5 + (0.25 - td_rate) * 0.7)
        
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
            'confidence': confidence,
            'td_rate': td_rate,
            'games_with_td': games_with_td,
            'total_tds': total_tds,
            'avg_tds_per_game': avg_tds,
            'games_analyzed': games_played,
            'last_5_games': history['last_5_games'],
            'season': history.get('season', '2024'),
            'data_source': history.get('source', 'unknown'),
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
        'TB': 'Tampa Bay Buccaneers', 'ARI': 'Arizona Cardinals', 'LAR': 'Los Angeles Rams',
        'SF': 'San Francisco 49ers', 'SEA': 'Seattle Seahawks',
    }
    
    def __init__(self, model_dir: str = 'models/trained'):
        """Initialize the quick predictor.
        
        Args:
            model_dir: Directory containing trained models
        """
        self.model_dir = Path(model_dir)
        self.models = None
        self.scaler = None
        self.metadata = None
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
    
    def find_game(self, team1: str, team2: str) -> Optional[Dict[str, Any]]:
        """Find an upcoming or recent game between two teams.
        
        Args:
            team1: First team abbreviation
            team2: Second team abbreviation
            
        Returns:
            Game info dict or None if not found
        """
        schedule = self.get_current_schedule()
        
        # Find games between these teams
        games = schedule[
            ((schedule['home_team'] == team1) & (schedule['away_team'] == team2)) |
            ((schedule['home_team'] == team2) & (schedule['away_team'] == team1))
        ].sort_values('gameday', ascending=False)
        
        if len(games) == 0:
            return None
        
        # Get the most recent/upcoming game
        game = games.iloc[0]
        
        return {
            'home_team': game['home_team'],
            'away_team': game['away_team'],
            'week': int(game['week']),
            'gameday': game['gameday'],
            'spread': game.get('spread_line', 0) if pd.notna(game.get('spread_line')) else 0,
            'total': game.get('total_line', 45) if pd.notna(game.get('total_line')) else 45,
            'home_ml': game.get('home_moneyline', -110) if pd.notna(game.get('home_moneyline')) else -110,
            'away_ml': game.get('away_moneyline', -110) if pd.notna(game.get('away_moneyline')) else -110,
            'completed': pd.notna(game.get('home_score')),
            'home_score': game.get('home_score'),
            'away_score': game.get('away_score'),
        }
    
    def load_models(self) -> bool:
        """Load trained models from disk.
        
        Returns:
            True if models loaded successfully
        """
        if not self.model_dir.exists():
            logger.warning(f"Model directory not found: {self.model_dir}")
            return False
        
        # Load metadata
        metadata_path = self.model_dir / 'metadata.json'
        if metadata_path.exists():
            import json
            with open(metadata_path) as f:
                self.metadata = json.load(f)
        
        # Load scaler
        scaler_path = self.model_dir / 'scaler.pkl'
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
        
        # Load models
        self.models = {}
        for model_file in self.model_dir.glob('*.pkl'):
            if model_file.name in ['scaler.pkl', 'preprocessor.pkl']:
                continue
            try:
                model = joblib.load(model_file)
                self.models[model_file.stem] = model
            except Exception as e:
                logger.warning(f"Could not load {model_file}: {e}")
        
        logger.info(f"Loaded {len(self.models)} models")
        return len(self.models) > 0
    
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
        
        # Make prediction using simple model (no need for full feature engineering)
        spread = game['spread']
        home_ml = game['home_ml']
        
        # Calculate implied probability from moneyline
        if home_ml < 0:
            impl_home = abs(home_ml) / (abs(home_ml) + 100)
        else:
            impl_home = 100 / (home_ml + 100)
        
        # Simple prediction based on spread and implied probability
        # Positive spread means away team is favored
        home_favored = spread < 0
        
        # Confidence based on spread magnitude
        confidence = min(0.5 + abs(spread) / 20, 0.85)
        
        # Determine prediction
        if home_favored:
            predicted_winner = game['home_team']
            predicted_confidence = confidence
        else:
            predicted_winner = game['away_team']
            predicted_confidence = confidence
        
        # Apply selective strategy check
        meets_criteria = abs(spread) >= 5.0 and predicted_confidence >= 0.70
        
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
            'strategy_note': "High confidence pick" if meets_criteria else "Lower confidence - consider skipping"
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

