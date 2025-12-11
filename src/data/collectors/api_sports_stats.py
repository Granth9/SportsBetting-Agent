"""API-SPORTS player stats collector for NFL data."""

import requests
import os
from typing import Dict, Any, List, Optional
import numpy as np
from src.data.collectors.player_stats_base import PlayerStatsCollector
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class APISportsStats(PlayerStatsCollector):
    """Collector for NFL player stats from API-SPORTS.
    
    API-SPORTS provides free tier access (100 requests/day) to NFL player statistics.
    Register at https://api-sports.io/ to get an API key.
    """
    
    BASE_URL = "https://v1.american-football.api-sports.io"
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the API-SPORTS stats collector.
        
        Args:
            api_key: API key from API-SPORTS. If None, will try to get from environment variable
                     API_SPORTS_KEY or config file.
        """
        self.api_key = api_key or os.getenv('API_SPORTS_KEY')
        if not self.api_key:
            logger.warning("API-SPORTS API key not found. Set API_SPORTS_KEY environment variable.")
        
        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update({
                'x-rapidapi-key': self.api_key,
                'x-rapidapi-host': 'v1.american-football.api-sports.io'
            })
        
        self._current_season = None
        self._player_cache = {}  # Cache player lookups to reduce API calls
    
    def _get_current_season(self) -> int:
        """Get current NFL season year."""
        if self._current_season is None:
            # Default to 2025, can be updated based on API response
            try:
                # Try to get current season from API
                url = f"{self.BASE_URL}/seasons"
                r = self.session.get(url, timeout=10)
                if r.status_code == 200:
                    data = r.json()
                    if 'response' in data and data['response']:
                        # Get the most recent season
                        seasons = sorted([int(s) for s in data['response'] if str(s).isdigit()])
                        if seasons:
                            self._current_season = seasons[-1]
                            return self._current_season
            except Exception as e:
                logger.debug(f"Could not fetch current season from API: {e}")
            
            # Fallback to 2025
            self._current_season = 2025
        
        return self._current_season
    
    def _find_player_id(self, player_name: str) -> Optional[str]:
        """Find player ID from name using API-SPORTS search.
        
        Args:
            player_name: Player name (e.g., "Puka Nacua")
            
        Returns:
            Player ID string or None if not found
        """
        # Check cache first
        name_lower = player_name.lower().strip()
        if name_lower in self._player_cache:
            return self._player_cache[name_lower]
        
        if not self.api_key:
            return None
        
        try:
            # Search for player by name
            # Note: API-SPORTS endpoint may vary, this is a common pattern
            url = f"{self.BASE_URL}/players"
            params = {
                'search': player_name,
                'league': 'nfl'  # NFL league ID
            }
            
            r = self.session.get(url, params=params, timeout=10)
            
            if r.status_code == 200:
                data = r.json()
                
                # Handle different response formats
                players = []
                if isinstance(data, dict):
                    if 'response' in data:
                        players = data['response']
                    elif 'data' in data:
                        players = data['data']
                elif isinstance(data, list):
                    players = data
                
                # Find best match
                for player in players:
                    full_name = f"{player.get('firstname', '')} {player.get('lastname', '')}".strip()
                    if full_name.lower() == name_lower:
                        player_id = str(player.get('id', player.get('player_id')))
                        self._player_cache[name_lower] = player_id
                        return player_id
                
                # Try partial match (last name)
                name_parts = name_lower.split()
                if len(name_parts) >= 2:
                    last_name = name_parts[-1]
                    for player in players:
                        player_last = player.get('lastname', '').lower()
                        if player_last == last_name:
                            player_id = str(player.get('id', player.get('player_id')))
                            self._player_cache[name_lower] = player_id
                            return player_id
            
            elif r.status_code == 429:
                logger.warning("API-SPORTS rate limit reached")
            else:
                logger.debug(f"API-SPORTS search returned status {r.status_code}")
        
        except Exception as e:
            logger.debug(f"Error searching for player {player_name}: {e}")
        
        return None
    
    def get_player_weekly_stats(self, player_name: str, stat_type: str = 'receiving_yards') -> Dict[str, Any]:
        """Get weekly stats for a player in the current season.
        
        Args:
            player_name: Player name (e.g., "Puka Nacua")
            stat_type: Type of stat ('receiving_yards', 'rushing_yards', 'passing_yards', etc.)
            
        Returns:
            Dictionary with player stats in standardized format
        """
        if not self.api_key:
            return {
                'success': False,
                'error': 'API-SPORTS API key not configured. Set API_SPORTS_KEY environment variable.'
            }
        
        player_id = self._find_player_id(player_name)
        
        if not player_id:
            return {
                'success': False,
                'error': f"Could not find player: {player_name}"
            }
        
        # Map stat type to API-SPORTS field names
        stat_field_map = {
            'receiving_yards': 'receiving_yards',
            'rushing_yards': 'rushing_yards',
            'passing_yards': 'passing_yards',
            'total_yards': ['receiving_yards', 'rushing_yards'],
            'receiving_td': 'receiving_touchdowns',
            'rushing_td': 'rushing_touchdowns',
            'passing_td': 'passing_touchdowns',
            'anytime_td': ['receiving_touchdowns', 'rushing_touchdowns'],
        }
        
        stat_fields = stat_field_map.get(stat_type, ['receiving_yards'])
        is_td_stat = '_td' in stat_type
        is_total_yards = stat_type == 'total_yards'
        
        season = self._get_current_season()
        weekly_stats = []
        
        try:
            # Get player statistics for the season
            # API-SPORTS endpoint for player statistics
            url = f"{self.BASE_URL}/players/statistics"
            params = {
                'player': player_id,
                'season': season,
                'league': 'nfl'
            }
            
            r = self.session.get(url, params=params, timeout=10)
            
            if r.status_code == 429:
                return {
                    'success': False,
                    'error': 'API-SPORTS rate limit reached (100 requests/day on free tier)'
                }
            
            if r.status_code != 200:
                return {
                    'success': False,
                    'error': f"API-SPORTS returned status {r.status_code}"
                }
            
            data = r.json()
            
            # Parse response - API-SPORTS format may vary
            games = []
            if isinstance(data, dict):
                if 'response' in data:
                    games = data['response']
                elif 'data' in data:
                    games = data['data']
            elif isinstance(data, list):
                games = data
            
            # Extract weekly stats
            for game in games:
                week = game.get('week', game.get('game', {}).get('week'))
                if not week:
                    continue
                
                stats = game.get('statistics', game.get('stats', {}))
                
                if is_td_stat:
                    # For touchdowns
                    if isinstance(stat_fields, list):
                        value = sum(int(stats.get(field, 0) or 0) for field in stat_fields)
                    else:
                        value = int(stats.get(stat_fields, 0) or 0)
                    
                    weekly_stats.append({
                        'week': int(week),
                        'tds': value,
                        'opponent': game.get('opponent', {}).get('name', 'UNK') if isinstance(game.get('opponent'), dict) else 'UNK',
                    })
                else:
                    # For yards
                    if is_total_yards:
                        rec_yd = float(stats.get('receiving_yards', 0) or 0)
                        rush_yd = float(stats.get('rushing_yards', 0) or 0)
                        value = rec_yd + rush_yd
                    elif isinstance(stat_fields, list):
                        value = sum(float(stats.get(field, 0) or 0) for field in stat_fields)
                    else:
                        value = float(stats.get(stat_fields, 0) or 0)
                    
                    weekly_stats.append({
                        'week': int(week),
                        'yards': value,
                        'opponent': game.get('opponent', {}).get('name', 'UNK') if isinstance(game.get('opponent'), dict) else 'UNK',
                        'rec_yd': float(stats.get('receiving_yards', 0) or 0),
                        'rush_yd': float(stats.get('rushing_yards', 0) or 0),
                    })
            
            if not weekly_stats:
                return {
                    'success': False,
                    'error': f"No {stat_type} stats found for {player_name} in {season}"
                }
            
            # Sort by week (most recent first)
            weekly_stats.sort(key=lambda x: x['week'], reverse=True)
            last_5 = weekly_stats[:5]
            
            if is_td_stat:
                # Calculate TD-specific stats
                game_tds = [w['tds'] for w in weekly_stats]
                games_with_td = sum(1 for t in game_tds if t > 0)
                total_tds = sum(game_tds)
                td_rate = games_with_td / len(game_tds) if game_tds else 0
                avg_tds = np.mean(game_tds)
                
                last_5_str = [f"W{w['week']}: {w['tds']} TD{'s' if w['tds'] != 1 else ''}" for w in last_5]
                
                return {
                    'success': True,
                    'player_name': player_name,
                    'player_id': player_id,
                    'season': season,
                    'stat_type': stat_type,
                    'is_td_stat': True,
                    'games_played': len(game_tds),
                    'total_tds': total_tds,
                    'games_with_td': games_with_td,
                    'td_rate': round(td_rate, 3),
                    'avg_tds_per_game': round(avg_tds, 2),
                    'game_tds': game_tds,
                    'last_5_games': last_5_str,
                    'weekly_breakdown': weekly_stats,
                    'source': 'API-SPORTS',
                }
            
            # Calculate yards summary stats
            game_yards = [w['yards'] for w in weekly_stats]
            avg_yards = np.mean(game_yards)
            median_yards = np.median(game_yards)
            std_yards = np.std(game_yards) if len(game_yards) > 1 else 10.0
            last_5_str = [f"W{w['week']}: {int(w['yards'])}" for w in last_5]
            
            return {
                'success': True,
                'player_name': player_name,
                'player_id': player_id,
                'season': season,
                'games_played': len(game_yards),
                'avg_yards': round(avg_yards, 1),
                'median_yards': round(median_yards, 1),
                'std_yards': round(std_yards, 1),
                'max_yards': max(game_yards),
                'min_yards': min(game_yards),
                'total_yards': sum(game_yards),
                'game_yards': game_yards,
                'last_5_games': last_5_str,
                'weekly_breakdown': weekly_stats,
                'source': 'API-SPORTS',
            }
        
        except requests.exceptions.RequestException as e:
            logger.error(f"API-SPORTS request failed: {e}")
            return {
                'success': False,
                'error': f"API-SPORTS request failed: {str(e)}"
            }
        except Exception as e:
            logger.error(f"Error fetching stats from API-SPORTS: {e}")
            return {
                'success': False,
                'error': f"Error processing API-SPORTS response: {str(e)}"
            }

