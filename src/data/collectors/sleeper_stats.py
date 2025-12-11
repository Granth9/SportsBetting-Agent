"""Sleeper API player stats collector for 2025 NFL season data."""

import requests
from typing import Dict, Any, List, Optional
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class SleeperPlayerStats:
    """Collector for NFL player stats from Sleeper API.
    
    Sleeper provides free, public API access to NFL player stats
    including weekly stats for the current 2025 season.
    """
    
    BASE_URL = "https://api.sleeper.app"
    
    # Mapping of player names to Sleeper player IDs (top players)
    PLAYER_IDS = {
        # Wide Receivers
        "devonta smith": "7525",
        "ja'marr chase": "7564",
        "ceedee lamb": "6786",
        "tyreek hill": "4981",
        "justin jefferson": "6794",
        "amon-ra st. brown": "7588",
        "a.j. brown": "5859",
        "davante adams": "2133",
        "mike evans": "1433",
        "cooper kupp": "4039",
        "stefon diggs": "2449",
        "chris olave": "8155",
        "garrett wilson": "8150",
        "nico collins": "7561",
        "puka nacua": "9509",
        "deebo samuel": "5857",
        "jaylen waddle": "7547",
        "dj moore": "5347",
        "tee higgins": "6770",
        "malik nabers": "11632",
        "brian thomas jr": "11631",
        "brian thomas": "11631",
        "marvin harrison jr": "11628",
        "marvin harrison": "11628",
        "ladd mcconkey": "11635",
        
        # Running Backs
        "r.j. harvey": "12489",
        "rj harvey": "12489",
        "omarion hampton": "12507",
        "saquon barkley": "4866",
        "derrick henry": "3198",
        "josh jacobs": "5872",
        "jonathan taylor": "6813",
        "bijan robinson": "9221",
        "jahmyr gibbs": "9220",
        "breece hall": "8154",
        "alvin kamara": "3242",
        "nick chubb": "4988",
        "austin ekeler": "4034",
        "isaiah pacheco": "8159",
        "joe mixon": "3199",
        "rachaad white": "8153",
        "travis etienne": "7543",
        "de'von achane": "9491",
        
        # Quarterbacks
        "patrick mahomes": "4046",
        "josh allen": "4881",
        "lamar jackson": "4881",
        "jalen hurts": "6904",
        "dak prescott": "2331",
        "joe burrow": "6770",
        "cj stroud": "9493",
        "tua tagovailoa": "6768",
        "jordan love": "6797",
        "anthony richardson": "9222",
        "brock purdy": "8162",
        "caleb williams": "11560",
        "jayden daniels": "11566",
        
        # Tight Ends
        "travis kelce": "1466",
        "george kittle": "4215",
        "mark andrews": "4943",
        "t.j. hockenson": "5844",
        "sam laporta": "9497",
        "dallas goedert": "4993",
        "trey mcbride": "8136",
        "evan engram": "3286",
        "brock bowers": "11604",
    }
    
    def __init__(self):
        """Initialize the Sleeper stats collector."""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'
        })
        self._all_players = None
        self._current_season = None
    
    def _get_nfl_state(self) -> Dict[str, Any]:
        """Get current NFL state (season, week)."""
        try:
            url = f"{self.BASE_URL}/v1/state/nfl"
            r = self.session.get(url, timeout=10)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            logger.error(f"Failed to get NFL state: {e}")
            return {'season': 2025, 'week': 14}
    
    def _get_current_season(self) -> int:
        """Get current NFL season year."""
        if self._current_season is None:
            state = self._get_nfl_state()
            self._current_season = state.get('season', 2025)
        return self._current_season
    
    def _load_all_players(self) -> Dict[str, Any]:
        """Load all NFL players from Sleeper."""
        if self._all_players is not None:
            return self._all_players
        
        try:
            url = f"{self.BASE_URL}/v1/players/nfl"
            r = self.session.get(url, timeout=30)
            r.raise_for_status()
            self._all_players = r.json()
            logger.info(f"Loaded {len(self._all_players)} players from Sleeper")
            return self._all_players
        except Exception as e:
            logger.error(f"Failed to load players: {e}")
            return {}
    
    def _find_player_id(self, player_name: str) -> Optional[str]:
        """Find Sleeper player ID from name.
        
        Args:
            player_name: Player name (e.g., "DeVonta Smith")
            
        Returns:
            Player ID or None if not found
        """
        # Check our known mapping first
        name_lower = player_name.lower().strip()
        if name_lower in self.PLAYER_IDS:
            return self.PLAYER_IDS[name_lower]
        
        # Try variations
        name_parts = name_lower.replace("'", "").replace(".", "").split()
        for known_name, pid in self.PLAYER_IDS.items():
            known_parts = known_name.replace("'", "").split()
            # Check if last names match and first initial matches
            if len(name_parts) >= 2 and len(known_parts) >= 2:
                if name_parts[-1] == known_parts[-1]:  # Last name match
                    if name_parts[0][0] == known_parts[0][0]:  # First initial match
                        return pid
        
        # Search all players if not in our mapping
        all_players = self._load_all_players()
        
        for player_id, player in all_players.items():
            full_name = f"{player.get('first_name', '')} {player.get('last_name', '')}".lower().strip()
            if full_name == name_lower:
                return player_id
            # Try partial match
            if name_parts[-1] in full_name and (len(name_parts) < 2 or name_parts[0] in full_name):
                return player_id
        
        return None
    
    def get_player_weekly_stats(self, player_name: str, stat_type: str = 'receiving_yards') -> Dict[str, Any]:
        """Get weekly stats for a player in the current season.
        
        Args:
            player_name: Player name (e.g., "DeVonta Smith")
            stat_type: Type of stat ('receiving_yards', 'rushing_yards', 'passing_yards')
            
        Returns:
            Dictionary with player stats
        """
        player_id = self._find_player_id(player_name)
        
        if not player_id:
            return {
                'success': False,
                'error': f"Could not find player: {player_name}"
            }
        
        # Map stat type to Sleeper field(s)
        # total_yards combines receiving + rushing, anytime_td combines rec_td + rush_td
        stat_field_map = {
            # Yards
            'receiving_yards': ['rec_yd'],
            'rushing_yards': ['rush_yd'],
            'passing_yards': ['pass_yd'],
            'total_yards': ['rec_yd', 'rush_yd'],  # Combined receiving + rushing
            # Touchdowns
            'receiving_td': ['rec_td'],
            'rushing_td': ['rush_td'],
            'passing_td': ['pass_td'],
            'anytime_td': ['rec_td', 'rush_td'],  # Any TD scored (receiving or rushing)
        }
        stat_fields = stat_field_map.get(stat_type, ['rec_yd'])
        is_td_stat = '_td' in stat_type
        
        season = self._get_current_season()
        weekly_stats = []
        
        # Get stats for weeks 1-18
        for week in range(1, 19):
            try:
                url = f"{self.BASE_URL}/stats/nfl/{season}/{week}?season_type=regular"
                r = self.session.get(url, timeout=10)
                
                if r.status_code != 200:
                    continue
                
                data = r.json()
                
                # Handle list format (newer Sleeper API)
                if isinstance(data, list):
                    for player_data in data:
                        if str(player_data.get('player_id')) == str(player_id):
                            stats = player_data.get('stats', {})
                            # Sum all stat fields (for total_yards this sums rec_yd + rush_yd)
                            value = sum(stats.get(field, 0) or 0 for field in stat_fields)
                            
                            if is_td_stat:
                                # For TDs, track count (including 0)
                                if stats.get('gp', 0) > 0:  # Player played this week
                                    weekly_stats.append({
                                        'week': week,
                                        'tds': int(value),
                                        'opponent': player_data.get('opponent', 'UNK'),
                                        'rec_td': int(stats.get('rec_td', 0) or 0),
                                        'rush_td': int(stats.get('rush_td', 0) or 0),
                                    })
                            else:
                                # For yards
                                if value and value > 0:
                                    weekly_stats.append({
                                        'week': week,
                                        'yards': float(value),
                                        'opponent': player_data.get('opponent', 'UNK'),
                                        'rec_yd': stats.get('rec_yd', 0) or 0,
                                        'rush_yd': stats.get('rush_yd', 0) or 0,
                                    })
                                elif stats.get('gp', 0) > 0:  # Played but got 0 yards
                                    weekly_stats.append({
                                        'week': week,
                                        'yards': 0.0,
                                        'opponent': player_data.get('opponent', 'UNK'),
                                        'rec_yd': 0,
                                        'rush_yd': 0,
                                    })
                            break
                # Handle dict format (older API)
                elif isinstance(data, dict) and player_id in data:
                    stats = data[player_id]
                    value = sum(stats.get(field, 0) or 0 for field in stat_fields)
                    if is_td_stat:
                        if stats.get('gp', 0) > 0:
                            weekly_stats.append({
                                'week': week,
                                'tds': int(value),
                            })
                    elif value and value > 0:
                        weekly_stats.append({
                            'week': week,
                            'yards': float(value),
                        })
            except Exception as e:
                logger.debug(f"Error fetching week {week}: {e}")
                continue
        
        if not weekly_stats:
            return {
                'success': False,
                'error': f"No {stat_type} stats found for {player_name} in {season}"
            }
        
        import numpy as np
        
        # Last 5 games (most recent first)
        last_5 = sorted(weekly_stats, key=lambda x: x['week'], reverse=True)[:5]
        
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
        }
    
    def get_player_season_totals(self, player_name: str) -> Dict[str, Any]:
        """Get season total stats for a player.
        
        Args:
            player_name: Player name
            
        Returns:
            Dictionary with season totals
        """
        player_id = self._find_player_id(player_name)
        
        if not player_id:
            return {'success': False, 'error': f"Could not find player: {player_name}"}
        
        season = self._get_current_season()
        
        try:
            url = f"{self.BASE_URL}/stats/nfl/{season}?season_type=regular"
            r = self.session.get(url, timeout=10)
            r.raise_for_status()
            data = r.json()
            
            if player_id in data:
                stats = data[player_id]
                return {
                    'success': True,
                    'player_name': player_name,
                    'season': season,
                    'receiving_yards': stats.get('rec_yd', 0),
                    'receptions': stats.get('rec', 0),
                    'receiving_tds': stats.get('rec_td', 0),
                    'rushing_yards': stats.get('rush_yd', 0),
                    'rushing_tds': stats.get('rush_td', 0),
                    'passing_yards': stats.get('pass_yd', 0),
                    'passing_tds': stats.get('pass_td', 0),
                    'games_played': stats.get('gp', 0),
                }
            else:
                return {'success': False, 'error': f"No season stats found for {player_name}"}
        except Exception as e:
            return {'success': False, 'error': str(e)}

