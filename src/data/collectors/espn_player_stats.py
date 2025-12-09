"""
ESPN API Player Stats Collector for 2025 NFL Season.

This module fetches player game-by-game statistics directly from the ESPN API,
which has up-to-date 2025 season data.
"""

import requests
import logging
from typing import Dict, List, Optional, Any
from functools import lru_cache
import time

logger = logging.getLogger(__name__)


class ESPNPlayerStats:
    """Fetches player statistics from ESPN API."""
    
    BASE_URL = "https://site.web.api.espn.com/apis"
    STATS_URL = f"{BASE_URL}/site/v2/sports/football/nfl/statistics"
    GAMELOG_URL = f"{BASE_URL}/common/v3/sports/football/nfl/athletes"
    SEARCH_URL = f"{BASE_URL}/common/v3/search"
    
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    }
    
    # Stat column indices from ESPN gamelog
    # ['REC', 'TGTS', 'YDS', 'AVG', 'TD', 'LNG', 'CAR', 'YDS', 'AVG', 'LNG', 'TD', 'FUM', 'LST', 'FF', 'KB']
    RECEIVING_COLS = {
        'receptions': 0,
        'targets': 1,
        'receiving_yards': 2,
        'yards_per_reception': 3,
        'receiving_touchdowns': 4,
        'long_reception': 5,
    }
    
    RUSHING_COLS = {
        'rushing_attempts': 6,
        'rushing_yards': 7,
        'yards_per_rush': 8,
        'long_rush': 9,
        'rushing_touchdowns': 10,
    }
    
    # Team abbreviation mapping (ESPN uses different abbreviations)
    TEAM_MAPPING = {
        'ARI': 'ARI', 'ATL': 'ATL', 'BAL': 'BAL', 'BUF': 'BUF',
        'CAR': 'CAR', 'CHI': 'CHI', 'CIN': 'CIN', 'CLE': 'CLE',
        'DAL': 'DAL', 'DEN': 'DEN', 'DET': 'DET', 'GB': 'GB',
        'HOU': 'HOU', 'IND': 'IND', 'JAX': 'JAC', 'JAC': 'JAC',
        'KC': 'KC', 'LAC': 'LAC', 'LAR': 'LA', 'LA': 'LA',
        'LV': 'LV', 'MIA': 'MIA', 'MIN': 'MIN', 'NE': 'NE',
        'NO': 'NO', 'NYG': 'NYG', 'NYJ': 'NYJ', 'PHI': 'PHI',
        'PIT': 'PIT', 'SEA': 'SEA', 'SF': 'SF', 'TB': 'TB',
        'TEN': 'TEN', 'WAS': 'WSH', 'WSH': 'WSH',
    }
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)
        self._player_cache: Dict[str, str] = {}  # name -> ESPN ID
    
    def _make_request(self, url: str, params: Dict = None) -> Optional[Dict]:
        """Make a request to ESPN API with error handling."""
        try:
            resp = self.session.get(url, params=params, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            logger.error(f"ESPN API request failed: {e}")
            return None
    
    @lru_cache(maxsize=500)
    def search_player(self, player_name: str, team_abbr: Optional[str] = None) -> Optional[str]:
        """
        Search for a player and return their ESPN ID.
        
        Args:
            player_name: Player name (e.g., "Ja'Marr Chase", "Devonta Smith")
            team_abbr: Optional team abbreviation to filter results
            
        Returns:
            ESPN player ID or None if not found
        """
        # Check cache first
        cache_key = f"{player_name.lower()}_{team_abbr or ''}"
        if cache_key in self._player_cache:
            return self._player_cache[cache_key]
        
        # Try the statistics endpoint to find player by name
        data = self._make_request(self.STATS_URL)
        if not data:
            return None
        
        # Normalize search name
        search_name = player_name.lower().strip()
        search_parts = search_name.split()
        
        # Search through stat categories
        stats = data.get('stats', {})
        categories = stats.get('categories', [])
        
        for cat in categories:
            leaders = cat.get('leaders', [])
            for leader in leaders:
                athlete = leader.get('athlete', {})
                display_name = athlete.get('displayName', '').lower()
                
                # Check if player name matches
                if search_name in display_name or display_name in search_name:
                    player_id = athlete.get('id')
                    if player_id:
                        self._player_cache[cache_key] = player_id
                        logger.info(f"Found player {player_name}: ESPN ID {player_id}")
                        return player_id
                
                # Check partial match (last name)
                if len(search_parts) > 0:
                    last_name = search_parts[-1]
                    if last_name in display_name.split():
                        # Verify first name initial if available
                        if len(search_parts) > 1:
                            first_initial = search_parts[0][0]
                            if display_name.startswith(first_initial):
                                player_id = athlete.get('id')
                                if player_id:
                                    self._player_cache[cache_key] = player_id
                                    logger.info(f"Found player {player_name}: ESPN ID {player_id}")
                                    return player_id
        
        logger.warning(f"Player not found: {player_name}")
        return None
    
    def get_player_gamelog(self, player_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a player's game-by-game statistics.
        
        Args:
            player_id: ESPN player ID
            
        Returns:
            Dictionary with game stats or None if not found
        """
        url = f"{self.GAMELOG_URL}/{player_id}/gamelog"
        data = self._make_request(url)
        
        if not data:
            return None
        
        # Parse the game log
        result = {
            'player_id': player_id,
            'labels': data.get('names', []),
            'display_labels': data.get('displayNames', []),
            'games': [],
        }
        
        # Get games from seasonTypes
        for season_type in data.get('seasonTypes', []):
            season_name = season_type.get('displayName', '')
            
            for category in season_type.get('categories', []):
                events = category.get('events', [])
                
                for event in events:
                    game = {
                        'season': season_name,
                        'week': event.get('week'),
                        'stats': event.get('stats', []),
                    }
                    result['games'].append(game)
        
        # Also get event details for opponent info
        events_dict = data.get('events', {})
        for game in result['games']:
            # Try to match event by looking at all events
            for event_id, event_data in events_dict.items():
                opponent = event_data.get('opponent', {})
                game['opponent'] = opponent.get('abbreviation', '')
                game['opponent_name'] = opponent.get('displayName', '')
                game['game_date'] = event_data.get('gameDate', '')
                game['game_result'] = event_data.get('gameResult', '')
                break  # Just get first for now since we don't have week in events
        
        return result
    
    def get_player_weekly_stats(
        self, 
        player_name: str, 
        stat_type: str = 'receiving_yards',
        team_abbr: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get weekly stats for a player.
        
        Args:
            player_name: Player name
            stat_type: Type of stat ('receiving_yards', 'rushing_yards', 'passing_yards')
            team_abbr: Optional team abbreviation
            
        Returns:
            Dictionary with player stats and game-by-game data
        """
        # Find player ID
        player_id = self.search_player(player_name, team_abbr)
        
        if not player_id:
            return {
                'success': False,
                'error': f'Player not found: {player_name}',
                'player_name': player_name,
            }
        
        # Get game log
        gamelog = self.get_player_gamelog(player_id)
        
        if not gamelog:
            return {
                'success': False,
                'error': f'Could not fetch game log for player: {player_name}',
                'player_name': player_name,
            }
        
        # Extract relevant stats
        games = gamelog.get('games', [])
        
        if not games:
            return {
                'success': False,
                'error': f'No games found for player: {player_name}',
                'player_name': player_name,
            }
        
        # Determine stat column
        if 'receiving' in stat_type:
            col_idx = self.RECEIVING_COLS.get('receiving_yards', 2)
        elif 'rushing' in stat_type:
            col_idx = self.RUSHING_COLS.get('rushing_yards', 7)
        elif 'passing' in stat_type:
            # For passing stats, we need different handling
            # ESPN gamelog for QBs has different columns
            col_idx = 2  # Placeholder - would need to detect position
        else:
            col_idx = 2  # Default to receiving yards
        
        # Extract yard values
        yard_values = []
        for game in games:
            stats = game.get('stats', [])
            if len(stats) > col_idx:
                try:
                    yards = float(stats[col_idx]) if stats[col_idx] not in ['--', '-', ''] else 0
                    yard_values.append(yards)
                except (ValueError, TypeError):
                    yard_values.append(0)
        
        if not yard_values:
            return {
                'success': False,
                'error': f'No {stat_type} data found for player: {player_name}',
                'player_name': player_name,
            }
        
        # Calculate statistics
        import statistics
        avg_yards = statistics.mean(yard_values)
        median_yards = statistics.median(yard_values)
        std_yards = statistics.stdev(yard_values) if len(yard_values) > 1 else 0
        
        return {
            'success': True,
            'player_name': player_name,
            'player_id': player_id,
            'stat_type': stat_type,
            'games_played': len(yard_values),
            'avg_yards': round(avg_yards, 1),
            'median_yards': round(median_yards, 1),
            'std_yards': round(std_yards, 1),
            'total_yards': sum(yard_values),
            'game_yards': yard_values,
            'last_5_games': yard_values[-5:] if len(yard_values) >= 5 else yard_values,
            'season': '2025',
        }
    
    def get_season_leaders(self, stat_category: str = 'receivingYards') -> List[Dict]:
        """
        Get season leaders for a stat category.
        
        Args:
            stat_category: One of 'receivingYards', 'rushingYards', 'passingYards', etc.
            
        Returns:
            List of player stats
        """
        data = self._make_request(self.STATS_URL)
        if not data:
            return []
        
        leaders = []
        stats = data.get('stats', {})
        categories = stats.get('categories', [])
        
        for cat in categories:
            if cat.get('name') == stat_category:
                for leader in cat.get('leaders', []):
                    athlete = leader.get('athlete', {})
                    team = leader.get('team', {})
                    leaders.append({
                        'player_name': athlete.get('displayName', ''),
                        'player_id': athlete.get('id', ''),
                        'team': team.get('abbreviation', ''),
                        'value': leader.get('value', 0),
                        'display_value': leader.get('displayValue', ''),
                    })
                break
        
        return leaders


# Singleton instance for easy access
_espn_stats = None

def get_espn_stats() -> ESPNPlayerStats:
    """Get singleton ESPN stats collector."""
    global _espn_stats
    if _espn_stats is None:
        _espn_stats = ESPNPlayerStats()
    return _espn_stats


if __name__ == "__main__":
    # Test the collector
    logging.basicConfig(level=logging.INFO)
    
    espn = ESPNPlayerStats()
    
    # Test season leaders
    print("\n=== 2025 Receiving Leaders ===")
    leaders = espn.get_season_leaders('receivingYards')
    for i, leader in enumerate(leaders[:10], 1):
        print(f"{i}. {leader['player_name']} ({leader['team']}): {leader['display_value']} yards")
    
    # Test player search and stats
    print("\n=== Ja'Marr Chase Stats ===")
    stats = espn.get_player_weekly_stats("Ja'Marr Chase", "receiving_yards")
    if stats['success']:
        print(f"Games: {stats['games_played']}")
        print(f"Avg: {stats['avg_yards']} yards")
        print(f"Median: {stats['median_yards']} yards")
        print(f"Last 5: {stats['last_5_games']}")
    else:
        print(f"Error: {stats['error']}")

