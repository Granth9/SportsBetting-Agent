"""Sleeper betting odds scraper."""

import requests
from bs4 import BeautifulSoup
import re
from typing import List, Optional, Dict, Any
from datetime import datetime
import time

from src.utils.data_types import BettingOption, BetType
from src.utils.logger import setup_logger


logger = setup_logger(__name__)


class SleeperScraper:
    """Scraper for Sleeper betting odds."""
    
    BASE_URL = "https://sleeper.com"
    API_BASE = "https://api.sleeper.app"
    
    def __init__(self):
        """Initialize the Sleeper scraper."""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def get_game_bets(
        self,
        home_team: str,
        away_team: str,
        game_date: Optional[datetime] = None,
        limit: int = 20
    ) -> List[BettingOption]:
        """Get the most popular betting options for a specific game.
        
        Args:
            home_team: Home team abbreviation (e.g., "NYG")
            away_team: Away team abbreviation (e.g., "NE")
            game_date: Date of the game (optional)
            limit: Maximum number of bets to return (default: 20)
            
        Returns:
            List of BettingOption objects sorted by popularity
        """
        logger.info(f"Fetching Sleeper bets for {away_team} @ {home_team}")
        
        try:
            # Sleeper uses an API - try to find the game first
            game_id = self._find_game_id(home_team, away_team, game_date)
            
            if not game_id:
                logger.warning(f"Could not find game {away_team} @ {home_team} on Sleeper")
                return []
            
            # Get betting options for this game
            bets = self._get_bets_for_game(game_id, limit)
            
            logger.info(f"Found {len(bets)} betting options on Sleeper")
            return bets
            
        except Exception as e:
            logger.error(f"Error scraping Sleeper: {e}", exc_info=True)
            return []
    
    def _find_game_id(self, home_team: str, away_team: str, game_date: Optional[datetime]) -> Optional[str]:
        """Find the Sleeper game ID for a matchup.
        
        Args:
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            game_date: Game date
            
        Returns:
            Game ID or None if not found
        """
        # Sleeper API endpoint for NFL games
        # Note: This is a simplified approach - actual implementation may need to
        # search through games or use a different endpoint
        
        try:
            # Try to get current week's games
            if game_date:
                week = self._get_week_from_date(game_date)
                season = game_date.year
            else:
                # Use current date
                from datetime import datetime
                now = datetime.now()
                week = self._get_week_from_date(now)
                season = now.year
            
            # Sleeper API endpoint (this may need adjustment based on actual API)
            url = f"{self.API_BASE}/v1/nfl/games/{season}/{week}"
            
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                games = response.json()
                
                # Find matching game
                for game in games:
                    if (game.get('home_team') == home_team and 
                        game.get('away_team') == away_team):
                        return game.get('game_id')
            
        except Exception as e:
            logger.debug(f"Error finding game ID: {e}")
        
        # Fallback: try web scraping
        return self._find_game_id_web(home_team, away_team)
    
    def _find_game_id_web(self, home_team: str, away_team: str) -> Optional[str]:
        """Find game ID by scraping the web interface."""
        try:
            # Map team abbreviations to Sleeper team names
            team_map = self._get_team_map()
            home_name = team_map.get(home_team, home_team)
            away_name = team_map.get(away_team, away_team)
            
            # Search for the game on Sleeper
            search_url = f"{self.BASE_URL}/nfl/betting"
            response = self.session.get(search_url, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                # Look for game cards or links
                # This is a placeholder - actual implementation depends on Sleeper's HTML structure
                # You may need to inspect the actual page structure
                pass
        
        except Exception as e:
            logger.debug(f"Error in web scraping: {e}")
        
        return None
    
    def _get_bets_for_game(self, game_id: str, limit: int) -> List[BettingOption]:
        """Get betting options for a specific game.
        
        Args:
            game_id: Sleeper game ID
            limit: Maximum number of bets to return
            
        Returns:
            List of BettingOption objects
        """
        bets = []
        
        try:
            # Sleeper API endpoint for game bets
            # Note: This endpoint structure may need adjustment
            url = f"{self.API_BASE}/v1/nfl/games/{game_id}/bets"
            
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                
                # Parse betting options
                for idx, bet_data in enumerate(data.get('bets', [])[:limit]):
                    option = self._parse_bet_option(bet_data, idx + 1)
                    if option:
                        bets.append(option)
            
        except Exception as e:
            logger.debug(f"Error fetching bets: {e}")
            # Fallback to web scraping
            bets = self._scrape_bets_web(game_id, limit)
        
        return bets
    
    def _parse_bet_option(self, bet_data: Dict[str, Any], rank: int) -> Optional[BettingOption]:
        """Parse a bet option from API response.
        
        Args:
            bet_data: Raw bet data from API
            rank: Popularity rank
            
        Returns:
            BettingOption or None
        """
        try:
            title = bet_data.get('title', '')
            description = bet_data.get('description', title)
            odds = self._parse_odds(bet_data.get('odds', '-110'))
            
            # Determine bet type
            bet_type = self._classify_bet_type(title, description)
            
            # Extract line value, player name, etc.
            line_value = bet_data.get('line_value')
            player_name = bet_data.get('player_name')
            stat_type = bet_data.get('stat_type')
            
            return BettingOption(
                option_id=bet_data.get('id', f"sleeper_{rank}"),
                title=title,
                description=description,
                bet_type=bet_type,
                odds=odds,
                line_value=line_value,
                player_name=player_name,
                stat_type=stat_type,
                source="sleeper",
                popularity_rank=rank,
                metadata=bet_data
            )
        except Exception as e:
            logger.debug(f"Error parsing bet option: {e}")
            return None
    
    def _scrape_bets_web(self, game_id: str, limit: int) -> List[BettingOption]:
        """Fallback: scrape bets from web interface.
        
        Args:
            game_id: Game ID
            limit: Maximum number of bets
            
        Returns:
            List of BettingOption objects
        """
        bets = []
        
        try:
            url = f"{self.BASE_URL}/nfl/games/{game_id}/bets"
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find bet cards/containers
                # This structure will need to be determined by inspecting Sleeper's HTML
                bet_containers = soup.find_all('div', class_=re.compile('bet|option', re.I))[:limit]
                
                for idx, container in enumerate(bet_containers):
                    option = self._parse_bet_from_html(container, idx + 1)
                    if option:
                        bets.append(option)
        
        except Exception as e:
            logger.error(f"Error in web scraping: {e}")
        
        return bets
    
    def _parse_bet_from_html(self, element, rank: int) -> Optional[BettingOption]:
        """Parse a bet option from HTML element."""
        try:
            # Extract title, odds, etc. from HTML
            # This is a placeholder - actual implementation depends on HTML structure
            title_elem = element.find('span', class_=re.compile('title|name', re.I))
            title = title_elem.text.strip() if title_elem else "Unknown Bet"
            
            odds_elem = element.find('span', class_=re.compile('odds', re.I))
            odds_str = odds_elem.text.strip() if odds_elem else "-110"
            odds = self._parse_odds(odds_str)
            
            bet_type = self._classify_bet_type(title, "")
            
            return BettingOption(
                option_id=f"sleeper_web_{rank}",
                title=title,
                description=title,
                bet_type=bet_type,
                odds=odds,
                source="sleeper",
                popularity_rank=rank,
                metadata={'html_element': str(element)}
            )
        except Exception as e:
            logger.debug(f"Error parsing HTML bet: {e}")
            return None
    
    def _classify_bet_type(self, title: str, description: str) -> BetType:
        """Classify the bet type from title/description.
        
        Args:
            title: Bet title
            description: Bet description
            
        Returns:
            BetType enum
        """
        text = (title + " " + description).lower()
        
        if any(word in text for word in ['spread', 'points', 'favored by', 'underdog']):
            return BetType.SPREAD
        elif any(word in text for word in ['over', 'under', 'total', 'o/u']):
            return BetType.TOTAL
        elif any(word in text for word in ['player', 'yards', 'touchdowns', 'receptions', 'passing', 'rushing', 'receiving']):
            return BetType.PLAYER_PROP
        else:
            return BetType.GAME_OUTCOME
    
    def _parse_odds(self, odds_str: str) -> float:
        """Parse odds string to float.
        
        Args:
            odds_str: Odds string (e.g., "-110", "+150")
            
        Returns:
            Odds as float
        """
        try:
            # Remove any non-numeric characters except + and -
            odds_str = re.sub(r'[^\d+\-.]', '', odds_str)
            return float(odds_str)
        except:
            return -110.0  # Default
    
    def _get_week_from_date(self, date: datetime) -> int:
        """Get NFL week number from date.
        
        Args:
            date: Game date
            
        Returns:
            Week number (1-18 for regular season)
        """
        # Simplified - actual implementation should account for NFL schedule
        # Week 1 typically starts in early September
        if date.month >= 9:
            # Fall season
            week = ((date.day - 1) // 7) + 1
        else:
            # Early season (January playoffs)
            week = 18
        
        return min(max(week, 1), 18)
    
    def _get_team_map(self) -> Dict[str, str]:
        """Map team abbreviations to Sleeper team names."""
        return {
            'NYG': 'New York Giants',
            'NE': 'New England Patriots',
            'KC': 'Kansas City Chiefs',
            'BUF': 'Buffalo Bills',
            # Add more mappings as needed
        }

