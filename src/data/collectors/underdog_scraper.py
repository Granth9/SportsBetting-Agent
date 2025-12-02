"""Underdog betting odds scraper."""

import requests
from bs4 import BeautifulSoup
import re
from typing import List, Optional, Dict, Any
from datetime import datetime

from src.utils.data_types import BettingOption, BetType
from src.utils.logger import setup_logger


logger = setup_logger(__name__)


class UnderdogScraper:
    """Scraper for Underdog betting odds."""
    
    BASE_URL = "https://underdogfantasy.com"
    API_BASE = "https://api.underdogfantasy.com"
    
    def __init__(self):
        """Initialize the Underdog scraper."""
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
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            game_date: Date of the game (optional)
            limit: Maximum number of bets to return
            
        Returns:
            List of BettingOption objects sorted by popularity
        """
        logger.info(f"Fetching Underdog bets for {away_team} @ {home_team}")
        
        try:
            # Underdog focuses on player props and fantasy-style bets
            game_id = self._find_game_id(home_team, away_team, game_date)
            
            if not game_id:
                logger.warning(f"Could not find game {away_team} @ {home_team} on Underdog")
                return []
            
            bets = self._get_bets_for_game(game_id, limit)
            
            logger.info(f"Found {len(bets)} betting options on Underdog")
            return bets
            
        except Exception as e:
            logger.error(f"Error scraping Underdog: {e}", exc_info=True)
            return []
    
    def _find_game_id(self, home_team: str, away_team: str, game_date: Optional[datetime]) -> Optional[str]:
        """Find the Underdog game ID."""
        try:
            # Underdog API endpoint
            url = f"{self.API_BASE}/v1/nfl/games"
            
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                games = response.json()
                
                for game in games:
                    if (game.get('home_team') == home_team and 
                        game.get('away_team') == away_team):
                        return game.get('game_id')
        
        except Exception as e:
            logger.debug(f"Error finding game ID: {e}")
        
        return self._find_game_id_web(home_team, away_team)
    
    def _find_game_id_web(self, home_team: str, away_team: str) -> Optional[str]:
        """Find game ID by scraping web interface."""
        try:
            url = f"{self.BASE_URL}/nfl"
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                # Parse game listings
                # Implementation depends on Underdog's HTML structure
                pass
        
        except Exception as e:
            logger.debug(f"Error in web scraping: {e}")
        
        return None
    
    def _get_bets_for_game(self, game_id: str, limit: int) -> List[BettingOption]:
        """Get betting options for a specific game."""
        bets = []
        
        try:
            # Underdog API endpoint
            url = f"{self.API_BASE}/v1/nfl/games/{game_id}/props"
            
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                
                # Underdog typically has player props
                for idx, prop_data in enumerate(data.get('props', [])[:limit]):
                    option = self._parse_bet_option(prop_data, idx + 1)
                    if option:
                        bets.append(option)
            
        except Exception as e:
            logger.debug(f"Error fetching bets: {e}")
            bets = self._scrape_bets_web(game_id, limit)
        
        return bets
    
    def _parse_bet_option(self, prop_data: Dict[str, Any], rank: int) -> Optional[BettingOption]:
        """Parse a bet option from API response."""
        try:
            title = prop_data.get('title', '')
            description = prop_data.get('description', title)
            odds = self._parse_odds(prop_data.get('odds', '-110'))
            
            bet_type = self._classify_bet_type(title, description)
            
            # Underdog focuses on player props
            line_value = prop_data.get('line')
            player_name = prop_data.get('player_name')
            stat_type = prop_data.get('stat_type')
            
            return BettingOption(
                option_id=prop_data.get('id', f"underdog_{rank}"),
                title=title,
                description=description,
                bet_type=bet_type,
                odds=odds,
                line_value=line_value,
                player_name=player_name,
                stat_type=stat_type,
                source="underdog",
                popularity_rank=rank,
                metadata=prop_data
            )
        except Exception as e:
            logger.debug(f"Error parsing bet option: {e}")
            return None
    
    def _scrape_bets_web(self, game_id: str, limit: int) -> List[BettingOption]:
        """Fallback: scrape bets from web interface."""
        bets = []
        
        try:
            url = f"{self.BASE_URL}/nfl/games/{game_id}"
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find prop cards
                prop_containers = soup.find_all('div', class_=re.compile('prop|pick', re.I))[:limit]
                
                for idx, container in enumerate(prop_containers):
                    option = self._parse_bet_from_html(container, idx + 1)
                    if option:
                        bets.append(option)
        
        except Exception as e:
            logger.error(f"Error in web scraping: {e}")
        
        return bets
    
    def _parse_bet_from_html(self, element, rank: int) -> Optional[BettingOption]:
        """Parse a bet option from HTML element."""
        try:
            title_elem = element.find('span', class_=re.compile('title|name', re.I))
            title = title_elem.text.strip() if title_elem else "Unknown Bet"
            
            odds_elem = element.find('span', class_=re.compile('odds', re.I))
            odds_str = odds_elem.text.strip() if odds_elem else "-110"
            odds = self._parse_odds(odds_str)
            
            bet_type = self._classify_bet_type(title, "")
            
            return BettingOption(
                option_id=f"underdog_web_{rank}",
                title=title,
                description=title,
                bet_type=bet_type,
                odds=odds,
                source="underdog",
                popularity_rank=rank,
                metadata={'html_element': str(element)}
            )
        except Exception as e:
            logger.debug(f"Error parsing HTML bet: {e}")
            return None
    
    def _classify_bet_type(self, title: str, description: str) -> BetType:
        """Classify the bet type."""
        text = (title + " " + description).lower()
        
        if any(word in text for word in ['spread', 'points']):
            return BetType.SPREAD
        elif any(word in text for word in ['over', 'under', 'total']):
            return BetType.TOTAL
        elif any(word in text for word in ['player', 'yards', 'touchdowns', 'receptions']):
            return BetType.PLAYER_PROP
        else:
            return BetType.GAME_OUTCOME
    
    def _parse_odds(self, odds_str: str) -> float:
        """Parse odds string to float."""
        try:
            odds_str = re.sub(r'[^\d+\-.]', '', odds_str)
            return float(odds_str)
        except:
            return -110.0

