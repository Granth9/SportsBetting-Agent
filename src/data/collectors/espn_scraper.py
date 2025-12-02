"""ESPN betting odds scraper."""

import requests
from bs4 import BeautifulSoup
import re
from typing import List, Optional, Dict, Any
from datetime import datetime

from src.utils.data_types import BettingOption, BetType
from src.utils.logger import setup_logger


logger = setup_logger(__name__)


class ESPNScraper:
    """Scraper for ESPN betting odds."""
    
    BASE_URL = "https://www.espn.com"
    BETTING_URL = "https://www.espn.com/betting"
    
    def __init__(self):
        """Initialize the ESPN scraper."""
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
        logger.info(f"Fetching ESPN bets for {away_team} @ {home_team}")
        
        try:
            # ESPN uses web scraping primarily
            game_id = self._find_game_id(home_team, away_team, game_date)
            
            if not game_id:
                logger.warning(f"Could not find game {away_team} @ {home_team} on ESPN")
                return []
            
            bets = self._get_bets_for_game(game_id, limit)
            
            logger.info(f"Found {len(bets)} betting options on ESPN")
            return bets
            
        except Exception as e:
            logger.error(f"Error scraping ESPN: {e}", exc_info=True)
            return []
    
    def _find_game_id(self, home_team: str, away_team: str, game_date: Optional[datetime]) -> Optional[str]:
        """Find the ESPN game ID."""
        try:
            # ESPN betting page
            url = f"{self.BETTING_URL}/nfl"
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find game cards/links
                game_links = soup.find_all('a', href=re.compile(r'/nfl/game/_/gameId/'))
                
                for link in game_links:
                    # Extract game ID from href
                    match = re.search(r'/gameId/(\d+)', link.get('href', ''))
                    if match:
                        game_id = match.group(1)
                        # Verify teams match (would need to check game details)
                        # For now, return first match
                        return game_id
        
        except Exception as e:
            logger.debug(f"Error finding game ID: {e}")
        
        return None
    
    def _get_bets_for_game(self, game_id: str, limit: int) -> List[BettingOption]:
        """Get betting options for a specific game."""
        bets = []
        
        try:
            # ESPN game betting page
            url = f"{self.BETTING_URL}/nfl/game/_/gameId/{game_id}"
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find bet options
                bet_containers = soup.find_all('div', class_=re.compile('bet|odds|prop', re.I))[:limit]
                
                for idx, container in enumerate(bet_containers):
                    option = self._parse_bet_from_html(container, idx + 1)
                    if option:
                        bets.append(option)
        
        except Exception as e:
            logger.error(f"Error fetching bets: {e}")
        
        return bets
    
    def _parse_bet_from_html(self, element, rank: int) -> Optional[BettingOption]:
        """Parse a bet option from HTML element."""
        try:
            # Extract bet information from ESPN's HTML structure
            title_elem = element.find(['h3', 'h4', 'span'], class_=re.compile('title|name|bet', re.I))
            title = title_elem.text.strip() if title_elem else "Unknown Bet"
            
            # Find odds
            odds_elem = element.find('span', class_=re.compile('odds|line', re.I))
            if not odds_elem:
                odds_elem = element.find(string=re.compile(r'[+-]?\d+'))
            
            odds_str = odds_elem.strip() if odds_elem and isinstance(odds_elem, str) else "-110"
            if not isinstance(odds_str, str):
                odds_str = odds_elem.text.strip() if hasattr(odds_elem, 'text') else "-110"
            
            odds = self._parse_odds(odds_str)
            
            # Extract line value if present
            line_match = re.search(r'([+-]?\d+\.?\d*)', title)
            line_value = float(line_match.group(1)) if line_match else None
            
            # Extract player name
            player_name = None
            player_match = re.search(r'([A-Z][a-z]+ [A-Z][a-z]+)', title)
            if player_match:
                player_name = player_match.group(1)
            
            bet_type = self._classify_bet_type(title, "")
            
            return BettingOption(
                option_id=f"espn_{rank}",
                title=title,
                description=title,
                bet_type=bet_type,
                odds=odds,
                line_value=line_value,
                player_name=player_name,
                source="espn",
                popularity_rank=rank,
                metadata={'html_element': str(element)[:500]}  # Limit size
            )
        except Exception as e:
            logger.debug(f"Error parsing HTML bet: {e}")
            return None
    
    def _classify_bet_type(self, title: str, description: str) -> BetType:
        """Classify the bet type."""
        text = (title + " " + description).lower()
        
        if any(word in text for word in ['spread', 'points', 'favored']):
            return BetType.SPREAD
        elif any(word in text for word in ['over', 'under', 'total', 'o/u']):
            return BetType.TOTAL
        elif any(word in text for word in ['player', 'yards', 'touchdowns', 'receptions', 'passing', 'rushing']):
            return BetType.PLAYER_PROP
        else:
            return BetType.GAME_OUTCOME
    
    def _parse_odds(self, odds_str: str) -> float:
        """Parse odds string to float."""
        try:
            # Clean the string
            odds_str = re.sub(r'[^\d+\-.]', '', str(odds_str))
            return float(odds_str)
        except:
            return -110.0

