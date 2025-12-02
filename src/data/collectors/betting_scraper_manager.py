"""Unified interface for all betting scrapers."""

from typing import List, Optional, Dict, Any
from datetime import datetime

from src.data.collectors.sleeper_scraper import SleeperScraper
from src.data.collectors.underdog_scraper import UnderdogScraper
from src.data.collectors.espn_scraper import ESPNScraper
from src.utils.data_types import BettingOption, GameInfo
from src.utils.logger import setup_logger


logger = setup_logger(__name__)


class BettingScraperManager:
    """Unified manager for all betting scrapers."""
    
    SUPPORTED_SITES = ['sleeper', 'underdog', 'espn']
    
    def __init__(self):
        """Initialize the scraper manager."""
        self.scrapers = {
            'sleeper': SleeperScraper(),
            'underdog': UnderdogScraper(),
            'espn': ESPNScraper()
        }
    
    def get_bets_for_game(
        self,
        site: str,
        home_team: str,
        away_team: str,
        game_date: Optional[datetime] = None,
        limit: int = 20
    ) -> List[BettingOption]:
        """Get betting options from a specific site.
        
        Args:
            site: Site name ('sleeper', 'underdog', or 'espn')
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            game_date: Date of the game (optional)
            limit: Maximum number of bets to return
            
        Returns:
            List of BettingOption objects
        """
        site = site.lower()
        
        if site not in self.SUPPORTED_SITES:
            logger.error(f"Unsupported site: {site}. Supported: {self.SUPPORTED_SITES}")
            return []
        
        scraper = self.scrapers[site]
        
        try:
            bets = scraper.get_game_bets(home_team, away_team, game_date, limit)
            return bets
        except Exception as e:
            logger.error(f"Error getting bets from {site}: {e}", exc_info=True)
            return []
    
    def get_bets_all_sites(
        self,
        home_team: str,
        away_team: str,
        game_date: Optional[datetime] = None,
        limit_per_site: int = 20
    ) -> Dict[str, List[BettingOption]]:
        """Get betting options from all supported sites.
        
        Args:
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            game_date: Date of the game (optional)
            limit_per_site: Maximum number of bets per site
            
        Returns:
            Dictionary mapping site names to lists of BettingOption objects
        """
        all_bets = {}
        
        for site in self.SUPPORTED_SITES:
            logger.info(f"Fetching bets from {site}...")
            bets = self.get_bets_for_game(site, home_team, away_team, game_date, limit_per_site)
            all_bets[site] = bets
        
        return all_bets
    
    def search_bets(
        self,
        site: str,
        home_team: str,
        away_team: str,
        search_term: Optional[str] = None,
        bet_type: Optional[str] = None,
        game_date: Optional[datetime] = None,
        limit: int = 20
    ) -> List[BettingOption]:
        """Search for specific betting options.
        
        Args:
            site: Site name
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            search_term: Search term to filter bets (e.g., player name)
            bet_type: Filter by bet type ('spread', 'total', 'player_prop', 'game_outcome')
            game_date: Date of the game
            limit: Maximum number of results
            
        Returns:
            Filtered list of BettingOption objects
        """
        bets = self.get_bets_for_game(site, home_team, away_team, game_date, limit * 2)
        
        # Apply filters
        filtered = bets
        
        if search_term:
            search_lower = search_term.lower()
            filtered = [
                bet for bet in filtered
                if search_lower in bet.title.lower() or 
                   search_lower in bet.description.lower() or
                   (bet.player_name and search_lower in bet.player_name.lower())
            ]
        
        if bet_type:
            from src.utils.data_types import BetType
            bet_type_map = {
                'spread': BetType.SPREAD,
                'total': BetType.TOTAL,
                'player_prop': BetType.PLAYER_PROP,
                'game_outcome': BetType.GAME_OUTCOME
            }
            target_type = bet_type_map.get(bet_type.lower())
            if target_type:
                filtered = [bet for bet in filtered if bet.bet_type == target_type]
        
        return filtered[:limit]
    
    def format_bets_for_display(self, bets: List[BettingOption]) -> str:
        """Format betting options for display.
        
        Args:
            bets: List of BettingOption objects
            
        Returns:
            Formatted string for display
        """
        if not bets:
            return "No betting options found."
        
        lines = []
        lines.append(f"\n{'='*80}")
        lines.append(f"Found {len(bets)} betting options:")
        lines.append(f"{'='*80}\n")
        
        for bet in bets:
            rank_str = f"[#{bet.popularity_rank}]" if bet.popularity_rank else ""
            odds_str = f"{bet.odds:+.0f}" if bet.odds else "N/A"
            
            lines.append(f"{rank_str} {bet.title}")
            lines.append(f"    Type: {bet.bet_type.value} | Odds: {odds_str} | Source: {bet.source}")
            
            if bet.player_name:
                lines.append(f"    Player: {bet.player_name}")
            if bet.line_value is not None:
                lines.append(f"    Line: {bet.line_value}")
            
            lines.append(f"    Description: {bet.description[:100]}...")
            lines.append("")
        
        return "\n".join(lines)

