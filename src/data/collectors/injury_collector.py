"""NFL injury report collection from NFL.com."""

import requests
from bs4 import BeautifulSoup
import pandas as pd
from typing import Optional, Dict, Any
from pathlib import Path
from datetime import datetime
import time

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class InjuryCollector:
    """Collect NFL injury reports from NFL.com."""
    
    def __init__(self, cache_dir: str = "data/raw"):
        """Initialize the injury collector.
        
        Args:
            cache_dir: Directory to cache scraped data
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.base_url = "https://www.nfl.com/injuries/league"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def get_injury_report(self, season: int, week: int, force_refresh: bool = False) -> pd.DataFrame:
        """Get injury report for specific week.
        
        Args:
            season: Season year (e.g., 2025)
            week: Week number (1-18 for regular season)
            force_refresh: If True, force refresh even if cache exists
            
        Returns:
            DataFrame with columns:
            - player_name, team, position, injury_type, practice_status, game_status
        """
        cache_path = self.cache_dir / f"injuries_{season}_week{week}.parquet"
        
        # Try to load from cache if not forcing refresh
        if not force_refresh and cache_path.exists():
            try:
                cache_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
                # For current week, refresh if cache is older than 1 day
                if cache_age.days < 1:
                    logger.info(f"Loading cached injury report from {cache_path}")
                    return pd.read_parquet(cache_path)
                else:
                    logger.info(f"Injury cache is {cache_age.days} days old. Refreshing...")
            except Exception as e:
                logger.warning(f"Error loading cached injuries: {e}. Fetching fresh data.")
        
        # Construct URL
        # NFL.com uses format: /injuries/league/{season}/reg{week}
        # For 2025 week 16, it's: /injuries/league/2025/reg18 (week 16 = reg18)
        # Actually, let me check the format - it might be reg16 or week16
        url = f"{self.base_url}/{season}/reg{week}"
        
        logger.info(f"Fetching injury report from {url}")
        
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract injury data
            injuries = self._parse_injury_report(soup, season, week)
            
            if len(injuries) == 0:
                logger.warning(f"No injuries found. URL might be incorrect or page structure changed.")
                # Try alternative URL format
                url_alt = f"{self.base_url}/{season}/week{week}"
                logger.info(f"Trying alternative URL: {url_alt}")
                response = self.session.get(url_alt, timeout=10)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    injuries = self._parse_injury_report(soup, season, week)
            
            if len(injuries) > 0:
                df = pd.DataFrame(injuries)
                df.to_parquet(cache_path)
                logger.info(f"Saved {len(df)} injury records to {cache_path}")
                return df
            else:
                logger.warning("No injury data extracted. Returning empty DataFrame.")
                return pd.DataFrame(columns=['player_name', 'team', 'position', 'injury_type', 'practice_status', 'game_status'])
                
        except requests.RequestException as e:
            logger.error(f"Error fetching injury report: {e}")
            # Try to return cached data if available
            if cache_path.exists():
                logger.warning(f"Using stale cached data due to fetch error")
                return pd.read_parquet(cache_path)
            raise
        except Exception as e:
            logger.error(f"Error parsing injury report: {e}", exc_info=True)
            raise
    
    def _parse_injury_report(self, soup: BeautifulSoup, season: int, week: int) -> list:
        """Parse injury report HTML.
        
        Args:
            soup: BeautifulSoup object of the injury report page
            season: Season year
            week: Week number
            
        Returns:
            List of injury dictionaries
        """
        injuries = []
        
        # NFL.com injury reports are typically in tables or div structures
        # Look for common patterns:
        # 1. Tables with team/player/injury info
        # 2. Div structures with class names like "injury-report", "player-injury", etc.
        
        # Try to find injury tables
        tables = soup.find_all('table')
        
        for table in tables:
            # Check if this looks like an injury table
            headers = table.find_all(['th', 'thead'])
            if not headers:
                continue
            
            # Extract headers to understand structure
            header_text = ' '.join([h.get_text(strip=True) for h in headers])
            
            # Look for injury-related keywords
            if any(keyword in header_text.lower() for keyword in ['player', 'injury', 'status', 'practice']):
                rows = table.find_all('tr')
                for row in rows[1:]:  # Skip header row
                    cells = row.find_all(['td', 'th'])
                    if len(cells) < 3:
                        continue
                    
                    # Try to extract player name, team, position, injury, status
                    cell_texts = [cell.get_text(strip=True) for cell in cells]
                    
                    # Common structure: Player Name | Team | Position | Injury | Practice Status | Game Status
                    if len(cell_texts) >= 4:
                        injury = {
                            'player_name': cell_texts[0] if len(cell_texts) > 0 else '',
                            'team': cell_texts[1] if len(cell_texts) > 1 else '',
                            'position': cell_texts[2] if len(cell_texts) > 2 else '',
                            'injury_type': cell_texts[3] if len(cell_texts) > 3 else '',
                            'practice_status': cell_texts[4] if len(cell_texts) > 4 else '',
                            'game_status': cell_texts[5] if len(cell_texts) > 5 else '',
                            'season': season,
                            'week': week
                        }
                        injuries.append(injury)
        
        # If no tables found, try div-based structure
        if len(injuries) == 0:
            # Look for divs with injury-related classes
            injury_divs = soup.find_all('div', class_=lambda x: x and ('injury' in x.lower() or 'player' in x.lower()))
            
            for div in injury_divs:
                # Try to extract player info from div
                player_name = div.find(['span', 'div', 'a'], class_=lambda x: x and 'name' in x.lower() if x else False)
                if player_name:
                    injury = {
                        'player_name': player_name.get_text(strip=True),
                        'team': '',
                        'position': '',
                        'injury_type': '',
                        'practice_status': '',
                        'game_status': '',
                        'season': season,
                        'week': week
                    }
                    injuries.append(injury)
        
        # If still no data, try a more generic approach
        # Look for any structure that might contain injury data
        if len(injuries) == 0:
            # Try to find JSON data embedded in the page
            scripts = soup.find_all('script', type='application/json')
            for script in scripts:
                try:
                    import json
                    data = json.loads(script.string)
                    # Look for injury-related data in JSON
                    if isinstance(data, dict):
                        injuries = self._extract_from_json(data, season, week)
                        if len(injuries) > 0:
                            break
                except:
                    pass
        
        return injuries
    
    def _extract_from_json(self, data: Dict[str, Any], season: int, week: int) -> list:
        """Extract injury data from JSON structure.
        
        Args:
            data: JSON data dictionary
            season: Season year
            week: Week number
            
        Returns:
            List of injury dictionaries
        """
        injuries = []
        
        # Recursively search for injury-related data
        def search_dict(obj, path=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if 'injury' in key.lower() or 'player' in key.lower():
                        if isinstance(value, list):
                            for item in value:
                                if isinstance(item, dict):
                                    injury = {
                                        'player_name': item.get('name', item.get('playerName', '')),
                                        'team': item.get('team', item.get('teamAbbr', '')),
                                        'position': item.get('position', item.get('pos', '')),
                                        'injury_type': item.get('injury', item.get('injuryType', '')),
                                        'practice_status': item.get('practiceStatus', item.get('practice', '')),
                                        'game_status': item.get('gameStatus', item.get('status', '')),
                                        'season': season,
                                        'week': week
                                    }
                                    if injury['player_name']:
                                        injuries.append(injury)
                    else:
                        search_dict(value, f"{path}.{key}")
            elif isinstance(obj, list):
                for item in obj:
                    search_dict(item, path)
        
        search_dict(data)
        return injuries
    
    def get_week_16_injuries(self, season: int = 2025, force_refresh: bool = False) -> pd.DataFrame:
        """Convenience method for Week 16.
        
        Args:
            season: Season year
            force_refresh: Force refresh even if cache exists
            
        Returns:
            DataFrame with Week 16 injury reports
        """
        return self.get_injury_report(season, 16, force_refresh=force_refresh)
    
    def get_current_week_injuries(self, season: Optional[int] = None, force_refresh: bool = False) -> pd.DataFrame:
        """Get injury report for current week.
        
        Args:
            season: Season year (defaults to current year)
            force_refresh: Force refresh even if cache exists
            
        Returns:
            DataFrame with current week injury reports
        """
        if season is None:
            season = datetime.now().year
        
        # Determine current week (simplified - would need actual NFL week calculation)
        # For now, assume week 16 for December 2025
        current_date = datetime.now()
        if current_date.month == 12 and current_date.day >= 19:
            week = 16
        else:
            # Rough estimate - would need proper NFL week calculation
            week = min(18, max(1, (current_date.month - 9) * 4 + (current_date.day // 7)))
        
        logger.info(f"Fetching injuries for season {season}, week {week}")
        return self.get_injury_report(season, week, force_refresh=force_refresh)

