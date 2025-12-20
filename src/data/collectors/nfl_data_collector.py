"""NFL data collection from various sources."""

import nfl_data_py as nfl
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from pathlib import Path
import os

from src.utils.logger import setup_logger
from src.utils.data_types import GameInfo


logger = setup_logger(__name__)


class NFLDataCollector:
    """Collect NFL game and team data using nfl_data_py."""
    
    def __init__(self, cache_dir: str = "data/raw"):
        """Initialize the NFL data collector.
        
        Args:
            cache_dir: Directory to cache downloaded data
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_schedule(self, years: List[int], force_refresh: bool = False) -> pd.DataFrame:
        """Get NFL schedule for specified years.
        
        Args:
            years: List of years to fetch
            force_refresh: If True, force refresh even if cache exists
            
        Returns:
            DataFrame with schedule data
        """
        cache_path = self.cache_dir / f"schedule_{'_'.join(map(str, years))}.parquet"
        
        # Try to load from cache if not forcing refresh
        if not force_refresh and cache_path.exists():
            try:
                logger.info(f"Loading cached schedule from {cache_path}")
                schedule = pd.read_parquet(cache_path)
                
                # Check if current season data needs refresh
                current_year = datetime.now().year
                if current_year in years:
                    cache_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
                    # Refresh if cache is older than 1 day for current season
                    if cache_age > timedelta(days=1):
                        logger.info(f"Schedule cache for {current_year} is {cache_age.days} days old. Consider refreshing.")
                
                return schedule
            except Exception as e:
                logger.warning(f"Error loading cached schedule: {e}. Fetching fresh data.")
        
        logger.info(f"Fetching NFL schedule for years: {years}")
        try:
            schedule = nfl.import_schedules(years)
            
            # Cache the data
            schedule.to_parquet(cache_path)
            
            logger.info(f"Fetched {len(schedule)} games")
            return schedule
        except Exception as e:
            logger.error(f"Error fetching schedule: {e}")
            raise
    
    def get_team_stats(self, years: List[int], stat_type: str = 'offense') -> pd.DataFrame:
        """Get team statistics for specified years.
        
        Args:
            years: List of years to fetch
            stat_type: Type of stats ('offense', 'defense')
            
        Returns:
            DataFrame with team stats
        """
        logger.info(f"Fetching {stat_type} team stats for years: {years}")
        try:
            if stat_type == 'offense':
                stats = nfl.import_seasonal_data(years)
            else:
                # Get defensive stats from player data aggregated by team
                stats = nfl.import_seasonal_data(years)
            
            cache_path = self.cache_dir / f"team_stats_{stat_type}_{'_'.join(map(str, years))}.parquet"
            stats.to_parquet(cache_path)
            
            return stats
        except Exception as e:
            logger.error(f"Error fetching team stats: {e}")
            raise
    
    def get_player_stats(self, years: List[int], stat_type: str = 'weekly') -> pd.DataFrame:
        """Get player statistics.
        
        Args:
            years: List of years to fetch
            stat_type: 'weekly' or 'seasonal'
            
        Returns:
            DataFrame with player stats
        """
        logger.info(f"Fetching {stat_type} player stats for years: {years}")
        try:
            if stat_type == 'weekly':
                stats = nfl.import_weekly_data(years)
            else:
                stats = nfl.import_seasonal_data(years)
            
            cache_path = self.cache_dir / f"player_stats_{stat_type}_{'_'.join(map(str, years))}.parquet"
            stats.to_parquet(cache_path)
            
            logger.info(f"Fetched stats for {len(stats)} player-game records")
            return stats
        except Exception as e:
            logger.error(f"Error fetching player stats: {e}")
            raise
    
    def get_rosters(self, years: List[int], force_refresh: bool = False) -> pd.DataFrame:
        """Get team rosters for specified years.
        
        Args:
            years: List of years to fetch
            force_refresh: If True, force refresh even if cache exists
            
        Returns:
            DataFrame with roster data
        """
        # Check if we need to refresh current season data
        current_year = datetime.now().year
        if current_year in years and not force_refresh:
            if self._is_roster_data_stale(current_year):
                logger.warning(f"Roster data for {current_year} is stale. Consider refreshing.")
                force_refresh = True
        
        cache_path = self.cache_dir / f"rosters_{'_'.join(map(str, years))}.parquet"
        
        # Try to load from cache if not forcing refresh
        if not force_refresh and cache_path.exists():
            try:
                logger.info(f"Loading cached rosters from {cache_path}")
                rosters = pd.read_parquet(cache_path)
                
                # Check freshness for current season
                if current_year in years:
                    if self._is_roster_data_stale(current_year, cache_path):
                        logger.warning(f"Using stale roster data for {current_year}. Last updated: {datetime.fromtimestamp(cache_path.stat().st_mtime)}")
                
                return rosters
            except Exception as e:
                logger.warning(f"Error loading cached rosters: {e}. Fetching fresh data.")
        
        logger.info(f"Fetching rosters for years: {years}")
        try:
            rosters = nfl.import_rosters(years)
            
            rosters.to_parquet(cache_path)
            logger.info(f"Saved rosters to {cache_path}")
            
            return rosters
        except Exception as e:
            logger.error(f"Error fetching rosters: {e}")
            raise
    
    def _is_roster_data_stale(self, year: int, cache_path: Optional[Path] = None) -> bool:
        """Check if roster data is stale for the given year.
        
        For current season, data is considered stale if:
        - Cache doesn't exist, OR
        - Cache is older than 7 days (roster changes can happen weekly)
        
        Args:
            year: Year to check
            cache_path: Optional path to cache file (will construct if not provided)
            
        Returns:
            True if data is stale, False otherwise
        """
        current_year = datetime.now().year
        
        # Only check freshness for current season
        if year != current_year:
            return False
        
        if cache_path is None:
            cache_path = self.cache_dir / f"rosters_{year}.parquet"
        
        if not cache_path.exists():
            return True
        
        # Check if cache is older than 7 days
        cache_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
        stale_threshold = timedelta(days=7)
        
        return cache_age > stale_threshold
    
    def refresh_current_season_rosters(self) -> pd.DataFrame:
        """Force refresh roster data for the current season.
        
        Returns:
            DataFrame with refreshed roster data
        """
        current_year = datetime.now().year
        logger.info(f"Force refreshing roster data for {current_year}")
        return self.get_rosters([current_year], force_refresh=True)
    
    def refresh_week_betting_odds(self, season: int, week: int, force_refresh: bool = True) -> pd.DataFrame:
        """Refresh betting odds for specific week.
        
        Args:
            season: Season year
            week: Week number
            force_refresh: Force refresh even if cache exists
            
        Returns:
            Updated schedule DataFrame with current betting lines
        """
        logger.info(f"Refreshing betting odds for {season} Week {week}")
        
        # Get schedule for the season (force refresh to get latest odds)
        schedule = self.get_schedule([season], force_refresh=force_refresh)
        
        # Filter to the specific week
        week_games = schedule[schedule['week'] == week].copy()
        
        if len(week_games) == 0:
            logger.warning(f"No games found for {season} Week {week}")
            return pd.DataFrame()
        
        # Check if betting odds are present
        betting_cols = ['spread_line', 'total_line', 'home_moneyline', 'away_moneyline']
        missing_cols = [col for col in betting_cols if col not in week_games.columns]
        
        if missing_cols:
            logger.warning(f"Missing betting columns: {missing_cols}")
        
        # Check if odds are current (not all NaN)
        if 'spread_line' in week_games.columns:
            missing_spread = week_games['spread_line'].isna().sum()
            total_games = len(week_games)
            if missing_spread > 0:
                logger.warning(f"{missing_spread}/{total_games} games missing spread data")
            else:
                logger.info(f"All {total_games} games have spread data")
        
        # Check cache age to verify freshness
        cache_path = self.cache_dir / f"schedule_{season}.parquet"
        if cache_path.exists():
            cache_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
            if cache_age > timedelta(hours=6):
                logger.warning(f"Betting odds cache is {cache_age} old. Consider refreshing more frequently.")
            else:
                logger.info(f"Betting odds are fresh (cached {cache_age} ago)")
        
        return week_games
    
    def refresh_week_data(self, season: int, week: int) -> Dict[str, pd.DataFrame]:
        """Refresh all data for a specific week.
        
        Args:
            season: Season year
            week: Week number
            
        Returns:
            Dict with keys: 'rosters', 'injuries', 'schedule'
            Note: 'injuries' requires InjuryCollector to be imported separately
        """
        logger.info(f"Refreshing all data for {season} Week {week}")
        
        result = {}
        
        # Refresh rosters (if current season)
        current_year = datetime.now().year
        if season == current_year:
            result['rosters'] = self.refresh_current_season_rosters()
            logger.info(f"Refreshed rosters: {len(result['rosters'])} players")
        else:
            result['rosters'] = self.get_rosters([season], force_refresh=False)
            logger.info(f"Loaded rosters: {len(result['rosters'])} players")
        
        # Refresh betting odds
        result['schedule'] = self.refresh_week_betting_odds(season, week, force_refresh=True)
        logger.info(f"Refreshed betting odds for {len(result['schedule'])} games")
        
        # Note: Injuries need to be collected separately using InjuryCollector
        # This is done in the refresh script to avoid circular dependencies
        
        return result
    
    def get_pbp_data(self, years: List[int]) -> pd.DataFrame:
        """Get play-by-play data for detailed analysis.
        
        Args:
            years: List of years to fetch
            
        Returns:
            DataFrame with play-by-play data
        """
        logger.info(f"Fetching play-by-play data for years: {years}")
        try:
            pbp = nfl.import_pbp_data(years)
            
            cache_path = self.cache_dir / f"pbp_{'_'.join(map(str, years))}.parquet"
            pbp.to_parquet(cache_path)
            
            logger.info(f"Fetched {len(pbp)} plays")
            return pbp
        except Exception as e:
            logger.error(f"Error fetching play-by-play data: {e}")
            raise
    
    def load_cached_data(self, data_type: str, years: List[int]) -> Optional[pd.DataFrame]:
        """Load cached data if available.
        
        Args:
            data_type: Type of data (schedule, team_stats, player_stats, etc.)
            years: List of years
            
        Returns:
            DataFrame if cache exists, None otherwise
        """
        cache_path = self.cache_dir / f"{data_type}_{'_'.join(map(str, years))}.parquet"
        
        if cache_path.exists():
            logger.info(f"Loading cached data from {cache_path}")
            return pd.read_parquet(cache_path)
        
        return None
    
    def get_game_info(self, game_id: str, schedule_df: Optional[pd.DataFrame] = None) -> GameInfo:
        """Extract GameInfo object from schedule data.
        
        Args:
            game_id: NFL game ID
            schedule_df: Schedule DataFrame (will fetch if not provided)
            
        Returns:
            GameInfo object
        """
        if schedule_df is None:
            # Extract year from game_id (format: YYYY_WW_AWAY_HOME)
            year = int(game_id.split('_')[0])
            schedule_df = self.get_schedule([year])
        
        game = schedule_df[schedule_df['game_id'] == game_id].iloc[0]
        
        return GameInfo(
            game_id=game_id,
            home_team=game['home_team'],
            away_team=game['away_team'],
            game_date=pd.to_datetime(game['gameday']),
            season=int(game['season']),
            week=int(game['week']),
            home_score=int(game['home_score']) if pd.notna(game['home_score']) else None,
            away_score=int(game['away_score']) if pd.notna(game['away_score']) else None
        )

