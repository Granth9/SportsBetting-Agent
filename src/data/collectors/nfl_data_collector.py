"""NFL data collection from various sources."""

import nfl_data_py as nfl
import pandas as pd
from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path

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
    
    def get_schedule(self, years: List[int]) -> pd.DataFrame:
        """Get NFL schedule for specified years.
        
        Args:
            years: List of years to fetch
            
        Returns:
            DataFrame with schedule data
        """
        logger.info(f"Fetching NFL schedule for years: {years}")
        try:
            schedule = nfl.import_schedules(years)
            
            # Cache the data
            cache_path = self.cache_dir / f"schedule_{'_'.join(map(str, years))}.parquet"
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
    
    def get_rosters(self, years: List[int]) -> pd.DataFrame:
        """Get team rosters for specified years.
        
        Args:
            years: List of years to fetch
            
        Returns:
            DataFrame with roster data
        """
        logger.info(f"Fetching rosters for years: {years}")
        try:
            rosters = nfl.import_rosters(years)
            
            cache_path = self.cache_dir / f"rosters_{'_'.join(map(str, years))}.parquet"
            rosters.to_parquet(cache_path)
            
            return rosters
        except Exception as e:
            logger.error(f"Error fetching rosters: {e}")
            raise
    
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

