"""Refresh Week 16 data: rosters, injuries, and betting odds."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.collectors.nfl_data_collector import NFLDataCollector
from src.data.collectors.injury_collector import InjuryCollector
from src.utils.logger import setup_logger
import pandas as pd

logger = setup_logger(__name__)


def _verify_data_quality(rosters: pd.DataFrame, injuries: pd.DataFrame, schedule: pd.DataFrame) -> None:
    """Verify data quality and log warnings.
    
    Args:
        rosters: Roster DataFrame
        injuries: Injury DataFrame
        schedule: Schedule DataFrame
    """
    logger.info("=" * 60)
    logger.info("DATA QUALITY VERIFICATION")
    logger.info("=" * 60)
    
    # Check rosters
    if len(rosters) == 0:
        logger.warning("⚠️  No roster data collected!")
    else:
        logger.info(f"✅ Rosters: {len(rosters)} players")
        if 'team' in rosters.columns:
            unique_teams = rosters['team'].nunique()
            logger.info(f"   - {unique_teams} unique teams")
    
    # Check injuries
    if len(injuries) == 0:
        logger.warning("⚠️  No injury data collected!")
    else:
        logger.info(f"✅ Injuries: {len(injuries)} injury reports")
        if 'team' in injuries.columns:
            teams_with_injuries = injuries['team'].nunique()
            logger.info(f"   - {teams_with_injuries} teams with injuries")
        if 'game_status' in injuries.columns:
            out_players = injuries[injuries['game_status'].str.contains('Out', case=False, na=False)]
            logger.info(f"   - {len(out_players)} players listed as OUT")
    
    # Check schedule/betting odds
    if len(schedule) == 0:
        logger.warning("⚠️  No schedule data for Week 16!")
    else:
        logger.info(f"✅ Schedule: {len(schedule)} Week 16 games")
        
        # Check betting odds completeness
        betting_cols = ['spread_line', 'total_line', 'home_moneyline', 'away_moneyline']
        for col in betting_cols:
            if col in schedule.columns:
                missing = schedule[col].isna().sum()
                if missing > 0:
                    logger.warning(f"   - {missing} games missing {col}")
                else:
                    logger.info(f"   - All games have {col}")
            else:
                logger.warning(f"   - Column {col} not found in schedule")
    
    logger.info("=" * 60)


def refresh_week16_data(season: int = 2025, week: int = 16, force_refresh: bool = True) -> dict:
    """Refresh all data for Week 16.
    
    Args:
        season: Season year
        week: Week number
        force_refresh: Force refresh even if cache exists
        
    Returns:
        Dict with keys: 'rosters', 'injuries', 'schedule'
    """
    logger.info("=" * 60)
    logger.info(f"REFRESHING WEEK {week} DATA FOR {season}")
    logger.info("=" * 60)
    
    result = {}
    
    # 1. Refresh rosters
    logger.info("\n1. Refreshing rosters...")
    try:
        collector = NFLDataCollector()
        if season == 2025:  # Current season
            result['rosters'] = collector.refresh_current_season_rosters()
        else:
            result['rosters'] = collector.get_rosters([season], force_refresh=force_refresh)
        logger.info(f"✅ Refreshed rosters: {len(result['rosters'])} players")
    except Exception as e:
        logger.error(f"❌ Error refreshing rosters: {e}", exc_info=True)
        result['rosters'] = pd.DataFrame()
    
    # 2. Collect injuries
    logger.info("\n2. Collecting injury reports...")
    try:
        injury_collector = InjuryCollector()
        result['injuries'] = injury_collector.get_injury_report(season, week, force_refresh=force_refresh)
        logger.info(f"✅ Collected {len(result['injuries'])} injury reports")
    except Exception as e:
        logger.error(f"❌ Error collecting injuries: {e}", exc_info=True)
        result['injuries'] = pd.DataFrame()
    
    # 3. Refresh betting odds
    logger.info("\n3. Refreshing betting odds...")
    try:
        collector = NFLDataCollector()
        result['schedule'] = collector.refresh_week_betting_odds(season, week, force_refresh=force_refresh)
        logger.info(f"✅ Updated betting odds for {len(result['schedule'])} Week {week} games")
    except Exception as e:
        logger.error(f"❌ Error refreshing betting odds: {e}", exc_info=True)
        result['schedule'] = pd.DataFrame()
    
    # 4. Verify data quality
    logger.info("\n4. Verifying data quality...")
    _verify_data_quality(result['rosters'], result['injuries'], result['schedule'])
    
    logger.info("\n" + "=" * 60)
    logger.info("DATA REFRESH COMPLETE")
    logger.info("=" * 60)
    
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Refresh Week 16 data: rosters, injuries, and betting odds")
    parser.add_argument('--season', type=int, default=2025, help='Season year (default: 2025)')
    parser.add_argument('--week', type=int, default=16, help='Week number (default: 16)')
    parser.add_argument('--no-force', action='store_true', help='Do not force refresh (use cache if available)')
    
    args = parser.parse_args()
    
    refresh_week16_data(
        season=args.season,
        week=args.week,
        force_refresh=not args.no_force
    )

