"""Feature engineering for NFL betting predictions."""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from src.utils.logger import setup_logger
from src.utils.data_types import GameInfo, Proposition


logger = setup_logger(__name__)


class FeatureEngineer:
    """Engineer features from raw NFL data for model training and prediction."""
    
    def __init__(self):
        """Initialize the feature engineer."""
        self.team_stats_cache: Dict[str, pd.DataFrame] = {}
        self.player_stats_cache: Dict[str, pd.DataFrame] = {}
    
    def extract_features(
        self,
        proposition: Proposition,
        schedule_df: pd.DataFrame,
        team_stats_df: pd.DataFrame,
        player_stats_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """Extract features for a betting proposition.
        
        Args:
            proposition: The betting proposition
            schedule_df: Schedule/game data
            team_stats_df: Team statistics
            player_stats_df: Player statistics (optional, needed for player props)
            
        Returns:
            Dictionary of features
        """
        features = {}
        
        game_info = proposition.game_info
        
        # Basic game features
        features.update(self._extract_basic_features(game_info))
        
        # Team performance features
        features.update(self._extract_team_features(
            game_info.home_team,
            game_info.away_team,
            team_stats_df,
            schedule_df,
            game_info.game_date
        ))
        
        # Recent form features
        features.update(self._extract_recent_form(
            game_info.home_team,
            game_info.away_team,
            schedule_df,
            game_info.game_date
        ))
        
        # Head-to-head features
        features.update(self._extract_h2h_features(
            game_info.home_team,
            game_info.away_team,
            schedule_df,
            game_info.game_date
        ))
        
        # Player-specific features (if applicable)
        if proposition.player_name and player_stats_df is not None:
            features.update(self._extract_player_features(
                proposition.player_name,
                player_stats_df,
                game_info.game_date
            ))
        
        # Betting line features
        if proposition.line:
            features.update({
                'spread': proposition.line.spread,
                'total': proposition.line.total,
                'home_ml': proposition.line.home_ml,
                'away_ml': proposition.line.away_ml,
                'implied_home_prob': self._ml_to_prob(proposition.line.home_ml),
                'implied_away_prob': self._ml_to_prob(proposition.line.away_ml)
            })
        
        return features
    
    def _extract_basic_features(self, game_info: GameInfo) -> Dict[str, Any]:
        """Extract basic game features.
        
        Args:
            game_info: Game information
            
        Returns:
            Dictionary of basic features
        """
        return {
            'season': game_info.season,
            'week': game_info.week,
            'day_of_week': game_info.game_date.weekday(),
            'month': game_info.game_date.month,
            'is_playoffs': game_info.week > 18,
            'is_divisional': self._is_divisional_game(game_info.home_team, game_info.away_team)
        }
    
    def _extract_team_features(
        self,
        home_team: str,
        away_team: str,
        team_stats_df: pd.DataFrame,
        schedule_df: pd.DataFrame,
        game_date: datetime
    ) -> Dict[str, Any]:
        """Extract team-level features.
        
        Args:
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            team_stats_df: Team statistics DataFrame
            schedule_df: Schedule DataFrame
            game_date: Date of the game
            
        Returns:
            Dictionary of team features
        """
        features = {}
        
        # Get team stats (cumulative up to this date)
        home_stats = self._get_team_stats_at_date(home_team, game_date, team_stats_df, schedule_df)
        away_stats = self._get_team_stats_at_date(away_team, game_date, team_stats_df, schedule_df)
        
        # Offensive stats
        features['home_avg_points'] = home_stats.get('avg_points', 0)
        features['away_avg_points'] = away_stats.get('avg_points', 0)
        features['home_avg_yards'] = home_stats.get('avg_yards', 0)
        features['away_avg_yards'] = away_stats.get('avg_yards', 0)
        
        # Defensive stats
        features['home_avg_points_allowed'] = home_stats.get('avg_points_allowed', 0)
        features['away_avg_points_allowed'] = away_stats.get('avg_points_allowed', 0)
        
        # Differential features
        features['point_diff_home'] = features['home_avg_points'] - features['home_avg_points_allowed']
        features['point_diff_away'] = features['away_avg_points'] - features['away_avg_points_allowed']
        
        # Home/Away splits
        features['home_home_record'] = home_stats.get('home_record', 0.5)
        features['away_away_record'] = away_stats.get('away_record', 0.5)
        
        return features
    
    def _extract_recent_form(
        self,
        home_team: str,
        away_team: str,
        schedule_df: pd.DataFrame,
        game_date: datetime,
        n_games: int = 5
    ) -> Dict[str, Any]:
        """Extract recent form features.
        
        Args:
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            schedule_df: Schedule DataFrame
            game_date: Date of the game
            n_games: Number of recent games to consider
            
        Returns:
            Dictionary of recent form features
        """
        features = {}
        
        # Get recent games for each team
        home_recent = self._get_recent_games(home_team, schedule_df, game_date, n_games)
        away_recent = self._get_recent_games(away_team, schedule_df, game_date, n_games)
        
        # Calculate win percentage
        features[f'home_last_{n_games}_win_pct'] = self._calc_win_pct(home_recent, home_team)
        features[f'away_last_{n_games}_win_pct'] = self._calc_win_pct(away_recent, away_team)
        
        # Calculate average points scored/allowed
        features[f'home_last_{n_games}_avg_scored'] = self._calc_avg_points_scored(home_recent, home_team)
        features[f'away_last_{n_games}_avg_scored'] = self._calc_avg_points_scored(away_recent, away_team)
        
        features[f'home_last_{n_games}_avg_allowed'] = self._calc_avg_points_allowed(home_recent, home_team)
        features[f'away_last_{n_games}_avg_allowed'] = self._calc_avg_points_allowed(away_recent, away_team)
        
        # Momentum indicators
        features['home_momentum'] = self._calc_momentum(home_recent, home_team)
        features['away_momentum'] = self._calc_momentum(away_recent, away_team)
        
        return features
    
    def _extract_h2h_features(
        self,
        home_team: str,
        away_team: str,
        schedule_df: pd.DataFrame,
        game_date: datetime,
        n_years: int = 3
    ) -> Dict[str, Any]:
        """Extract head-to-head features.
        
        Args:
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            schedule_df: Schedule DataFrame
            game_date: Date of the game
            n_years: Number of years of history to consider
            
        Returns:
            Dictionary of H2H features
        """
        features = {}
        
        # Get historical matchups
        cutoff_date = game_date - timedelta(days=365 * n_years)
        
        h2h_games = schedule_df[
            (pd.to_datetime(schedule_df['gameday']) >= cutoff_date) &
            (pd.to_datetime(schedule_df['gameday']) < game_date) &
            (
                ((schedule_df['home_team'] == home_team) & (schedule_df['away_team'] == away_team)) |
                ((schedule_df['home_team'] == away_team) & (schedule_df['away_team'] == home_team))
            )
        ]
        
        if len(h2h_games) > 0:
            # Count wins for each team
            home_wins = 0
            for _, game in h2h_games.iterrows():
                if game['home_team'] == home_team:
                    if game['home_score'] > game['away_score']:
                        home_wins += 1
                else:
                    if game['away_score'] > game['home_score']:
                        home_wins += 1
            
            features['h2h_home_wins'] = home_wins
            features['h2h_total_games'] = len(h2h_games)
            features['h2h_home_win_pct'] = home_wins / len(h2h_games) if len(h2h_games) > 0 else 0.5
            
            # Average scores
            features['h2h_avg_total'] = h2h_games[['home_score', 'away_score']].sum(axis=1).mean()
        else:
            features['h2h_home_wins'] = 0
            features['h2h_total_games'] = 0
            features['h2h_home_win_pct'] = 0.5
            features['h2h_avg_total'] = 0
        
        return features
    
    def _extract_player_features(
        self,
        player_name: str,
        player_stats_df: pd.DataFrame,
        game_date: datetime,
        n_games: int = 5
    ) -> Dict[str, Any]:
        """Extract player-specific features for player props.
        
        Args:
            player_name: Player name
            player_stats_df: Player statistics DataFrame
            game_date: Date of the game
            n_games: Number of recent games to consider
            
        Returns:
            Dictionary of player features
        """
        # Check cache
        cache_key = f"{player_name}_{game_date.strftime('%Y%m%d')}_{n_games}"
        if cache_key in self.player_stats_cache:
            return self.player_stats_cache[cache_key]
        
        features = {}
        
        # Get player's recent games
        player_games = player_stats_df[
            (player_stats_df['player_name'] == player_name) &
            (pd.to_datetime(player_stats_df['gameday']) < game_date)
        ].sort_values('gameday', ascending=False).head(n_games)
        
        if len(player_games) > 0:
            # Key statistics
            features['player_avg_yards'] = player_games['passing_yards'].fillna(0).mean() + \
                                          player_games['rushing_yards'].fillna(0).mean() + \
                                          player_games['receiving_yards'].fillna(0).mean()
            
            features['player_avg_tds'] = player_games['passing_tds'].fillna(0).mean() + \
                                         player_games['rushing_tds'].fillna(0).mean() + \
                                         player_games['receiving_tds'].fillna(0).mean()
            
            features['player_games_played'] = len(player_games)
        else:
            features['player_avg_yards'] = 0
            features['player_avg_tds'] = 0
            features['player_games_played'] = 0
        
        # Cache the result
        self.player_stats_cache[cache_key] = features
        return features
    
    # Helper methods
    
    def _get_team_stats_at_date(
        self,
        team: str,
        game_date: datetime,
        team_stats_df: pd.DataFrame,
        schedule_df: pd.DataFrame
    ) -> Dict[str, float]:
        """Get team statistics up to a given date."""
        # Check cache
        cache_key = f"{team}_{game_date.strftime('%Y%m%d')}"
        if cache_key in self.team_stats_cache:
            return self.team_stats_cache[cache_key]
        
        # Get all games for this team before the current date
        team_games = schedule_df[
            (pd.to_datetime(schedule_df['gameday']) < game_date) &
            ((schedule_df['home_team'] == team) | (schedule_df['away_team'] == team))
        ]
        
        if len(team_games) == 0:
            stats = {}
            self.team_stats_cache[cache_key] = stats
            return stats
        
        # Calculate aggregate stats
        stats = {}
        
        # Points scored and allowed
        points_scored = []
        points_allowed = []
        home_games = []
        
        for _, game in team_games.iterrows():
            if game['home_team'] == team:
                points_scored.append(game['home_score'])
                points_allowed.append(game['away_score'])
                home_games.append(1)
            else:
                points_scored.append(game['away_score'])
                points_allowed.append(game['home_score'])
                home_games.append(0)
        
        stats['avg_points'] = np.mean(points_scored) if points_scored else 0
        stats['avg_points_allowed'] = np.mean(points_allowed) if points_allowed else 0
        stats['avg_yards'] = 350  # Placeholder - would need play-by-play data
        
        # Home/away records
        home_record_games = [i for i, h in enumerate(home_games) if h == 1]
        if home_record_games:
            home_wins = sum(1 for i in home_record_games if points_scored[i] > points_allowed[i])
            stats['home_record'] = home_wins / len(home_record_games)
        else:
            stats['home_record'] = 0.5
        
        away_record_games = [i for i, h in enumerate(home_games) if h == 0]
        if away_record_games:
            away_wins = sum(1 for i in away_record_games if points_scored[i] > points_allowed[i])
            stats['away_record'] = away_wins / len(away_record_games)
        else:
            stats['away_record'] = 0.5
        
        # Cache the result
        self.team_stats_cache[cache_key] = stats
        return stats
    
    def _get_recent_games(
        self,
        team: str,
        schedule_df: pd.DataFrame,
        game_date: datetime,
        n_games: int
    ) -> pd.DataFrame:
        """Get recent games for a team."""
        team_games = schedule_df[
            (pd.to_datetime(schedule_df['gameday']) < game_date) &
            ((schedule_df['home_team'] == team) | (schedule_df['away_team'] == team))
        ].sort_values('gameday', ascending=False).head(n_games)
        
        return team_games
    
    def _calc_win_pct(self, games_df: pd.DataFrame, team: str) -> float:
        """Calculate win percentage from recent games."""
        if len(games_df) == 0:
            return 0.5
        
        wins = 0
        for _, game in games_df.iterrows():
            if game['home_team'] == team:
                if game['home_score'] > game['away_score']:
                    wins += 1
            else:
                if game['away_score'] > game['home_score']:
                    wins += 1
        
        return wins / len(games_df)
    
    def _calc_avg_points_scored(self, games_df: pd.DataFrame, team: str) -> float:
        """Calculate average points scored."""
        if len(games_df) == 0:
            return 0
        
        points = []
        for _, game in games_df.iterrows():
            if game['home_team'] == team:
                points.append(game['home_score'])
            else:
                points.append(game['away_score'])
        
        return np.mean(points)
    
    def _calc_avg_points_allowed(self, games_df: pd.DataFrame, team: str) -> float:
        """Calculate average points allowed."""
        if len(games_df) == 0:
            return 0
        
        points = []
        for _, game in games_df.iterrows():
            if game['home_team'] == team:
                points.append(game['away_score'])
            else:
                points.append(game['home_score'])
        
        return np.mean(points)
    
    def _calc_momentum(self, games_df: pd.DataFrame, team: str) -> float:
        """Calculate momentum score (weighted recent wins)."""
        if len(games_df) == 0:
            return 0
        
        weights = np.linspace(1, 0.5, len(games_df))  # More recent games weighted higher
        wins = []
        
        for _, game in games_df.iterrows():
            if game['home_team'] == team:
                wins.append(1 if game['home_score'] > game['away_score'] else 0)
            else:
                wins.append(1 if game['away_score'] > game['home_score'] else 0)
        
        return np.average(wins, weights=weights)
    
    def _is_divisional_game(self, team1: str, team2: str) -> bool:
        """Check if teams are in the same division."""
        divisions = {
            'AFC East': ['BUF', 'MIA', 'NE', 'NYJ'],
            'AFC North': ['BAL', 'CIN', 'CLE', 'PIT'],
            'AFC South': ['HOU', 'IND', 'JAX', 'TEN'],
            'AFC West': ['DEN', 'KC', 'LV', 'LAC'],
            'NFC East': ['DAL', 'NYG', 'PHI', 'WAS'],
            'NFC North': ['CHI', 'DET', 'GB', 'MIN'],
            'NFC South': ['ATL', 'CAR', 'NO', 'TB'],
            'NFC West': ['ARI', 'LAR', 'SF', 'SEA']
        }
        
        for division, teams in divisions.items():
            if team1 in teams and team2 in teams:
                return True
        
        return False
    
    def _ml_to_prob(self, moneyline: float) -> float:
        """Convert moneyline odds to implied probability."""
        if moneyline > 0:
            return 100 / (moneyline + 100)
        else:
            return abs(moneyline) / (abs(moneyline) + 100)

