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
        self.epa_cache: Dict[str, Dict[str, float]] = {}  # Cache for EPA calculations
    
    def extract_features(
        self,
        proposition: Proposition,
        schedule_df: pd.DataFrame,
        team_stats_df: pd.DataFrame,
        player_stats_df: Optional[pd.DataFrame] = None,
        weekly_stats_df: Optional[pd.DataFrame] = None,
        roster_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """Extract features for a betting proposition.
        
        Args:
            proposition: The betting proposition
            schedule_df: Schedule/game data
            team_stats_df: Team statistics
            player_stats_df: Player statistics (optional, needed for player props)
            weekly_stats_df: Weekly player stats (optional, for EPA features)
            roster_df: Weekly roster data (optional, for IR return features)
            
        Returns:
            Dictionary of features
        """
        features = {}
        
        game_info = proposition.game_info
        
        # Basic game features
        features.update(self._extract_basic_features(game_info))
        
        # Weather and rest features
        features.update(self._extract_weather_rest_features(
            game_info,
            schedule_df
        ))
        
        # Team performance features
        features.update(self._extract_team_features(
            game_info.home_team,
            game_info.away_team,
            team_stats_df,
            schedule_df,
            game_info.game_date
        ))
        
        # EPA (Expected Points Added) metrics with caching
        if weekly_stats_df is not None:
            try:
                epa_features = self._extract_epa_features(
                    game_info.home_team,
                    game_info.away_team,
                    weekly_stats_df,
                    schedule_df,
                    game_info.game_date
                )
                features.update(epa_features)
            except Exception as e:
                # If EPA extraction fails, use default values
                logger.debug(f"EPA extraction failed for {game_info.game_id}: {e}")
                features.update({
                    'home_off_epa': 0.0,
                    'away_off_epa': 0.0,
                    'home_def_epa': 0.0,
                    'away_def_epa': 0.0,
                    'home_total_epa': 0.0,
                    'away_total_epa': 0.0,
                    'epa_diff': 0.0,
                    'off_epa_diff': 0.0,
                    'def_epa_diff': 0.0
                })
        else:
            # Default values if weekly stats not available
            features.update({
                'home_off_epa': 0.0,
                'away_off_epa': 0.0,
                'home_def_epa': 0.0,
                'away_def_epa': 0.0,
                'home_total_epa': 0.0,
                'away_total_epa': 0.0,
                'epa_diff': 0.0,
                'off_epa_diff': 0.0,
                'def_epa_diff': 0.0
            })
        
        # Relative matchup features
        features.update(self._extract_relative_features(features))
        
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
        
        # Playoff implications features
        features.update(self._extract_playoff_features(
            game_info.home_team,
            game_info.away_team,
            schedule_df,
            game_info.game_date,
            game_info.week
        ))
        
        # Travel/schedule features (distance, primetime, short week)
        features.update(self._extract_travel_schedule_features(
            game_info.home_team,
            game_info.away_team,
            schedule_df,
            game_info.game_date
        ))
        
        # Bye week advantage features
        features.update(self._extract_bye_features(
            game_info.home_team,
            game_info.away_team,
            schedule_df,
            game_info.game_date
        ))
        
        # Injury reserve return features
        if roster_df is not None:
            features.update(self._extract_ir_return_features(
                game_info.home_team,
                game_info.away_team,
                roster_df,
                game_info.week,
                game_info.season
            ))
        else:
            # Default values when roster data not available
            features.update({
                'home_ir_returns': 0,
                'away_ir_returns': 0,
                'home_ir_impact_score': 0.0,
                'away_ir_impact_score': 0.0,
                'ir_advantage': 0.0,
                'home_key_player_returning': 0,
                'away_key_player_returning': 0
            })
        
        # Interaction features (must be last, after all other features are collected)
        features.update(self._extract_interaction_features(features))
        
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
        
        # Additional defensive metrics from team_stats_df
        features['home_avg_yards_allowed'] = home_stats.get('avg_yards_allowed', 0)
        features['away_avg_yards_allowed'] = away_stats.get('avg_yards_allowed', 0)
        features['home_turnovers_forced'] = home_stats.get('turnovers_forced', 0)
        features['away_turnovers_forced'] = away_stats.get('turnovers_forced', 0)
        features['home_turnovers_lost'] = home_stats.get('turnovers_lost', 0)
        features['away_turnovers_lost'] = away_stats.get('turnovers_lost', 0)
        features['home_turnover_diff'] = home_stats.get('turnover_diff', 0)
        features['away_turnover_diff'] = away_stats.get('turnover_diff', 0)
        
        # Differential features
        features['point_diff_home'] = features['home_avg_points'] - features['home_avg_points_allowed']
        features['point_diff_away'] = features['away_avg_points'] - features['away_avg_points_allowed']
        features['yards_diff_home'] = features['home_avg_yards'] - features['home_avg_yards_allowed']
        features['yards_diff_away'] = features['away_avg_yards'] - features['away_avg_yards_allowed']
        features['turnover_diff_matchup'] = features['home_turnover_diff'] - features['away_turnover_diff']
        
        # Home/Away splits
        features['home_home_record'] = home_stats.get('home_record', 0.5)
        features['away_away_record'] = away_stats.get('away_record', 0.5)
        
        # =================================================================
        # RUSHING AND PASSING STYLE STATS (for matchup analysis)
        # =================================================================
        
        # Rush offense stats
        features['home_rush_yards_per_game'] = home_stats.get('rush_yards_per_game', 100.0)
        features['away_rush_yards_per_game'] = away_stats.get('rush_yards_per_game', 100.0)
        features['home_rush_epa_per_game'] = home_stats.get('rush_epa_per_game', 0.0)
        features['away_rush_epa_per_game'] = away_stats.get('rush_epa_per_game', 0.0)
        
        # Pass offense stats
        features['home_pass_yards_per_game'] = home_stats.get('pass_yards_per_game', 200.0)
        features['away_pass_yards_per_game'] = away_stats.get('pass_yards_per_game', 200.0)
        features['home_pass_epa_per_game'] = home_stats.get('pass_epa_per_game', 0.0)
        features['away_pass_epa_per_game'] = away_stats.get('pass_epa_per_game', 0.0)
        
        # Rush defense stats (yards allowed)
        features['home_rush_yards_allowed'] = home_stats.get('rush_yards_allowed', 100.0)
        features['away_rush_yards_allowed'] = away_stats.get('rush_yards_allowed', 100.0)
        features['home_rush_epa_allowed'] = home_stats.get('rush_epa_allowed', 0.0)
        features['away_rush_epa_allowed'] = away_stats.get('rush_epa_allowed', 0.0)
        
        # Pass defense stats (yards allowed)
        features['home_pass_yards_allowed'] = home_stats.get('pass_yards_allowed', 200.0)
        features['away_pass_yards_allowed'] = away_stats.get('pass_yards_allowed', 200.0)
        features['home_pass_epa_allowed'] = home_stats.get('pass_epa_allowed', 0.0)
        features['away_pass_epa_allowed'] = away_stats.get('pass_epa_allowed', 0.0)
        
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
        
        # Advanced metrics: Red zone efficiency (simplified - using points per game as proxy)
        # In a full implementation, would calculate actual red zone conversion rates
        features[f'home_redzone_efficiency'] = features[f'home_last_{n_games}_avg_scored'] / 25.0 if features[f'home_last_{n_games}_avg_scored'] > 0 else 0.5
        features[f'away_redzone_efficiency'] = features[f'away_last_{n_games}_avg_scored'] / 25.0 if features[f'away_last_{n_games}_avg_scored'] > 0 else 0.5
        
        # Third down efficiency (simplified - using yards per game as proxy)
        # In a full implementation, would calculate actual third down conversion rates
        home_yards = existing_features.get('home_avg_yards', 350) if 'existing_features' in locals() else 350
        away_yards = existing_features.get('away_avg_yards', 350) if 'existing_features' in locals() else 350
        features['home_third_down_eff'] = min(home_yards / 400.0, 1.0)  # Normalized
        features['away_third_down_eff'] = min(away_yards / 400.0, 1.0)  # Normalized
        
        # Time of possession (simplified - using point differential as proxy)
        # Teams that score more tend to have more TOP
        features['home_top_proxy'] = min((features[f'home_last_{n_games}_avg_scored'] - features[f'home_last_{n_games}_avg_allowed']) / 20.0 + 0.5, 1.0)
        features['away_top_proxy'] = min((features[f'away_last_{n_games}_avg_scored'] - features[f'away_last_{n_games}_avg_allowed']) / 20.0 + 0.5, 1.0)
        
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
        
        # =================================================================
        # SAME-SEASON REMATCH FEATURES
        # =================================================================
        # Check if these teams already played earlier this season
        current_season = game_date.year if game_date.month > 2 else game_date.year - 1
        season_start = datetime(current_season, 9, 1)  # NFL season starts in September
        
        same_season_games = schedule_df[
            (pd.to_datetime(schedule_df['gameday']) >= season_start) &
            (pd.to_datetime(schedule_df['gameday']) < game_date) &
            (
                ((schedule_df['home_team'] == home_team) & (schedule_df['away_team'] == away_team)) |
                ((schedule_df['home_team'] == away_team) & (schedule_df['away_team'] == home_team))
            ) &
            (schedule_df['home_score'].notna())
        ]
        
        features['is_rematch'] = 1.0 if len(same_season_games) > 0 else 0.0
        
        if len(same_season_games) > 0:
            last_game = same_season_games.sort_values('gameday', ascending=False).iloc[0]
            
            # Who won the first meeting this season?
            if last_game['home_team'] == home_team:
                home_won_first = 1 if last_game['home_score'] > last_game['away_score'] else 0
                first_game_margin = last_game['home_score'] - last_game['away_score']
            else:
                home_won_first = 1 if last_game['away_score'] > last_game['home_score'] else 0
                first_game_margin = last_game['away_score'] - last_game['home_score']
            
            features['rematch_home_won_first'] = home_won_first
            features['rematch_margin'] = first_game_margin
            features['rematch_total_points'] = last_game['home_score'] + last_game['away_score']
            
            # Revenge game indicator (team that lost is now playing)
            features['home_revenge_game'] = 1.0 if home_won_first == 0 else 0.0
            features['away_revenge_game'] = 1.0 if home_won_first == 1 else 0.0
        else:
            features['rematch_home_won_first'] = 0.5
            features['rematch_margin'] = 0.0
            features['rematch_total_points'] = 0.0
            features['home_revenge_game'] = 0.0
            features['away_revenge_game'] = 0.0
        
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
    
    def _extract_relative_features(self, existing_features: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relative matchup features (offense vs defense comparisons).
        
        Args:
            existing_features: Dictionary of already-extracted features
            
        Returns:
            Dictionary of relative features
        """
        features = {}
        
        # Helper to safely get feature value
        def get_feat(key, default=0.0):
            return existing_features.get(key, default)
        
        # Home offense vs Away defense
        features['home_off_vs_away_def'] = get_feat('home_avg_points') - get_feat('away_avg_points_allowed')
        features['home_off_yards_vs_away_def'] = get_feat('home_avg_yards') - get_feat('away_avg_yards_allowed')
        features['home_off_epa_vs_away_def'] = get_feat('home_off_epa') - get_feat('away_def_epa')
        
        # Away offense vs Home defense
        features['away_off_vs_home_def'] = get_feat('away_avg_points') - get_feat('home_avg_points_allowed')
        features['away_off_yards_vs_home_def'] = get_feat('away_avg_yards') - get_feat('home_avg_yards_allowed')
        features['away_off_epa_vs_home_def'] = get_feat('away_off_epa') - get_feat('home_def_epa')
        
        # Strength of schedule adjustments
        features['sos_adjusted_diff'] = get_feat('point_diff_home') - get_feat('point_diff_away')
        
        # Overall matchup strength
        features['matchup_strength_diff'] = get_feat('home_total_epa') - get_feat('away_total_epa')
        
        # =================================================================
        # STYLE MATCHUP FEATURES (Run vs Rush Def, Pass vs Pass Def)
        # =================================================================
        
        # Rush offense vs Rush defense matchup
        # Positive = home has advantage (strong run vs weak run D)
        features['home_rush_off_vs_away_rush_def'] = get_feat('home_rush_yards_per_game') - get_feat('away_rush_yards_allowed')
        features['away_rush_off_vs_home_rush_def'] = get_feat('away_rush_yards_per_game') - get_feat('home_rush_yards_allowed')
        features['rush_matchup_advantage'] = features['home_rush_off_vs_away_rush_def'] - features['away_rush_off_vs_home_rush_def']
        
        # Pass offense vs Pass defense matchup
        # Positive = home has advantage (strong passing vs weak secondary)
        features['home_pass_off_vs_away_pass_def'] = get_feat('home_pass_yards_per_game') - get_feat('away_pass_yards_allowed')
        features['away_pass_off_vs_home_pass_def'] = get_feat('away_pass_yards_per_game') - get_feat('home_pass_yards_allowed')
        features['pass_matchup_advantage'] = features['home_pass_off_vs_away_pass_def'] - features['away_pass_off_vs_home_pass_def']
        
        # EPA-based style matchups (more predictive than raw yards)
        features['home_rush_epa_vs_away_def'] = get_feat('home_rush_epa_per_game') - get_feat('away_rush_epa_allowed')
        features['away_rush_epa_vs_home_def'] = get_feat('away_rush_epa_per_game') - get_feat('home_rush_epa_allowed')
        features['home_pass_epa_vs_away_def'] = get_feat('home_pass_epa_per_game') - get_feat('away_pass_epa_allowed')
        features['away_pass_epa_vs_home_def'] = get_feat('away_pass_epa_per_game') - get_feat('home_pass_epa_allowed')
        
        # Style dominance indicators
        # Is home team run-heavy? (rush yards > pass yards)
        home_rush = get_feat('home_rush_yards_per_game')
        home_pass = get_feat('home_pass_yards_per_game')
        away_rush = get_feat('away_rush_yards_per_game')
        away_pass = get_feat('away_pass_yards_per_game')
        
        features['home_run_heavy'] = 1.0 if home_rush > home_pass * 0.6 else 0.0
        features['away_run_heavy'] = 1.0 if away_rush > away_pass * 0.6 else 0.0
        features['home_pass_heavy'] = 1.0 if home_pass > home_rush * 1.5 else 0.0
        features['away_pass_heavy'] = 1.0 if away_pass > away_rush * 1.5 else 0.0
        
        # Style clash indicator (run-heavy team vs weak run D, or pass-heavy vs weak pass D)
        features['home_style_exploit'] = (
            features['home_run_heavy'] * max(0, features['home_rush_off_vs_away_rush_def']) +
            features['home_pass_heavy'] * max(0, features['home_pass_off_vs_away_pass_def'])
        )
        features['away_style_exploit'] = (
            features['away_run_heavy'] * max(0, features['away_rush_off_vs_home_rush_def']) +
            features['away_pass_heavy'] * max(0, features['away_pass_off_vs_home_pass_def'])
        )
        features['style_exploit_advantage'] = features['home_style_exploit'] - features['away_style_exploit']
        
        return features
    
    def _extract_interaction_features(self, existing_features: Dict[str, Any]) -> Dict[str, Any]:
        """Extract interaction and polynomial features.
        
        Args:
            existing_features: Dictionary of already-extracted features
            
        Returns:
            Dictionary of interaction features
        """
        features = {}
        
        # Helper to safely get feature value
        def get_feat(key, default=0.0):
            return existing_features.get(key, default)
        
        # Spread interactions
        spread = get_feat('spread')
        features['spread_x_home_offense'] = spread * get_feat('home_avg_points')
        features['spread_x_away_offense'] = spread * get_feat('away_avg_points')
        features['spread_x_home_epa'] = spread * get_feat('home_off_epa')
        features['spread_x_rest_diff'] = spread * get_feat('rest_days_diff')
        
        # Total interactions
        total = get_feat('total')
        combined_offense = get_feat('home_avg_points') + get_feat('away_avg_points')
        features['total_x_combined_offense'] = total * combined_offense
        combined_epa = get_feat('home_off_epa') + get_feat('away_off_epa')
        features['total_x_combined_epa'] = total * combined_epa
        
        # Rest days interactions
        rest_diff = get_feat('rest_days_diff')
        features['rest_diff_x_home_advantage'] = rest_diff * get_feat('home_home_record')
        
        # EPA interactions
        features['home_off_epa_x_away_def_epa'] = get_feat('home_off_epa') * get_feat('away_def_epa')
        features['away_off_epa_x_home_def_epa'] = get_feat('away_off_epa') * get_feat('home_def_epa')
        
        # Turnover differential interactions
        to_diff = get_feat('turnover_diff_matchup')
        features['to_diff_x_point_diff'] = to_diff * get_feat('point_diff_home')
        
        # Weather interactions
        features['temp_x_wind'] = get_feat('temperature') * get_feat('wind_speed')
        features['dome_x_temp'] = get_feat('is_dome') * get_feat('temperature')
        
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
        
        # Calculate yards from opponent's perspective (for defensive stats)
        # We'll estimate yards from points (simplified approach)
        opponent_points_scored = []  # Points scored by opponents = points allowed by this team
        
        for _, game in team_games.iterrows():
            if game['home_team'] == team:
                points_scored.append(game['home_score'])
                points_allowed.append(game['away_score'])
                opponent_points_scored.append(game['away_score'])
                home_games.append(1)
            else:
                points_scored.append(game['away_score'])
                points_allowed.append(game['home_score'])
                opponent_points_scored.append(game['home_score'])
                home_games.append(0)
        
        stats['avg_points'] = np.mean(points_scored) if points_scored else 0
        stats['avg_points_allowed'] = np.mean(points_allowed) if points_allowed else 0
        
        # Estimate yards from points (rough approximation: ~15 yards per point)
        # This is a simplified approach - in production, would use actual play-by-play data
        stats['avg_yards'] = stats['avg_points'] * 15.0  # Rough estimate
        stats['avg_yards_allowed'] = stats['avg_points_allowed'] * 15.0  # Rough estimate
        
        # Calculate turnovers from weekly stats if available
        # For now, use simplified estimates based on points (teams that score more tend to have fewer turnovers)
        # In production, would calculate from actual game data
        games_played = len(team_games)
        
        # Estimate turnovers: teams that score more have fewer turnovers on average
        # Rough estimate: ~1.5 turnovers per game, adjusted by scoring
        base_turnovers = 1.5
        scoring_factor = stats['avg_points'] / 24.0  # Normalize to league average
        stats['turnovers_lost'] = base_turnovers * (1.0 - scoring_factor * 0.3)  # Better teams turn over less
        
        # Turnovers forced: estimate from points allowed (better defenses force more turnovers)
        defense_factor = (24.0 - stats['avg_points_allowed']) / 24.0  # Better defense = higher factor
        stats['turnovers_forced'] = base_turnovers * (0.7 + defense_factor * 0.6)  # Range: 0.7-1.3
        
        stats['turnover_diff'] = stats['turnovers_forced'] - stats['turnovers_lost']
        
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
        
        # =================================================================
        # RUSHING AND PASSING STATS (Style-based features)
        # =================================================================
        # These are estimated from schedule data - in production would use play-by-play
        # NFL averages: ~110 rush yards, ~230 pass yards per game
        
        # Estimate rush/pass split based on scoring pattern
        # High-scoring teams tend to pass more, lower-scoring run more
        pass_ratio = min(0.7, 0.55 + (stats['avg_points'] - 20) * 0.01)  # 55-70% pass
        rush_ratio = 1 - pass_ratio
        
        total_yards = stats['avg_yards']
        stats['rush_yards_per_game'] = total_yards * rush_ratio
        stats['pass_yards_per_game'] = total_yards * pass_ratio
        
        # EPA estimates (simplified: positive EPA = efficient)
        point_diff = stats['avg_points'] - stats['avg_points_allowed']
        stats['rush_epa_per_game'] = point_diff * 0.3 * rush_ratio  # 30% of efficiency from rush
        stats['pass_epa_per_game'] = point_diff * 0.7 * pass_ratio  # 70% of efficiency from pass
        
        # Defensive stats (yards/EPA allowed)
        total_yards_allowed = stats['avg_yards_allowed']
        stats['rush_yards_allowed'] = total_yards_allowed * 0.35  # ~35% of yards allowed are rushing
        stats['pass_yards_allowed'] = total_yards_allowed * 0.65  # ~65% of yards allowed are passing
        
        # Defensive EPA (negative = good defense)
        opp_point_diff = stats['avg_points_allowed'] - stats['avg_points']
        stats['rush_epa_allowed'] = opp_point_diff * 0.3 * 0.35
        stats['pass_epa_allowed'] = opp_point_diff * 0.7 * 0.65
        
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
    
    def _extract_playoff_features(
        self,
        home_team: str,
        away_team: str,
        schedule_df: pd.DataFrame,
        game_date: datetime,
        week: int
    ) -> Dict[str, Any]:
        """Extract playoff implications features.
        
        Args:
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            schedule_df: Schedule DataFrame
            game_date: Date of the game
            week: Week number
            
        Returns:
            Dictionary of playoff features
        """
        features = {}
        
        # Playoff race intensifies in later weeks (after week 10)
        playoff_urgency_multiplier = max(0, (week - 10) / 8)  # 0 to 1 scale, peaks at week 18
        
        # Get team records to estimate playoff contention
        def get_team_record(team):
            team_games = schedule_df[
                (pd.to_datetime(schedule_df['gameday']) < game_date) &
                ((schedule_df['home_team'] == team) | (schedule_df['away_team'] == team)) &
                (schedule_df['home_score'].notna())
            ]
            
            if len(team_games) == 0:
                return 0.5, 0
            
            wins = 0
            for _, game in team_games.iterrows():
                if game['home_team'] == team:
                    if game['home_score'] > game['away_score']:
                        wins += 1
                else:
                    if game['away_score'] > game['home_score']:
                        wins += 1
            
            win_pct = wins / len(team_games)
            return win_pct, len(team_games)
        
        home_win_pct, home_games = get_team_record(home_team)
        away_win_pct, away_games = get_team_record(away_team)
        
        # Playoff contender: win% > 0.5 and late in season
        features['home_playoff_contender'] = 1.0 if home_win_pct > 0.5 and week > 10 else 0.0
        features['away_playoff_contender'] = 1.0 if away_win_pct > 0.5 and week > 10 else 0.0
        
        # Playoff stakes difference (how much more one team needs the win)
        home_urgency = home_win_pct * playoff_urgency_multiplier if home_win_pct > 0.4 else 0
        away_urgency = away_win_pct * playoff_urgency_multiplier if away_win_pct > 0.4 else 0
        features['playoff_stakes_diff'] = home_urgency - away_urgency
        
        # Elimination game indicator (late season, close to .500)
        features['home_must_win'] = 1.0 if week > 14 and 0.4 < home_win_pct < 0.6 else 0.0
        features['away_must_win'] = 1.0 if week > 14 and 0.4 < away_win_pct < 0.6 else 0.0
        
        return features
    
    def _extract_travel_schedule_features(
        self,
        home_team: str,
        away_team: str,
        schedule_df: pd.DataFrame,
        game_date: datetime
    ) -> Dict[str, Any]:
        """Extract travel and schedule features.
        
        Args:
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            schedule_df: Schedule DataFrame
            game_date: Date of the game
            
        Returns:
            Dictionary of travel/schedule features
        """
        features = {}
        
        # Team locations (approximate time zones/regions)
        team_regions = {
            # West Coast (Pacific)
            'SEA': 'west', 'SF': 'west', 'LAR': 'west', 'LAC': 'west', 'LV': 'west', 'ARI': 'west',
            # Mountain
            'DEN': 'mountain',
            # Central
            'KC': 'central', 'DAL': 'central', 'HOU': 'central', 'NO': 'central', 'MIN': 'central',
            'GB': 'central', 'CHI': 'central', 'DET': 'central', 'IND': 'central', 'TEN': 'central',
            'JAX': 'central',
            # East Coast
            'MIA': 'east', 'TB': 'east', 'ATL': 'east', 'CAR': 'east', 'WAS': 'east', 'PHI': 'east',
            'NYG': 'east', 'NYJ': 'east', 'NE': 'east', 'BUF': 'east', 'BAL': 'east', 'CLE': 'east',
            'CIN': 'east', 'PIT': 'east'
        }
        
        # Calculate travel distance (0-3 scale)
        region_distances = {
            ('west', 'west'): 0, ('west', 'mountain'): 1, ('west', 'central'): 2, ('west', 'east'): 3,
            ('mountain', 'west'): 1, ('mountain', 'mountain'): 0, ('mountain', 'central'): 1, ('mountain', 'east'): 2,
            ('central', 'west'): 2, ('central', 'mountain'): 1, ('central', 'central'): 0, ('central', 'east'): 1,
            ('east', 'west'): 3, ('east', 'mountain'): 2, ('east', 'central'): 1, ('east', 'east'): 0,
        }
        
        away_region = team_regions.get(away_team, 'central')
        home_region = team_regions.get(home_team, 'central')
        
        features['away_travel_distance'] = region_distances.get((away_region, home_region), 1)
        features['is_cross_country'] = 1.0 if features['away_travel_distance'] == 3 else 0.0
        
        # Check if game is primetime or short week
        game_row = schedule_df[schedule_df['gameday'] == game_date.strftime('%Y-%m-%d')]
        if len(game_row) == 0:
            game_row = schedule_df[
                (schedule_df['home_team'] == home_team) & 
                (schedule_df['away_team'] == away_team) &
                (pd.to_datetime(schedule_df['gameday']).dt.date == game_date.date())
            ]
        
        features['is_primetime'] = 0.0
        features['is_short_week'] = 0.0
        
        if len(game_row) > 0:
            game = game_row.iloc[0]
            # Check for primetime (SNF, MNF, TNF)
            game_time = game.get('gametime', '')
            if pd.notna(game_time):
                if '20:' in str(game_time) or '21:' in str(game_time):  # 8pm or later = primetime
                    features['is_primetime'] = 1.0
            
            # Thursday = short week (day_of_week == 3)
            if game_date.weekday() == 3:  # Thursday
                features['is_short_week'] = 1.0
        
        return features
    
    def _extract_bye_features(
        self,
        home_team: str,
        away_team: str,
        schedule_df: pd.DataFrame,
        game_date: datetime
    ) -> Dict[str, Any]:
        """Extract bye week advantage features.
        
        Args:
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            schedule_df: Schedule DataFrame
            game_date: Date of the game
            
        Returns:
            Dictionary of bye week features
        """
        features = {}
        
        def check_post_bye(team):
            """Check if team is coming off a bye week."""
            # Get team's last game before current date
            team_games = schedule_df[
                (pd.to_datetime(schedule_df['gameday']) < game_date) &
                ((schedule_df['home_team'] == team) | (schedule_df['away_team'] == team)) &
                (schedule_df['home_score'].notna())
            ].sort_values('gameday', ascending=False)
            
            if len(team_games) == 0:
                return 0.0
            
            last_game_date = pd.to_datetime(team_games.iloc[0]['gameday'])
            days_since_last = (game_date - last_game_date).days
            
            # If 12+ days since last game, team had a bye
            return 1.0 if days_since_last >= 12 else 0.0
        
        features['home_post_bye'] = check_post_bye(home_team)
        features['away_post_bye'] = check_post_bye(away_team)
        
        # Bye advantage (being rested vs opponent)
        features['bye_advantage'] = features['home_post_bye'] - features['away_post_bye']
        
        # Double advantage: post-bye at home
        features['home_bye_at_home'] = features['home_post_bye'] * 1.0  # Already at home
        
        return features
    
    def _extract_ir_return_features(
        self,
        home_team: str,
        away_team: str,
        roster_df: pd.DataFrame,
        week: int,
        season: int
    ) -> Dict[str, Any]:
        """Extract features for players returning from injured reserve.
        
        Args:
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            roster_df: Weekly roster DataFrame with status info
            week: Current week number
            season: Current season year
            
        Returns:
            Dictionary of IR return features
        """
        features = {}
        
        # Position impact weights (higher = more impactful return)
        position_weights = {
            'QB': 3.0,      # Quarterback return is massive
            'RB': 1.5,      # Running backs
            'WR': 1.2,      # Wide receivers
            'TE': 1.0,      # Tight ends
            'OL': 1.0, 'T': 1.0, 'G': 0.8, 'C': 0.8,  # O-line
            'DL': 1.0, 'DE': 1.0, 'DT': 0.9,  # D-line
            'LB': 1.1,      # Linebackers
            'DB': 1.0, 'CB': 1.0, 'S': 0.9,   # Secondary
            'K': 0.3, 'P': 0.2  # Special teams
        }
        
        def get_ir_returns(team: str) -> tuple:
            """Get IR returns for a team this week.
            
            Returns:
                Tuple of (count of returns, impact score, has key player)
            """
            # Filter roster data for the team and relevant seasons
            team_roster = roster_df[
                (roster_df['team'] == team) & 
                (roster_df['season'] == season)
            ]
            
            if len(team_roster) == 0:
                return 0, 0.0, 0
            
            # Sort by player and week to track transitions
            team_roster = team_roster.sort_values(['player_id', 'week'])
            
            ir_returns = 0
            impact_score = 0.0
            key_player = 0
            
            # Track unique players
            seen_returns = set()
            
            for player_id in team_roster['player_id'].unique():
                player_data = team_roster[team_roster['player_id'] == player_id]
                
                # Get status for current week and previous week
                current_week_data = player_data[player_data['week'] == week]
                prev_week_data = player_data[player_data['week'] == week - 1]
                
                if len(current_week_data) == 0 or len(prev_week_data) == 0:
                    continue
                
                current_status = current_week_data.iloc[0]['status']
                prev_status = prev_week_data.iloc[0]['status']
                position = current_week_data.iloc[0].get('position', 'UNK')
                
                # Check if player just came off IR (RES -> ACT)
                if prev_status == 'RES' and current_status == 'ACT':
                    if player_id not in seen_returns:
                        seen_returns.add(player_id)
                        ir_returns += 1
                        
                        # Calculate impact based on position
                        weight = position_weights.get(position, 0.5)
                        impact_score += weight
                        
                        # Mark if key player (QB or high-impact position)
                        if position == 'QB' or weight >= 1.5:
                            key_player = 1
            
            return ir_returns, impact_score, key_player
        
        # Get IR return features for each team
        home_returns, home_impact, home_key = get_ir_returns(home_team)
        away_returns, away_impact, away_key = get_ir_returns(away_team)
        
        features['home_ir_returns'] = home_returns
        features['away_ir_returns'] = away_returns
        features['home_ir_impact_score'] = home_impact
        features['away_ir_impact_score'] = away_impact
        features['ir_advantage'] = home_impact - away_impact
        features['home_key_player_returning'] = home_key
        features['away_key_player_returning'] = away_key
        
        return features
    
    def _extract_weather_rest_features(
        self,
        game_info: GameInfo,
        schedule_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Extract weather and rest day features.
        
        Args:
            game_info: Game information
            schedule_df: Schedule DataFrame
            
        Returns:
            Dictionary of weather and rest features
        """
        features = {}
        
        # Find the game in schedule
        game_row = schedule_df[schedule_df['game_id'] == game_info.game_id]
        
        if len(game_row) > 0:
            game = game_row.iloc[0]
            
            # Rest days (days since last game)
            if pd.notna(game.get('away_rest')):
                features['away_rest_days'] = float(game['away_rest'])
            else:
                features['away_rest_days'] = 7.0  # Default to 7 if missing
            
            if pd.notna(game.get('home_rest')):
                features['home_rest_days'] = float(game['home_rest'])
            else:
                features['home_rest_days'] = 7.0  # Default to 7 if missing
            
            features['rest_days_diff'] = features['home_rest_days'] - features['away_rest_days']
            
            # Weather features
            if pd.notna(game.get('temp')):
                features['temperature'] = float(game['temp'])
            else:
                features['temperature'] = 70.0  # Default moderate temperature
            
            if pd.notna(game.get('wind')):
                features['wind_speed'] = float(game['wind'])
            else:
                features['wind_speed'] = 0.0  # Default no wind
            
            # Roof type (dome = 1, outdoor = 0)
            roof = str(game.get('roof', 'outdoors')).lower()
            features['is_dome'] = 1.0 if 'dome' in roof or 'closed' in roof else 0.0
            
            # Surface type (turf = 1, grass = 0)
            surface = str(game.get('surface', 'grass')).lower()
            features['is_turf'] = 1.0 if 'turf' in surface or 'artificial' in surface else 0.0
            
            # Weather impact indicators
            features['cold_weather'] = 1.0 if features['temperature'] < 40 else 0.0
            features['windy'] = 1.0 if features['wind_speed'] > 15 else 0.0
        else:
            # Default values if game not found
            features['away_rest_days'] = 7.0
            features['home_rest_days'] = 7.0
            features['rest_days_diff'] = 0.0
            features['temperature'] = 70.0
            features['wind_speed'] = 0.0
            features['is_dome'] = 0.0
            features['is_turf'] = 0.0
            features['cold_weather'] = 0.0
            features['windy'] = 0.0
        
        return features
    
    def _extract_epa_features(
        self,
        home_team: str,
        away_team: str,
        weekly_stats_df: pd.DataFrame,
        schedule_df: pd.DataFrame,
        game_date: datetime,
        n_games: int = 5
    ) -> Dict[str, Any]:
        """Extract EPA (Expected Points Added) metrics with caching.
        
        Args:
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            weekly_stats_df: Weekly player statistics DataFrame
            schedule_df: Schedule DataFrame
            game_date: Date of the game
            n_games: Number of recent games to consider
            
        Returns:
            Dictionary of EPA features
        """
        # Check cache first
        cache_key = f"{home_team}_{away_team}_{game_date.strftime('%Y%m%d')}"
        if cache_key in self.epa_cache:
            return self.epa_cache[cache_key]
        
        features = {}
        
        # Get recent games for each team
        home_recent_games = self._get_recent_games(home_team, schedule_df, game_date, n_games)
        away_recent_games = self._get_recent_games(away_team, schedule_df, game_date, n_games)
        
        # Helper function to calculate team EPA from weekly stats
        def get_team_epa(team: str, recent_games: pd.DataFrame, weekly_df: pd.DataFrame) -> Dict[str, float]:
            """Calculate team-level EPA metrics."""
            if len(recent_games) == 0:
                return {
                    'off_epa': 0.0,
                    'def_epa': 0.0,
                    'total_epa': 0.0
                }
            
            # Get game IDs
            game_ids = recent_games['game_id'].tolist()
            
            # Get team's offensive EPA (from their players)
            team_off_epa = []
            team_def_epa = []
            
            for game_id in game_ids:
                # Get game info
                game = recent_games[recent_games['game_id'] == game_id]
                if len(game) == 0:
                    continue
                game_row = game.iloc[0]
                game_week = int(game_row['week'])
                game_season = int(game_row['season'])
                
                # Get team's offensive EPA (passing + rushing + receiving)
                try:
                    team_week_stats = weekly_df[
                        (weekly_df['recent_team'] == team) &
                        (weekly_df['season'] == game_season) &
                        (weekly_df['week'] == game_week)
                    ]
                    
                    if len(team_week_stats) > 0:
                        # Sum EPA from all offensive plays
                        passing_epa = team_week_stats.get('passing_epa', pd.Series([0])).fillna(0).sum()
                        rushing_epa = team_week_stats.get('rushing_epa', pd.Series([0])).fillna(0).sum()
                        receiving_epa = team_week_stats.get('receiving_epa', pd.Series([0])).fillna(0).sum()
                        off_epa = passing_epa + rushing_epa + receiving_epa
                        team_off_epa.append(off_epa)
                except Exception:
                    # Skip this game if there's an error
                    pass
                
                # Get opponent's offensive EPA (which is this team's defensive EPA allowed)
                try:
                    opponent = game_row['away_team'] if game_row['home_team'] == team else game_row['home_team']
                    opp_week_stats = weekly_df[
                        (weekly_df['recent_team'] == opponent) &
                        (weekly_df['season'] == game_season) &
                        (weekly_df['week'] == game_week)
                    ]
                    
                    if len(opp_week_stats) > 0:
                        opp_passing_epa = opp_week_stats.get('passing_epa', pd.Series([0])).fillna(0).sum()
                        opp_rushing_epa = opp_week_stats.get('rushing_epa', pd.Series([0])).fillna(0).sum()
                        opp_receiving_epa = opp_week_stats.get('receiving_epa', pd.Series([0])).fillna(0).sum()
                        def_epa_allowed = opp_passing_epa + opp_rushing_epa + opp_receiving_epa
                        team_def_epa.append(-def_epa_allowed)  # Negative because it's points allowed
                except Exception:
                    # Skip this game if there's an error
                    pass
            
            return {
                'off_epa': np.mean(team_off_epa) if team_off_epa else 0.0,
                'def_epa': np.mean(team_def_epa) if team_def_epa else 0.0,
                'total_epa': (np.mean(team_off_epa) if team_off_epa else 0.0) + 
                            (np.mean(team_def_epa) if team_def_epa else 0.0)
            }
        
        # Calculate EPA for both teams
        home_epa = get_team_epa(home_team, home_recent_games, weekly_stats_df)
        away_epa = get_team_epa(away_team, away_recent_games, weekly_stats_df)
        
        # Offensive EPA
        features['home_off_epa'] = home_epa['off_epa']
        features['away_off_epa'] = away_epa['off_epa']
        
        # Defensive EPA (negative is better - means allowing fewer points)
        features['home_def_epa'] = home_epa['def_epa']
        features['away_def_epa'] = away_epa['def_epa']
        
        # Total EPA (offense + defense)
        features['home_total_epa'] = home_epa['total_epa']
        features['away_total_epa'] = away_epa['total_epa']
        
        # EPA differentials
        features['epa_diff'] = home_epa['total_epa'] - away_epa['total_epa']
        features['off_epa_diff'] = home_epa['off_epa'] - away_epa['off_epa']
        features['def_epa_diff'] = home_epa['def_epa'] - away_epa['def_epa']
        
        # Cache the result
        self.epa_cache[cache_key] = features
        return features

