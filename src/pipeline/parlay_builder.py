"""Parlay builder module for generating and managing multi-leg parlays."""

import re
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class BetOption:
    """Represents a single betting option."""
    bet_type: str  # 'game_winner', 'spread', 'total', 'player_yards', 'player_td'
    description: str
    pick: str
    confidence: float
    game: str  # "PHI @ LAC"
    details: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self):
        return f"{self.pick} ({self.confidence:.0%})"


class ParlayBuilder:
    """Build parlays by generating and ranking betting options."""
    
    # Key players per team (top skill position players) - 2025 SEASON ROSTERS
    TEAM_KEY_PLAYERS = {
        'PHI': ['A.J. Brown', 'Devonta Smith', 'Saquon Barkley', 'Jalen Hurts', 'Dallas Goedert'],
        'LAC': ['Ladd McConkey', 'Quentin Johnston', 'Gus Edwards', 'Justin Herbert'],
        'KC': ['Travis Kelce', 'Xavier Worthy', 'Isiah Pacheco', 'Patrick Mahomes', 'DeAndre Hopkins'],
        'BUF': ['Amari Cooper', 'Keon Coleman', 'James Cook', 'Dalton Kincaid', 'Josh Allen'],
        'SF': ['Deebo Samuel', 'George Kittle', 'Christian McCaffrey', 'Brock Purdy'],
        'DAL': ['CeeDee Lamb', 'Rico Dowdle', 'Jake Ferguson', 'Cooper Rush'],
        'MIA': ['Tyreek Hill', 'Jaylen Waddle', 'De\'Von Achane', 'Tua Tagovailoa'],
        'DET': ['Amon-Ra St. Brown', 'Jahmyr Gibbs', 'Sam LaPorta', 'Jared Goff', 'Jameson Williams'],
        'BAL': ['Zay Flowers', 'Mark Andrews', 'Derrick Henry', 'Lamar Jackson'],
        'CIN': ['Ja\'Marr Chase', 'Tee Higgins', 'Chase Brown', 'Joe Burrow'],
        'MIN': ['Justin Jefferson', 'Jordan Addison', 'Aaron Jones', 'Sam Darnold'],
        'GB': ['Jayden Reed', 'Romeo Doubs', 'Josh Jacobs', 'Jordan Love'],
        'NYJ': ['Garrett Wilson', 'Davante Adams', 'Breece Hall', 'Aaron Rodgers'],
        'DEN': ['Courtland Sutton', 'J.K. Dobbins', 'Bo Nix', 'Marvin Mims Jr'],
        'HOU': ['Nico Collins', 'Tank Dell', 'Stefon Diggs', 'Joe Mixon', 'C.J. Stroud'],
        'CLE': ['Jerry Jeudy', 'David Njoku', 'Nick Chubb', 'Jameis Winston'],
        'PIT': ['George Pickens', 'Najee Harris', 'Pat Freiermuth', 'Russell Wilson'],
        'SEA': ['DK Metcalf', 'Jaxon Smith-Njigba', 'Kenneth Walker', 'Geno Smith'],
        'TB': ['Mike Evans', 'Chris Godwin', 'Bucky Irving', 'Baker Mayfield'],
        'NO': ['Chris Olave', 'Alvin Kamara', 'Derek Carr'],
        'ATL': ['Drake London', 'Bijan Robinson', 'Kyle Pitts', 'Kirk Cousins', 'Darnell Mooney'],
        'CAR': ['Adam Thielen', 'Chuba Hubbard', 'Bryce Young', 'Xavier Legette'],
        'LV': ['Jakobi Meyers', 'Brock Bowers', 'Alexander Mattison', 'Aidan O\'Connell'],
        'ARI': ['Marvin Harrison Jr', 'Trey McBride', 'James Conner', 'Kyler Murray'],
        'LAR': ['Puka Nacua', 'Cooper Kupp', 'Kyren Williams', 'Matthew Stafford'],
        'NYG': ['Malik Nabers', 'Wan\'Dale Robinson', 'Tyrone Tracy Jr', 'Tommy DeVito'],
        'CHI': ['D.J. Moore', 'Rome Odunze', 'D\'Andre Swift', 'Caleb Williams'],
        'WAS': ['Terry McLaurin', 'Brian Robinson Jr', 'Zach Ertz', 'Jayden Daniels'],
        'JAX': ['Brian Thomas Jr', 'Travis Etienne', 'Evan Engram', 'Mac Jones'],
        'TEN': ['Calvin Ridley', 'Tony Pollard', 'Will Levis', 'Chig Okonkwo'],
        'IND': ['Michael Pittman Jr', 'Jonathan Taylor', 'Anthony Richardson'],
        'NE': ['Rhamondre Stevenson', 'Hunter Henry', 'Drake Maye'],
    }
    
    # Team name mapping
    TEAM_MAPPING = {
        'chiefs': 'KC', 'kansas city': 'KC', 'kansas city chiefs': 'KC',
        'raiders': 'LV', 'las vegas': 'LV', 'las vegas raiders': 'LV',
        'broncos': 'DEN', 'denver': 'DEN', 'denver broncos': 'DEN',
        'chargers': 'LAC', 'los angeles chargers': 'LAC', 'la chargers': 'LAC',
        'bills': 'BUF', 'buffalo': 'BUF', 'buffalo bills': 'BUF',
        'dolphins': 'MIA', 'miami': 'MIA', 'miami dolphins': 'MIA',
        'patriots': 'NE', 'new england': 'NE', 'new england patriots': 'NE',
        'jets': 'NYJ', 'new york jets': 'NYJ', 'ny jets': 'NYJ',
        'ravens': 'BAL', 'baltimore': 'BAL', 'baltimore ravens': 'BAL',
        'bengals': 'CIN', 'cincinnati': 'CIN', 'cincinnati bengals': 'CIN',
        'browns': 'CLE', 'cleveland': 'CLE', 'cleveland browns': 'CLE',
        'steelers': 'PIT', 'pittsburgh': 'PIT', 'pittsburgh steelers': 'PIT',
        'texans': 'HOU', 'houston': 'HOU', 'houston texans': 'HOU',
        'colts': 'IND', 'indianapolis': 'IND', 'indianapolis colts': 'IND',
        'jaguars': 'JAX', 'jacksonville': 'JAX', 'jacksonville jaguars': 'JAX',
        'titans': 'TEN', 'tennessee': 'TEN', 'tennessee titans': 'TEN',
        'cowboys': 'DAL', 'dallas': 'DAL', 'dallas cowboys': 'DAL',
        'giants': 'NYG', 'new york giants': 'NYG', 'ny giants': 'NYG',
        'eagles': 'PHI', 'philadelphia': 'PHI', 'philadelphia eagles': 'PHI',
        'commanders': 'WAS', 'washington': 'WAS', 'washington commanders': 'WAS',
        'bears': 'CHI', 'chicago': 'CHI', 'chicago bears': 'CHI',
        'lions': 'DET', 'detroit': 'DET', 'detroit lions': 'DET',
        'packers': 'GB', 'green bay': 'GB', 'green bay packers': 'GB',
        'vikings': 'MIN', 'minnesota': 'MIN', 'minnesota vikings': 'MIN',
        'falcons': 'ATL', 'atlanta': 'ATL', 'atlanta falcons': 'ATL',
        'panthers': 'CAR', 'carolina': 'CAR', 'carolina panthers': 'CAR',
        'saints': 'NO', 'new orleans': 'NO', 'new orleans saints': 'NO',
        'buccaneers': 'TB', 'tampa bay': 'TB', 'tampa bay buccaneers': 'TB', 'bucs': 'TB',
        'cardinals': 'ARI', 'arizona': 'ARI', 'arizona cardinals': 'ARI',
        'rams': 'LAR', 'los angeles rams': 'LAR', 'la rams': 'LAR',
        '49ers': 'SF', 'san francisco': 'SF', 'san francisco 49ers': 'SF', 'niners': 'SF',
        'seahawks': 'SEA', 'seattle': 'SEA', 'seattle seahawks': 'SEA',
    }
    
    TEAM_FULL_NAMES = {
        'KC': 'Kansas City Chiefs', 'LV': 'Las Vegas Raiders', 'DEN': 'Denver Broncos',
        'LAC': 'Los Angeles Chargers', 'BUF': 'Buffalo Bills', 'MIA': 'Miami Dolphins',
        'NE': 'New England Patriots', 'NYJ': 'New York Jets', 'BAL': 'Baltimore Ravens',
        'CIN': 'Cincinnati Bengals', 'CLE': 'Cleveland Browns', 'PIT': 'Pittsburgh Steelers',
        'HOU': 'Houston Texans', 'IND': 'Indianapolis Colts', 'JAX': 'Jacksonville Jaguars',
        'TEN': 'Tennessee Titans', 'DAL': 'Dallas Cowboys', 'NYG': 'New York Giants',
        'PHI': 'Philadelphia Eagles', 'WAS': 'Washington Commanders', 'CHI': 'Chicago Bears',
        'DET': 'Detroit Lions', 'GB': 'Green Bay Packers', 'MIN': 'Minnesota Vikings',
        'ATL': 'Atlanta Falcons', 'CAR': 'Carolina Panthers', 'NO': 'New Orleans Saints',
        'TB': 'Tampa Bay Buccaneers', 'ARI': 'Arizona Cardinals', 'LAR': 'Los Angeles Rams',
        'SF': 'San Francisco 49ers', 'SEA': 'Seattle Seahawks',
    }
    
    def __init__(self, model_dir: str = 'models/trained'):
        """Initialize parlay builder.
        
        Args:
            model_dir: Directory containing trained models
        """
        self.model_dir = Path(model_dir)
        self.schedule = None
        self.player_prop_predictor = None
        self.quick_predictor = None
        
    def _get_player_prop_predictor(self):
        """Lazy load player prop predictor."""
        if self.player_prop_predictor is None:
            from src.pipeline.quick_predict import PlayerPropPredictor
            self.player_prop_predictor = PlayerPropPredictor()
        return self.player_prop_predictor
    
    def _get_quick_predictor(self):
        """Lazy load quick predictor."""
        if self.quick_predictor is None:
            from src.pipeline.quick_predict import QuickPredictor
            self.quick_predictor = QuickPredictor(str(self.model_dir))
        return self.quick_predictor
    
    def load_schedule(self, season: int = 2025) -> pd.DataFrame:
        """Load NFL schedule.
        
        Args:
            season: Season year
            
        Returns:
            Schedule dataframe
        """
        if self.schedule is None:
            import nfl_data_py as nfl
            try:
                self.schedule = nfl.import_schedules([season])
                logger.info(f"Loaded {len(self.schedule)} games for {season}")
            except Exception as e:
                logger.error(f"Error loading schedule: {e}")
                return pd.DataFrame()
        return self.schedule
    
    def normalize_team(self, team_name: str) -> Optional[str]:
        """Normalize team name to abbreviation.
        
        Args:
            team_name: Team name in any format
            
        Returns:
            Team abbreviation (e.g., 'PHI') or None
        """
        team_lower = team_name.lower().strip()
        
        # Check direct mapping
        if team_lower in self.TEAM_MAPPING:
            return self.TEAM_MAPPING[team_lower]
        
        # Check if already abbreviation
        if team_name.upper() in self.TEAM_FULL_NAMES:
            return team_name.upper()
        
        # Try partial match
        for key, abbr in self.TEAM_MAPPING.items():
            if key in team_lower or team_lower in key:
                return abbr
        
        return None
    
    def parse_parlay_query(self, query: str) -> Dict[str, Any]:
        """Parse parlay query to extract parameters.
        
        Args:
            query: Natural language query like "4 leg parlay Eagles vs Chargers"
            
        Returns:
            Dictionary with legs, teams, week
        """
        query_lower = query.lower()
        
        # Extract number of legs
        leg_patterns = [
            r'(\d+)\s*(?:leg|legs)',
            r'(\d+)\s*(?:pick|picks)',
            r'build\s*(?:a\s*)?(\d+)',
        ]
        num_legs = 4  # default
        for pattern in leg_patterns:
            match = re.search(pattern, query_lower)
            if match:
                num_legs = int(match.group(1))
                break
        
        # Extract week number
        week_pattern = r'week\s*(\d+)'
        week_match = re.search(week_pattern, query_lower)
        week = int(week_match.group(1)) if week_match else None
        
        # Extract teams
        teams = []
        for team_name, abbr in self.TEAM_MAPPING.items():
            if team_name in query_lower:
                if abbr not in teams:
                    teams.append(abbr)
        
        # Also check for abbreviations
        words = query.upper().split()
        for word in words:
            if word in self.TEAM_FULL_NAMES and word not in teams:
                teams.append(word)
        
        return {
            'num_legs': num_legs,
            'teams': teams[:2],  # Max 2 teams for single game
            'week': week,
            'is_single_game': len(teams) == 2 and week is None,
            'is_week_parlay': week is not None,
        }
    
    def find_game(self, team1: str, team2: str) -> Optional[Dict[str, Any]]:
        """Find upcoming game between two teams.
        
        Args:
            team1: First team abbreviation
            team2: Second team abbreviation
            
        Returns:
            Game info dict or None
        """
        schedule = self.load_schedule()
        if schedule.empty:
            return None
        
        # Find game with these teams
        games = schedule[
            ((schedule['home_team'] == team1) & (schedule['away_team'] == team2)) |
            ((schedule['home_team'] == team2) & (schedule['away_team'] == team1))
        ]
        
        # Filter to upcoming/current week games
        if 'gameday' in games.columns:
            today = datetime.now().strftime('%Y-%m-%d')
            upcoming = games[games['gameday'] >= today]
            if not upcoming.empty:
                games = upcoming
        
        if games.empty:
            return None
        
        game = games.iloc[0]
        return {
            'home_team': game['home_team'],
            'away_team': game['away_team'],
            'week': int(game['week']),
            'gameday': game.get('gameday', 'TBD'),
            'spread': game.get('spread_line', 0) if pd.notna(game.get('spread_line')) else 0,
            'total': game.get('total_line', 45) if pd.notna(game.get('total_line')) else 45,
        }
    
    def get_week_games(self, week: int) -> List[Dict[str, Any]]:
        """Get all games for a given week.
        
        Args:
            week: Week number
            
        Returns:
            List of game info dicts
        """
        schedule = self.load_schedule()
        if schedule.empty:
            return []
        
        week_games = schedule[schedule['week'] == week]
        games = []
        
        for _, game in week_games.iterrows():
            games.append({
                'home_team': game['home_team'],
                'away_team': game['away_team'],
                'week': int(game['week']),
                'gameday': game.get('gameday', 'TBD'),
                'spread': game.get('spread_line', 0) if pd.notna(game.get('spread_line')) else 0,
                'total': game.get('total_line', 45) if pd.notna(game.get('total_line')) else 45,
            })
        
        return games
    
    def generate_game_winner_options(self, game: Dict[str, Any]) -> List[BetOption]:
        """Generate game winner betting options.
        
        Args:
            game: Game info dict
            
        Returns:
            List of BetOption for game winner
        """
        options = []
        home = game['home_team']
        away = game['away_team']
        game_str = f"{away} @ {home}"
        
        # Get prediction from quick predictor
        qp = self._get_quick_predictor()
        try:
            result = qp.predict(f"{home} vs {away}")
            if result.get('success') and result.get('prediction'):
                winner = result['prediction']
                confidence = result.get('confidence', 0.5)
                
                # Home team win
                if winner == home:
                    options.append(BetOption(
                        bet_type='game_winner',
                        description=f"{self.TEAM_FULL_NAMES.get(home, home)} Moneyline",
                        pick=f"{home} ML",
                        confidence=confidence,
                        game=game_str,
                        details={'winner': home}
                    ))
                    # Also add away as lower confidence
                    options.append(BetOption(
                        bet_type='game_winner',
                        description=f"{self.TEAM_FULL_NAMES.get(away, away)} Moneyline",
                        pick=f"{away} ML",
                        confidence=1 - confidence,
                        game=game_str,
                        details={'winner': away}
                    ))
                else:
                    options.append(BetOption(
                        bet_type='game_winner',
                        description=f"{self.TEAM_FULL_NAMES.get(away, away)} Moneyline",
                        pick=f"{away} ML",
                        confidence=confidence,
                        game=game_str,
                        details={'winner': away}
                    ))
                    options.append(BetOption(
                        bet_type='game_winner',
                        description=f"{self.TEAM_FULL_NAMES.get(home, home)} Moneyline",
                        pick=f"{home} ML",
                        confidence=1 - confidence,
                        game=game_str,
                        details={'winner': home}
                    ))
        except Exception as e:
            logger.warning(f"Error getting game prediction: {e}")
            # Fallback: home team slight favorite
            options.append(BetOption(
                bet_type='game_winner',
                description=f"{self.TEAM_FULL_NAMES.get(home, home)} Moneyline",
                pick=f"{home} ML",
                confidence=0.55,
                game=game_str,
                details={'winner': home}
            ))
        
        return options
    
    def generate_spread_options(self, game: Dict[str, Any]) -> List[BetOption]:
        """Generate spread betting options.
        
        Args:
            game: Game info dict
            
        Returns:
            List of BetOption for spreads
        """
        options = []
        home = game['home_team']
        away = game['away_team']
        spread = game.get('spread', 0)
        game_str = f"{away} @ {home}"
        
        if spread == 0:
            return options  # No spread available
        
        # Home covers spread
        home_spread = spread if spread < 0 else -spread
        options.append(BetOption(
            bet_type='spread',
            description=f"{self.TEAM_FULL_NAMES.get(home, home)} {home_spread:+.1f}",
            pick=f"{home} {home_spread:+.1f}",
            confidence=0.52,  # Spreads are close to 50/50
            game=game_str,
            details={'spread': home_spread, 'team': home}
        ))
        
        # Away covers spread
        away_spread = -home_spread
        options.append(BetOption(
            bet_type='spread',
            description=f"{self.TEAM_FULL_NAMES.get(away, away)} {away_spread:+.1f}",
            pick=f"{away} {away_spread:+.1f}",
            confidence=0.48,
            game=game_str,
            details={'spread': away_spread, 'team': away}
        ))
        
        return options
    
    def generate_total_options(self, game: Dict[str, Any]) -> List[BetOption]:
        """Generate over/under total options.
        
        Args:
            game: Game info dict
            
        Returns:
            List of BetOption for totals
        """
        options = []
        home = game['home_team']
        away = game['away_team']
        total = game.get('total', 45)
        game_str = f"{away} @ {home}"
        
        if total == 0:
            return options
        
        options.append(BetOption(
            bet_type='total',
            description=f"Over {total}",
            pick=f"Over {total}",
            confidence=0.52,  # Slight lean to over in modern NFL
            game=game_str,
            details={'total': total, 'direction': 'over'}
        ))
        
        options.append(BetOption(
            bet_type='total',
            description=f"Under {total}",
            pick=f"Under {total}",
            confidence=0.48,
            game=game_str,
            details={'total': total, 'direction': 'under'}
        ))
        
        return options
    
    def generate_player_yards_options(self, game: Dict[str, Any], num_players: int = 3) -> List[BetOption]:
        """Generate player yards prop options.
        
        Args:
            game: Game info dict
            num_players: Number of players per team to generate props for
            
        Returns:
            List of BetOption for player yards
        """
        options = []
        home = game['home_team']
        away = game['away_team']
        game_str = f"{away} @ {home}"
        
        pp = self._get_player_prop_predictor()
        
        for team in [home, away]:
            players = self.TEAM_KEY_PLAYERS.get(team, [])[:num_players]
            
            for player in players:
                try:
                    # Determine stat type based on player position (2025 rosters)
                    # Use full names where there's ambiguity to avoid false matches
                    # QBs - passing yards (use full names to avoid matching WRs like Devonta Smith)
                    qb_full_names = ['Patrick Mahomes', 'Josh Allen', 'Lamar Jackson', 'Jalen Hurts', 
                                    'Justin Herbert', 'Joe Burrow', 'Brock Purdy', 'Jared Goff', 
                                    'Sam Darnold', 'Jordan Love', 'Aaron Rodgers', 'Bo Nix', 
                                    'C.J. Stroud', 'Jameis Winston', 'Russell Wilson', 'Geno Smith',
                                    'Baker Mayfield', 'Derek Carr', 'Kirk Cousins', 'Kyler Murray', 
                                    'Matthew Stafford', 'Caleb Williams', 'Jayden Daniels', 'Will Levis', 
                                    'Anthony Richardson', 'Drake Maye', 'Tommy DeVito', 'Aidan O\'Connell', 
                                    'Mac Jones', 'Cooper Rush', 'Bryce Young']
                    # RBs - rushing yards
                    rb_full_names = ['Saquon Barkley', 'Derrick Henry', 'Christian McCaffrey', 
                                    'Jonathan Taylor', 'Josh Jacobs', 'James Cook', 'Jahmyr Gibbs',
                                    'De\'Von Achane', 'Nick Chubb', 'J.K. Dobbins', 'Gus Edwards', 
                                    'Joe Mixon', 'Najee Harris', 'Kenneth Walker', 'Bucky Irving',
                                    'Alvin Kamara', 'Bijan Robinson', 'Brian Robinson Jr', 'Chuba Hubbard',
                                    'Alexander Mattison', 'James Conner', 'Kyren Williams', 'D\'Andre Swift',
                                    'Travis Etienne', 'Tony Pollard', 'Rhamondre Stevenson', 'Aaron Jones',
                                    'Tyrone Tracy Jr', 'Rico Dowdle', 'Isiah Pacheco']
                    
                    if any(qb == player for qb in qb_full_names):
                        stat_type = 'passing_yards'
                        default_line = 250
                    elif any(rb == player for rb in rb_full_names):
                        stat_type = 'rushing_yards'
                        default_line = 60
                    else:
                        stat_type = 'receiving_yards'
                        default_line = 55
                    
                    # Get player stats
                    result = pp.predict_over_under(player, default_line, stat_type)
                    
                    if result.get('success'):
                        avg = result.get('avg_yards', default_line)
                        line = round(avg * 0.9 / 5) * 5  # Set line slightly below average
                        confidence = result.get('confidence', 0.5)
                        prediction = result.get('prediction', 'OVER')
                        
                        stat_label = stat_type.replace('_', ' ').replace('yards', 'yds')
                        
                        if prediction == 'OVER':
                            options.append(BetOption(
                                bet_type='player_yards',
                                description=f"{player} Over {line} {stat_label}",
                                pick=f"{player} O{line} {stat_label}",
                                confidence=confidence,
                                game=game_str,
                                details={'player': player, 'line': line, 'stat': stat_type, 'direction': 'over'}
                            ))
                        else:
                            options.append(BetOption(
                                bet_type='player_yards',
                                description=f"{player} Under {line} {stat_label}",
                                pick=f"{player} U{line} {stat_label}",
                                confidence=confidence,
                                game=game_str,
                                details={'player': player, 'line': line, 'stat': stat_type, 'direction': 'under'}
                            ))
                            
                except Exception as e:
                    logger.debug(f"Error generating prop for {player}: {e}")
                    continue
        
        return options
    
    def generate_player_td_options(self, game: Dict[str, Any], num_players: int = 3) -> List[BetOption]:
        """Generate player touchdown prop options.
        
        Args:
            game: Game info dict
            num_players: Number of players per team to generate TD props for
            
        Returns:
            List of BetOption for player TDs
        """
        options = []
        home = game['home_team']
        away = game['away_team']
        game_str = f"{away} @ {home}"
        
        pp = self._get_player_prop_predictor()
        
        for team in [home, away]:
            players = self.TEAM_KEY_PLAYERS.get(team, [])[:num_players]
            
            for player in players:
                # Skip pocket QBs for anytime TD (they rarely score rushing/receiving TDs)
                # Note: Mobile QBs like Hurts, Jackson, Daniels are NOT skipped
                qb_skip = ['Patrick Mahomes', 'Josh Allen', 'Justin Herbert', 'Joe Burrow', 
                          'Brock Purdy', 'Jared Goff', 'Sam Darnold', 'Jordan Love', 
                          'Aaron Rodgers', 'Bo Nix', 'C.J. Stroud', 'Jameis Winston', 
                          'Russell Wilson', 'Geno Smith', 'Baker Mayfield', 'Derek Carr', 
                          'Kirk Cousins', 'Kyler Murray', 'Matthew Stafford', 'Will Levis', 
                          'Anthony Richardson', 'Drake Maye', 'Tommy DeVito', 'Aidan O\'Connell', 
                          'Mac Jones', 'Cooper Rush', 'Bryce Young']
                if any(qb == player for qb in qb_skip):
                    continue
                
                try:
                    result = pp.predict_over_under(player, 0.5, 'anytime_td')
                    
                    if result.get('success') and result.get('is_td_prop'):
                        td_rate = result.get('td_rate', 0)
                        confidence = result.get('confidence', 0.5)
                        prediction = result.get('prediction', 'NO')
                        
                        if prediction == 'YES' and td_rate >= 0.3:  # Only include if decent chance
                            options.append(BetOption(
                                bet_type='player_td',
                                description=f"{player} Anytime TD",
                                pick=f"{player} TD",
                                confidence=confidence,
                                game=game_str,
                                details={'player': player, 'td_rate': td_rate}
                            ))
                            
                except Exception as e:
                    logger.debug(f"Error generating TD prop for {player}: {e}")
                    continue
        
        return options
    
    def generate_game_options(self, team1: str, team2: str) -> List[BetOption]:
        """Generate all betting options for a single game.
        
        Args:
            team1: First team
            team2: Second team
            
        Returns:
            List of all BetOptions sorted by confidence
        """
        # Normalize teams
        t1 = self.normalize_team(team1)
        t2 = self.normalize_team(team2)
        
        if not t1 or not t2:
            logger.error(f"Could not identify teams: {team1}, {team2}")
            return []
        
        # Find the game
        game = self.find_game(t1, t2)
        if not game:
            logger.warning(f"Could not find game between {t1} and {t2}")
            # Create a placeholder game
            game = {
                'home_team': t1,
                'away_team': t2,
                'week': 0,
                'gameday': 'TBD',
                'spread': 0,
                'total': 45,
            }
        
        options = []
        
        # Generate all option types
        options.extend(self.generate_game_winner_options(game))
        options.extend(self.generate_spread_options(game))
        options.extend(self.generate_total_options(game))
        options.extend(self.generate_player_yards_options(game, num_players=4))
        options.extend(self.generate_player_td_options(game, num_players=4))
        
        # Sort by confidence (highest first)
        options.sort(key=lambda x: x.confidence, reverse=True)
        
        return options
    
    def generate_week_options(self, week: int, max_per_game: int = 5) -> List[BetOption]:
        """Generate betting options for all games in a week.
        
        Args:
            week: Week number
            max_per_game: Maximum options to include per game
            
        Returns:
            List of all BetOptions sorted by confidence
        """
        games = self.get_week_games(week)
        if not games:
            logger.error(f"No games found for week {week}")
            return []
        
        all_options = []
        
        for game in games:
            game_options = []
            game_options.extend(self.generate_game_winner_options(game))
            game_options.extend(self.generate_spread_options(game))
            game_options.extend(self.generate_total_options(game))
            game_options.extend(self.generate_player_yards_options(game, num_players=2))
            game_options.extend(self.generate_player_td_options(game, num_players=2))
            
            # Sort by confidence and take top N per game
            game_options.sort(key=lambda x: x.confidence, reverse=True)
            all_options.extend(game_options[:max_per_game])
        
        # Sort all options by confidence
        all_options.sort(key=lambda x: x.confidence, reverse=True)
        
        return all_options
    
    def build_parlay(self, selected_options: List[BetOption]) -> Dict[str, Any]:
        """Build a parlay from selected options.
        
        Args:
            selected_options: List of selected BetOptions
            
        Returns:
            Parlay summary dict
        """
        if not selected_options:
            return {'success': False, 'error': 'No options selected'}
        
        # Calculate combined confidence (multiply probabilities)
        combined_confidence = 1.0
        for opt in selected_options:
            combined_confidence *= opt.confidence
        
        # Risk assessment
        if combined_confidence >= 0.25:
            risk_level = 'LOW'
        elif combined_confidence >= 0.10:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'HIGH'
        
        return {
            'success': True,
            'num_legs': len(selected_options),
            'legs': selected_options,
            'combined_confidence': combined_confidence,
            'risk_level': risk_level,
        }
    
    def display_options(self, options: List[BetOption], max_display: int = 15) -> str:
        """Format options for display.
        
        Args:
            options: List of BetOptions
            max_display: Maximum options to display
            
        Returns:
            Formatted string for display
        """
        lines = []
        lines.append("")
        lines.append(" #  | Type         | Pick                           | Conf")
        lines.append("----+--------------+--------------------------------+------")
        
        for i, opt in enumerate(options[:max_display], 1):
            type_short = {
                'game_winner': 'Game Winner',
                'spread': 'Spread',
                'total': 'Total',
                'player_yards': 'Player Yds',
                'player_td': 'Player TD',
            }.get(opt.bet_type, opt.bet_type)
            
            pick_display = opt.pick[:30]  # Truncate if too long
            lines.append(f" {i:2} | {type_short:<12} | {pick_display:<30} | {opt.confidence:.0%}")
        
        return "\n".join(lines)
    
    def display_parlay(self, parlay: Dict[str, Any]) -> str:
        """Format parlay for display.
        
        Args:
            parlay: Parlay dict from build_parlay
            
        Returns:
            Formatted string for display
        """
        if not parlay.get('success'):
            return f"Error: {parlay.get('error', 'Unknown error')}"
        
        lines = []
        lines.append("")
        lines.append(f"🎫 YOUR {parlay['num_legs']}-LEG PARLAY:")
        lines.append("━" * 40)
        
        for leg in parlay['legs']:
            lines.append(f"✓ {leg.pick} ({leg.confidence:.0%}) - {leg.game}")
        
        lines.append("")
        lines.append(f"📊 Combined Confidence: {parlay['combined_confidence']:.1%}")
        
        risk_emoji = {'LOW': '🟢', 'MEDIUM': '🟡', 'HIGH': '🔴'}.get(parlay['risk_level'], '⚪')
        lines.append(f"💡 Risk Level: {risk_emoji} {parlay['risk_level']}")
        
        return "\n".join(lines)

