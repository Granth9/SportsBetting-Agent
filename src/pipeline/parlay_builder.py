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
    
    # Manual betting lines override (Week 15, 2025) - UPDATE WEEKLY
    # Format: (week, away_team, home_team) -> {'spread': float, 'home_ml': float, 'away_ml': float, 'total': float}
    # Spread is from home team perspective: negative = home favorite, positive = away favorite
    # Example: -4.5 means home team is favored by 4.5 points
    BETTING_LINES_OVERRIDE = {
        # Week 15, 2025 - Thursday, Dec 11
        (15, 'ATL', 'TB'): {
            'spread': -4.5,  # Bucs -4.5 (home favorite)
            'home_ml': -240,  # Bucs -245 to -238 (using -240 as average)
            'away_ml': 200,   # Falcons +200 to +195 (using +200 as average)
            'total': 45.0,    # Estimate, update if you have the actual total
        },
        # Week 15, 2025 - Sunday, Dec 14
        (15, 'NYJ', 'JAX'): {
            'spread': -13.5,  # Jaguars -13.5
            'home_ml': -900,  # Jaguars -900
            'away_ml': 600,   # Jets +600
            'total': 40.0,
        },
        (15, 'CLE', 'CHI'): {
            'spread': -7.5,   # Bears -7.5
            'home_ml': -425,  # Bears -425
            'away_ml': 330,   # Browns +330
            'total': 42.0,
        },
        (15, 'BUF', 'NE'): {
            'spread': -1.5,   # Bills -1.5
            'home_ml': -120,  # Bills -115 to -120 (using -120)
            'away_ml': 100,   # Patriots +100 to -105/+100 (using +100)
            'total': 42.0,
        },
        (15, 'BAL', 'CIN'): {
            'spread': -2.5,   # Ravens -2.5
            'home_ml': -135,  # Ravens -135
            'away_ml': 114,   # Bengals +114
            'total': 45.0,
        },
        (15, 'ARI', 'HOU'): {
            'spread': -9.5,   # Texans -9.5
            'home_ml': -550,  # Texans -525 to -575 (using -550)
            'away_ml': 400,   # Cardinals +390 to +425 (using +400)
            'total': 48.0,
        },
        (15, 'LV', 'PHI'): {
            'spread': -11.5,  # Eagles -11.5
            'home_ml': -750,  # Eagles -675 to -800 (using -750)
            'away_ml': 490,   # Raiders +490
            'total': 48.0,
        },
        (15, 'LAC', 'KC'): {
            'spread': -5.5,   # Chiefs -5.5
            'home_ml': -270,  # Chiefs -250 to -285 (using -270)
            'away_ml': 205,   # Chargers +205
            'total': 50.0,
        },
        (15, 'WAS', 'NYG'): {
            'spread': -2.5,   # Giants -2.5
            'home_ml': -140,  # Giants -134 to -142 (using -140)
            'away_ml': 120,   # Commanders +120
            'total': 42.0,
        },
        (15, 'IND', 'SEA'): {
            'spread': -14.0,  # Seahawks -14
            'home_ml': -1050, # Seahawks -1050
            'away_ml': 675,   # Colts +675
            'total': 45.0,
        },
        (15, 'TEN', 'SF'): {
            'spread': -12.5,  # 49ers -12.5
            'home_ml': -950,  # 49ers -950
            'away_ml': 625,   # Titans +625
            'total': 45.0,
        },
        (15, 'GB', 'DEN'): {
            'spread': -2.5,   # Packers -2.5
            'home_ml': -135,  # Packers -135
            'away_ml': 114,   # Broncos +114
            'total': 42.0,
        },
        (15, 'DET', 'LAR'): {
            'spread': -6.0,   # Rams -6
            'home_ml': -258,  # Rams -258
            'away_ml': 210,   # Lions +210
            'total': 48.0,
        },
        (15, 'CAR', 'NO'): {
            'spread': 2.5,    # Panthers -2.5 means away favorite, so home spread is +2.5
            'home_ml': 120,   # Saints +120
            'away_ml': -142,  # Panthers -142
            'total': 42.0,
        },
        (15, 'MIN', 'DAL'): {
            'spread': -5.5,   # Cowboys -5.5
            'home_ml': -270,  # Cowboys -270
            'away_ml': 220,   # Vikings +220
            'total': 48.0,
        },
        # Week 15, 2025 - Monday, Dec 15
        (15, 'MIA', 'PIT'): {
            'spread': -3.0,   # Steelers -3
            'home_ml': -177,  # Steelers -175 to -180 (using -177)
            'away_ml': 147,   # Dolphins +145 to +150 (using +147)
            'total': 45.0,
        },
    }
    
    # Known injured players (as of Week 14-15, 2025) - UPDATE WEEKLY
    # Format: player name -> status ('OUT', 'DOUBTFUL', 'QUESTIONABLE', 'IR')
    # Players with 'OUT', 'DOUBTFUL', or 'IR' will be excluded from parlay options
    # QUESTIONABLE players are still included (they may play)
    # 
    # To update: Check NFL.com injury reports, ESPN injury tracker, or team injury reports
    # Update this list before each week's games
    INJURED_PLAYERS = {
        # Season-ending injuries (IR) - Excluded from all parlays
        'Zach Ertz': 'IR',  # Torn ACL, season over (Washington)
        'Daniel Jones': 'IR',  # Achilles injury, season over (Indianapolis)
        'Trey Hendrickson': 'IR',  # Core muscle surgery, out for season (Cincinnati)
        'Kyu Blu Kelly': 'IR',  # Knee injury (Las Vegas)
        'J.K. Dobbins': 'IR',  # Out for season (Denver)
        
        # Currently OUT - Excluded from parlays
        'Nick Chubb': 'OUT',  # Rib injury, ruled out (Houston)
        'Geno Smith': 'OUT',  # Shoulder injury (Las Vegas)
        'Trent McDuffie': 'OUT',  # Knee injury (Kansas City)
        'Wanya Morris': 'OUT',  # Knee injury (Kansas City)
        'Drake London': 'OUT',  # Out (Atlanta)
        'Garrett Wilson': 'IR',  # On IR (New York Jets)
        
        # QUESTIONABLE - Still included (may play)
        # 'Cade Otton': 'QUESTIONABLE',  # Knee injury (Tampa Bay) - may play
        
        # IMPORTANT: Update this list weekly before building parlays
        # Check these sources for current injury reports:
        # - https://www.nfl.com/injuries/
        # - https://www.espn.com/nfl/injuries
        # - Team-specific injury reports (released Wed/Thu/Fri)
        # - NFL.com transaction wire for IR placements
    }
    
    # Minimum reasonable lines by position (to avoid unrealistic props)
    MIN_LINES = {
        'rushing_yards': 15,  # No RB should have under 15 yards unless confirmed out
        'receiving_yards': 10,  # No WR/TE should have under 10 yards unless confirmed out
        'passing_yards': 150,  # No QB should have under 150 yards unless confirmed out
        'total_yards': 20,  # Combined rushing + receiving
    }
    
    # Key players per team (top skill position players) - 2025 SEASON ROSTERS
    TEAM_KEY_PLAYERS = {
        'PHI': ['A.J. Brown', 'Devonta Smith', 'Saquon Barkley', 'Jalen Hurts', 'Dallas Goedert'],
        'LAC': ['Ladd McConkey', 'Quentin Johnston', 'Gus Edwards', 'Justin Herbert', 'Omarion Hampton'],
        'KC': ['Travis Kelce', 'Xavier Worthy', 'Isiah Pacheco', 'Patrick Mahomes', 'DeAndre Hopkins'],
        'BUF': ['Keon Coleman', 'James Cook', 'Dalton Kincaid', 'Josh Allen', 'Khalil Shakir'],
        'SF': ['Deebo Samuel', 'George Kittle', 'Christian McCaffrey', 'Brock Purdy'],
        'DAL': ['CeeDee Lamb', 'Rico Dowdle', 'Jake Ferguson', 'Cooper Rush', 'George Pickens'],
        'MIA': ['Tyreek Hill', 'Jaylen Waddle', 'De\'Von Achane', 'Tua Tagovailoa'],
        'DET': ['Amon-Ra St. Brown', 'Jahmyr Gibbs', 'Sam LaPorta', 'Jared Goff', 'Jameson Williams'],
        'BAL': ['Zay Flowers', 'Mark Andrews', 'Derrick Henry', 'Lamar Jackson'],
        'CIN': ['Ja\'Marr Chase', 'Tee Higgins', 'Chase Brown', 'Joe Burrow'],
        'MIN': ['Justin Jefferson', 'Jordan Addison', 'Aaron Jones', 'Sam Darnold'],
        'GB': ['Jayden Reed', 'Romeo Doubs', 'Josh Jacobs', 'Jordan Love'],
        'NYJ': ['Garrett Wilson', 'Davante Adams', 'Breece Hall', 'Aaron Rodgers'],
        'DEN': ['Courtland Sutton', 'Bo Nix', 'Marvin Mims Jr', 'Javonte Williams', 'R.J. Harvey'],
        'HOU': ['Nico Collins', 'Tank Dell', 'Stefon Diggs', 'Joe Mixon', 'C.J. Stroud'],
        'CLE': ['Amari Cooper', 'Jerry Jeudy', 'David Njoku', 'Deshaun Watson', 'Jerome Ford'],
        'PIT': ['Najee Harris', 'Pat Freiermuth', 'Russell Wilson'],
        'SEA': ['DK Metcalf', 'Jaxon Smith-Njigba', 'Kenneth Walker', 'Cooper Kupp', 'Sam Howell'],
        'TB': ['Mike Evans', 'Chris Godwin', 'Bucky Irving', 'Baker Mayfield', 'Emeka Egbuka'],
        'NO': ['Chris Olave', 'Alvin Kamara', 'Derek Carr'],
        'ATL': ['Drake London', 'Bijan Robinson', 'Kyle Pitts', 'Kirk Cousins', 'Darnell Mooney'],
        'CAR': ['Adam Thielen', 'Chuba Hubbard', 'Bryce Young', 'Xavier Legette', 'Tetairoa McMillan'],
        'LV': ['Jakobi Meyers', 'Brock Bowers', 'Alexander Mattison', 'Aidan O\'Connell'],
        'ARI': ['Marvin Harrison Jr', 'Trey McBride', 'James Conner', 'Kyler Murray'],
        'LAR': ['Puka Nacua', 'Kyren Williams', 'Matthew Stafford', 'Terrance Ferguson'],
        'NYG': ['Malik Nabers', 'Wan\'Dale Robinson', 'Tyrone Tracy Jr', 'Jaxson Dart'],
        'CHI': ['D.J. Moore', 'Rome Odunze', 'D\'Andre Swift', 'Caleb Williams'],
        'WAS': ['Terry McLaurin', 'Brian Robinson Jr', 'Jayden Daniels', 'Jahan Dotson'],
        'JAX': ['Brian Thomas Jr', 'Travis Etienne', 'Evan Engram', 'Mac Jones', 'Travis Hunter'],
        'TEN': ['Calvin Ridley', 'Tony Pollard', 'Will Levis', 'Chig Okonkwo'],
        'IND': ['Michael Pittman Jr', 'Jonathan Taylor', 'Anthony Richardson'],
        'NE': ['Rhamondre Stevenson', 'Hunter Henry', 'Drake Maye', 'TreVeyon Henderson'],
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
        'buccaneers': 'TB', 'bucaneers': 'TB', 'tampa bay': 'TB', 'tampa bay buccaneers': 'TB', 'bucs': 'TB',
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
        self.market_line_estimator = None
        
    def _get_player_prop_predictor(self):
        """Lazy load player prop predictor."""
        if self.player_prop_predictor is None:
            from src.pipeline.quick_predict import PlayerPropPredictor
            self.player_prop_predictor = PlayerPropPredictor()
        return self.player_prop_predictor
    
    def _get_market_line_estimator(self):
        """Lazy load market line estimator."""
        if self.market_line_estimator is None:
            from src.data.collectors.market_line_estimator import MarketLineEstimator
            self.market_line_estimator = MarketLineEstimator()
        return self.market_line_estimator
    
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
    
    def _get_game_spread(self, game: Dict[str, Any]) -> float:
        """Get game spread with manual override support.
        
        Args:
            game: Game dict with home_team, away_team, week, and optionally spread_line
            
        Returns:
            Spread from home team perspective (negative = home favorite)
        """
        week = game.get('week', 0)
        away = game.get('away_team', '')
        home = game.get('home_team', '')
        
        # Check for manual override first
        override_key = (week, away, home)
        if override_key in self.BETTING_LINES_OVERRIDE:
            return self.BETTING_LINES_OVERRIDE[override_key].get('spread', 0)
        
        # Fallback to schedule data
        return game.get('spread_line', 0) if pd.notna(game.get('spread_line', 0)) else 0
    
    def _get_game_moneylines(self, game: Dict[str, Any]) -> Tuple[float, float]:
        """Get game moneylines with manual override support.
        
        Args:
            game: Game dict with home_team, away_team, week
            
        Returns:
            Tuple of (home_ml, away_ml)
        """
        week = game.get('week', 0)
        away = game.get('away_team', '')
        home = game.get('home_team', '')
        
        # Check for manual override first
        override_key = (week, away, home)
        if override_key in self.BETTING_LINES_OVERRIDE:
            override = self.BETTING_LINES_OVERRIDE[override_key]
            return override.get('home_ml', -110), override.get('away_ml', -110)
        
        # Fallback to schedule data
        home_ml = game.get('home_moneyline', -110) if pd.notna(game.get('home_moneyline', -110)) else -110
        away_ml = game.get('away_moneyline', -110) if pd.notna(game.get('away_moneyline', -110)) else -110
        return home_ml, away_ml
    
    def parse_parlay_query(self, query: str) -> Dict[str, Any]:
        """Parse parlay query to extract parameters.
        
        Args:
            query: Natural language query like "4 leg parlay Eagles vs Chargers"
            
        Returns:
            Dictionary with legs, teams, week
        """
        query_lower = query.lower()
        
        # Extract number of legs (2-6 legs supported)
        leg_patterns = [
            r'(\d+)[\s-]*(?:leg|legs)',  # Matches "6 leg", "6-leg", "6leg"
            r'(\d+)\s*(?:pick|picks)',
            r'build\s*(?:a\s*)?(\d+)',
        ]
        num_legs = 4  # default
        for pattern in leg_patterns:
            match = re.search(pattern, query_lower)
            if match:
                parsed_legs = int(match.group(1))
                # Validate: must be between 2 and 6
                if 2 <= parsed_legs <= 6:
                    num_legs = parsed_legs
                else:
                    # Clamp to valid range
                    num_legs = max(2, min(6, parsed_legs))
                    logger.warning(f"Parlay legs must be between 2-6. Using {num_legs} legs.")
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
        week = int(game['week'])
        home_team = game['home_team']
        away_team = game['away_team']
        
        # Get spreads and moneylines with override support
        game_dict = {
            'week': week,
            'home_team': home_team,
            'away_team': away_team,
            'spread_line': game.get('spread_line', 0),
            'home_moneyline': game.get('home_moneyline', -110),
            'away_moneyline': game.get('away_moneyline', -110),
        }
        
        spread = self._get_game_spread(game_dict)
        home_ml, away_ml = self._get_game_moneylines(game_dict)
        
        return {
            'home_team': home_team,
            'away_team': away_team,
            'week': week,
            'gameday': game.get('gameday', 'TBD'),
            'spread': spread,
            'total': game.get('total_line', 45) if pd.notna(game.get('total_line')) else 45,
            'home_ml': home_ml,
            'away_ml': away_ml,
        }
    
    def get_week_winners(self, week: int, season: int = 2025) -> List[Dict[str, Any]]:
        """Get all projected winners for a given week.
        
        Args:
            week: Week number
            season: Season year
            
        Returns:
            List of winner predictions with confidence
        """
        games = self.get_week_games(week)
        if not games:
            return []
        
        # Get quick predictor for game predictions
        qp = self._get_quick_predictor()
        
        winners = []
        
        for game in games:
            home = game['home_team']
            away = game['away_team']
            game_str = f"{away} @ {home}"
            
            # Make prediction for this game
            try:
                # Create query string for the game
                home_full = self.TEAM_FULL_NAMES.get(home, home)
                away_full = self.TEAM_FULL_NAMES.get(away, away)
                query = f"{away_full} vs {home_full}"
                
                result = qp.predict(query)
                
                if result.get('success') and not result.get('completed'):
                    predicted_winner = result.get('prediction')
                    confidence = result.get('confidence', 0.5)
                    # Get spread with override support
                    spread = self._get_game_spread({
                        'week': week,
                        'home_team': home,
                        'away_team': away,
                        'spread_line': result.get('spread', 0),
                    })
                    
                    winner_full = self.TEAM_FULL_NAMES.get(predicted_winner, predicted_winner)
                    loser = away if predicted_winner == home else home
                    loser_full = self.TEAM_FULL_NAMES.get(loser, loser)
                    
                    winners.append({
                        'game': game_str,
                        'home_team': home,
                        'away_team': away,
                        'home_team_full': home_full,
                        'away_team_full': away_full,
                        'predicted_winner': predicted_winner,
                        'predicted_winner_full': winner_full,
                        'loser': loser,
                        'loser_full': loser_full,
                        'confidence': confidence,
                        'spread': spread,
                        'gameday': game.get('gameday', 'TBD'),
                    })
            except Exception as e:
                logger.debug(f"Error predicting {game_str}: {e}")
                continue
        
        # Sort by confidence (highest first)
        winners.sort(key=lambda x: x['confidence'], reverse=True)
        
        return winners
    
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
        
        def normalize_team_abbr(abbr: str) -> str:
            """Convert schedule abbreviations to our standard format."""
            if abbr == 'LA':  # Schedule uses 'LA' for Rams
                return 'LAR'
            return abbr
        
        for _, game_row in week_games.iterrows():
            home_team = normalize_team_abbr(game_row['home_team'])
            away_team = normalize_team_abbr(game_row['away_team'])
            week_num = int(game_row['week'])
            
            # Get spreads and moneylines with override support
            game_dict = {
                'week': week_num,
                'home_team': home_team,
                'away_team': away_team,
                'spread_line': game_row.get('spread_line', 0),
                'home_moneyline': game_row.get('home_moneyline', -110),
                'away_moneyline': game_row.get('away_moneyline', -110),
            }
            
            spread = self._get_game_spread(game_dict)
            home_ml, away_ml = self._get_game_moneylines(game_dict)
            
            games.append({
                'home_team': home_team,
                'away_team': away_team,
                'week': week_num,
                'gameday': game_row.get('gameday', 'TBD'),
                'spread': spread,
                'total': game_row.get('total_line', 45) if pd.notna(game_row.get('total_line')) else 45,
                'home_ml': home_ml,
                'away_ml': away_ml,
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
    
    def _is_player_injured(self, player_name: str) -> bool:
        """Check if a player is injured and should be excluded.
        
        Args:
            player_name: Player name
            
        Returns:
            True if player is injured (OUT, DOUBTFUL, IR), False otherwise
        """
        # Check exact match first
        injury_status = self.INJURED_PLAYERS.get(player_name, '').upper()
        if injury_status in ['OUT', 'DOUBTFUL', 'IR']:
            return True
        
        # Check case-insensitive match
        player_lower = player_name.lower()
        for injured_player, status in self.INJURED_PLAYERS.items():
            if injured_player.lower() == player_lower:
                if status.upper() in ['OUT', 'DOUBTFUL', 'IR']:
                    return True
        
        return False
    
    @classmethod
    def add_injured_player(cls, player_name: str, status: str) -> None:
        """Add or update an injured player.
        
        Args:
            player_name: Player name (exact match required)
            status: Injury status ('OUT', 'DOUBTFUL', 'QUESTIONABLE', 'IR')
        """
        cls.INJURED_PLAYERS[player_name] = status.upper()
        logger.info(f"Added injured player: {player_name} - {status}")
    
    @classmethod
    def remove_injured_player(cls, player_name: str) -> None:
        """Remove a player from the injured list (they've recovered).
        
        Args:
            player_name: Player name to remove
        """
        if player_name in cls.INJURED_PLAYERS:
            del cls.INJURED_PLAYERS[player_name]
            logger.info(f"Removed injured player: {player_name} (recovered)")
    
    @classmethod
    def get_injured_players(cls) -> Dict[str, str]:
        """Get current list of injured players.
        
        Returns:
            Dictionary of player name -> injury status
        """
        return cls.INJURED_PLAYERS.copy()
    
    def _validate_player_stats(self, result: Dict[str, Any], stat_type: str) -> bool:
        """Validate that player stats are reasonable.
        
        Args:
            result: Prediction result from player prop predictor
            stat_type: Type of stat
            
        Returns:
            True if stats are valid, False otherwise
        """
        if not result.get('success'):
            return False
        
        # Check if player has recent games
        games_analyzed = result.get('games_analyzed', 0)
        if games_analyzed < 3:
            logger.debug(f"Player has too few games: {games_analyzed}")
            return False
        
        # Check average yards is reasonable and positive
        avg_yards = result.get('avg_yards', 0)
        median_yards = result.get('median_yards', avg_yards)
        
        # Use median if available, otherwise average
        base_yards = median_yards if median_yards and median_yards > 0 else avg_yards
        
        # CRITICAL: Must be positive
        if base_yards <= 0:
            logger.debug(f"Player has non-positive yards: {base_yards}")
            return False
        
        min_line = self.MIN_LINES.get(stat_type, 10)
        
        # If average/median is below minimum, only allow if it's an UNDER bet with very high confidence
        if base_yards < min_line:
            prediction = result.get('prediction', '')
            confidence = result.get('confidence', 0)
            # Only allow if it's UNDER with very high confidence (player confirmed out/inactive)
            if prediction == 'UNDER' and confidence >= 0.9:
                # Still ensure it's at least 5 yards (absolute minimum)
                if base_yards < 5:
                    logger.debug(f"Player yards {base_yards} too low even for high-confidence UNDER")
                    return False
                return True
            logger.debug(f"Player average {base_yards} below minimum {min_line} for {stat_type}")
            return False
        
        return True
    
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
                    # Skip injured players
                    if self._is_player_injured(player):
                        logger.debug(f"Skipping injured player: {player}")
                        continue
                    
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
                    elif any(rb == player for rb in rb_full_names):
                        stat_type = 'rushing_yards'
                    else:
                        stat_type = 'receiving_yards'
                    
                    # Determine opponent team
                    opponent = away if team == home else home
                    
                    # Get market-consistent line estimate
                    estimator = self._get_market_line_estimator()
                    try:
                        # Get player stats first for better line estimation
                        player_stats = pp.get_player_history(player, stat_type)
                        estimated_line = estimator.estimate_player_line(
                            player_name=player,
                            stat_type=stat_type,
                            opponent_team=opponent,
                            season=2025,
                            player_stats=player_stats if player_stats.get('found') else None
                        )
                    except Exception as e:
                        logger.debug(f"Error estimating line for {player}: {e}")
                        continue
                    
                    # Validate estimated line is reasonable
                    min_line = self.MIN_LINES.get(stat_type, 10)
                    if estimated_line <= 0 or estimated_line < min_line:
                        logger.debug(f"Invalid estimated line ({estimated_line}) for {player}, skipping")
                        continue
                    
                    # Get model prediction using the estimated line
                    result = pp.predict_over_under(player, estimated_line, stat_type)
                    
                    # Validate stats are reasonable
                    if not self._validate_player_stats(result, stat_type):
                        logger.debug(f"Invalid stats for {player}, skipping")
                        continue
                    
                    if result.get('success'):
                        prediction = result.get('prediction', 'OVER')
                        confidence = result.get('confidence', 0.5)
                        
                        # Only generate options with reasonable confidence
                        if confidence < 0.45:
                            logger.debug(f"Low confidence ({confidence:.0%}) for {player}, skipping")
                            continue
                        
                        # Use the estimated market line
                        line = estimated_line
                        
                        # Final validation: ensure line is positive and reasonable
                        if line <= 0:
                            logger.debug(f"Invalid line ({line}) for {player}, skipping")
                            continue
                        
                        if line < min_line:
                            # Only allow very low UNDER lines if confidence is very high (player confirmed out)
                            if prediction == 'UNDER' and confidence >= 0.9:
                                # Allow it, but ensure it's at least 5 yards
                                line = max(5, line)
                            else:
                                logger.debug(f"Line {line} below minimum {min_line} for {stat_type}, skipping")
                                continue
                        
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
                # Skip injured players
                if self._is_player_injured(player):
                    logger.debug(f"Skipping injured player for TD: {player}")
                    continue
                
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
                        games_analyzed = result.get('games_analyzed', 0)
                        
                        # Validate: need at least 3 games of data
                        if games_analyzed < 3:
                            logger.debug(f"Insufficient games ({games_analyzed}) for {player} TD prop")
                            continue
                        
                        # Only include YES predictions (anytime TD bets)
                        # Sportsbooks don't offer "No TD" as a betting option
                        if prediction == 'YES' and td_rate >= 0.3 and confidence >= 0.45:
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
            selected_options: List of selected BetOptions (2-6 legs supported)
            
        Returns:
            Parlay summary dict
        """
        if not selected_options:
            return {'success': False, 'error': 'No options selected'}
        
        num_legs = len(selected_options)
        
        # Validate number of legs (2-6 supported)
        if num_legs < 2:
            return {'success': False, 'error': f'Parlay must have at least 2 legs (got {num_legs})'}
        if num_legs > 6:
            return {'success': False, 'error': f'Parlay can have at most 6 legs (got {num_legs})'}
        
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

