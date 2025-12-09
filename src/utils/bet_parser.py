"""Natural language bet parser for manual bet input."""

import re
from typing import Optional, Dict, Tuple
from src.utils.data_types import BetType, Proposition, GameInfo, BettingLine
from src.utils.logger import setup_logger


logger = setup_logger(__name__)


class BetParser:
    """Parser for natural language bet descriptions."""
    
    # Comprehensive team name mapping
    TEAM_MAPPING = {
        # Full team names
        'miami dolphins': 'MIA',
        'new york jets': 'NYJ',
        'new york giants': 'NYG',
        'new england patriots': 'NE',
        'buffalo bills': 'BUF',
        'kansas city chiefs': 'KC',
        'dallas cowboys': 'DAL',
        'philadelphia eagles': 'PHI',
        'green bay packers': 'GB',
        'chicago bears': 'CHI',
        'detroit lions': 'DET',
        'minnesota vikings': 'MIN',
        'tampa bay buccaneers': 'TB',
        'new orleans saints': 'NO',
        'atlanta falcons': 'ATL',
        'carolina panthers': 'CAR',
        'seattle seahawks': 'SEA',
        'san francisco 49ers': 'SF',
        'los angeles rams': 'LAR',
        'arizona cardinals': 'ARI',
        'baltimore ravens': 'BAL',
        'cincinnati bengals': 'CIN',
        'cleveland browns': 'CLE',
        'pittsburgh steelers': 'PIT',
        'houston texans': 'HOU',
        'indianapolis colts': 'IND',
        'jacksonville jaguars': 'JAX',
        'tennessee titans': 'TEN',
        'denver broncos': 'DEN',
        'las vegas raiders': 'LV',
        'los angeles chargers': 'LAC',
        'washington commanders': 'WAS',
        # Team names only (without city)
        'dolphins': 'MIA',
        'jets': 'NYJ',
        'giants': 'NYG',
        'patriots': 'NE',
        'bills': 'BUF',
        'chiefs': 'KC',
        'cowboys': 'DAL',
        'eagles': 'PHI',
        'packers': 'GB',
        'bears': 'CHI',
        'lions': 'DET',
        'vikings': 'MIN',
        'buccaneers': 'TB',
        'saints': 'NO',
        'falcons': 'ATL',
        'panthers': 'CAR',
        'seahawks': 'SEA',
        '49ers': 'SF',
        'rams': 'LAR',
        'cardinals': 'ARI',
        'ravens': 'BAL',
        'bengals': 'CIN',
        'browns': 'CLE',
        'steelers': 'PIT',
        'texans': 'HOU',
        'colts': 'IND',
        'jaguars': 'JAX',
        'titans': 'TEN',
        'broncos': 'DEN',
        'raiders': 'LV',
        'chargers': 'LAC',
        'commanders': 'WAS',
        # City names
        'miami': 'MIA',
        'new york': 'NYJ',  # Ambiguous, but Jets is more common
        'buffalo': 'BUF',
        'kansas city': 'KC',
        'dallas': 'DAL',
        'philadelphia': 'PHI',
        'green bay': 'GB',
        'chicago': 'CHI',
        'detroit': 'DET',
        'minnesota': 'MIN',
        'tampa': 'TB',
        'tampa bay': 'TB',
        'new orleans': 'NO',
        'atlanta': 'ATL',
        'carolina': 'CAR',
        'seattle': 'SEA',
        'san francisco': 'SF',
        'los angeles': 'LAR',  # Ambiguous
        'arizona': 'ARI',
        'baltimore': 'BAL',
        'cincinnati': 'CIN',
        'cleveland': 'CLE',
        'pittsburgh': 'PIT',
        'houston': 'HOU',
        'indianapolis': 'IND',
        'jacksonville': 'JAX',
        'tennessee': 'TEN',
        'denver': 'DEN',
        'las vegas': 'LV',
        'washington': 'WAS',
    }
    
    # Stat type mapping
    STAT_TYPE_MAPPING = {
        'touchdown': 'touchdowns',
        'touchdowns': 'touchdowns',
        'td': 'touchdowns',
        'tds': 'touchdowns',
        'score': 'touchdowns',
        'scores': 'touchdowns',
        'passing yards': 'passing_yards',
        'pass yards': 'passing_yards',
        'passing': 'passing_yards',
        'receiving yards': 'receiving_yards',
        'rec yards': 'receiving_yards',
        'receiving': 'receiving_yards',
        'rushing yards': 'rushing_yards',
        'rush yards': 'rushing_yards',
        'rushing': 'rushing_yards',
        'yards': 'receiving_yards',  # Default assumption
    }
    
    def __init__(self):
        """Initialize the bet parser."""
        pass
    
    def parse_bet_description(
        self,
        bet_description: str,
        game_info: GameInfo,
        home_team: str,
        away_team: str
    ) -> Tuple[Proposition, Optional[str]]:
        """Parse a natural language bet description into a Proposition.
        
        Args:
            bet_description: Natural language bet description
            game_info: Game information
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            
        Returns:
            Tuple of (Proposition, error_message). error_message is None if successful.
        """
        bet_description = bet_description.strip()
        bet_lower = bet_description.lower()
        
        # Try to parse different bet types
        # Order matters: more specific patterns first
        
        # 1. Player props
        prop, error = self._parse_player_prop(bet_description, bet_lower, game_info, home_team, away_team)
        if prop:
            return prop, None
        
        # 2. Spread bets
        prop, error = self._parse_spread(bet_description, bet_lower, game_info, home_team, away_team)
        if prop:
            return prop, None
        
        # 3. Total bets
        prop, error = self._parse_total(bet_description, bet_lower, game_info, home_team, away_team)
        if prop:
            return prop, None
        
        # 4. Game outcome bets
        prop, error = self._parse_game_outcome(bet_description, bet_lower, game_info, home_team, away_team)
        if prop:
            return prop, None
        
        # If we get here, couldn't parse
        return None, f"Could not parse bet description: '{bet_description}'. Please try a clearer format."
    
    def _parse_game_outcome(
        self,
        bet_description: str,
        bet_lower: str,
        game_info: GameInfo,
        home_team: str,
        away_team: str
    ) -> Tuple[Optional[Proposition], Optional[str]]:
        """Parse game outcome bets.
        
        Patterns:
        - "{team} will win" / "{team} wins"
        - "Who will win: {team1} or {team2}"
        - "{team1} vs {team2} winner"
        """
        # Pattern 1: "Who will win: {team1} or {team2}" (check this first to avoid matching "who" as a team)
        who_win_pattern = r'who\s+will\s+win[:\s]+([a-z]+(?:\s+[a-z]+)?)\s+or\s+([a-z]+(?:\s+[a-z]+)?)[\?]?'
        match = re.search(who_win_pattern, bet_lower)
        if match:
            team1_name = match.group(1).strip()
            team2_name = match.group(2).strip()
            team1 = self._normalize_team_name(team1_name)
            team2 = self._normalize_team_name(team2_name)
            
            if team1 and team2:
                # Check which team is in the game - prefer the first one mentioned
                if team1.upper() in [home_team.upper(), away_team.upper()]:
                    selected_team = team1.upper()
                elif team2.upper() in [home_team.upper(), away_team.upper()]:
                    selected_team = team2.upper()
                else:
                    return None, f"Neither team found in game ({home_team} vs {away_team})"
                
                prop = Proposition(
                    prop_id=f"{game_info.game_id}_game_outcome_{selected_team}",
                    game_info=game_info,
                    bet_type=BetType.GAME_OUTCOME,
                    metadata={'selected_team': selected_team, 'description': bet_description}
                )
                return prop, None
            elif team1:
                # Only team1 found, use it
                if team1.upper() in [home_team.upper(), away_team.upper()]:
                    selected_team = team1.upper()
                    prop = Proposition(
                        prop_id=f"{game_info.game_id}_game_outcome_{selected_team}",
                        game_info=game_info,
                        bet_type=BetType.GAME_OUTCOME,
                        metadata={'selected_team': selected_team, 'description': bet_description}
                    )
                    return prop, None
            elif team2:
                # Only team2 found, use it
                if team2.upper() in [home_team.upper(), away_team.upper()]:
                    selected_team = team2.upper()
                    prop = Proposition(
                        prop_id=f"{game_info.game_id}_game_outcome_{selected_team}",
                        game_info=game_info,
                        bet_type=BetType.GAME_OUTCOME,
                        metadata={'selected_team': selected_team, 'description': bet_description}
                    )
                    return prop, None
        
        # Pattern 2: "{team} will win" or "{team} wins"
        # Match team name (can be multiple words) followed by "will win" or "wins"
        # But NOT "who will win" (already handled above)
        win_pattern = r'^(?!who\s+will\s+win)([a-z\s]+?)\s+(?:will\s+)?win'
        match = re.search(win_pattern, bet_lower)
        if match:
            team_name = match.group(1).strip()
            team_abbr = self._normalize_team_name(team_name)
            if team_abbr:
                # Determine if it's home or away
                if team_abbr.upper() == home_team.upper():
                    selected_team = home_team
                elif team_abbr.upper() == away_team.upper():
                    selected_team = away_team
                else:
                    return None, f"Team '{team_name}' not found in game ({home_team} vs {away_team})"
                
                prop = Proposition(
                    prop_id=f"{game_info.game_id}_game_outcome_{selected_team}",
                    game_info=game_info,
                    bet_type=BetType.GAME_OUTCOME,
                    metadata={'selected_team': selected_team, 'description': bet_description}
                )
                return prop, None
        
        # Pattern 3: "{team1} vs {team2} winner" or "{team1} or {team2}"
        who_win_pattern = r'who\s+will\s+win[:\s]+([a-z]+(?:\s+[a-z]+)?)\s+or\s+([a-z]+(?:\s+[a-z]+)?)[\?]?'
        match = re.search(who_win_pattern, bet_lower)
        if match:
            team1_name = match.group(1).strip()
            team2_name = match.group(2).strip()
            team1 = self._normalize_team_name(team1_name)
            team2 = self._normalize_team_name(team2_name)
            
            if team1 and team2:
                # Check which team is in the game - prefer the first one mentioned
                if team1.upper() in [home_team.upper(), away_team.upper()]:
                    selected_team = team1.upper()
                elif team2.upper() in [home_team.upper(), away_team.upper()]:
                    selected_team = team2.upper()
                else:
                    return None, f"Neither team found in game ({home_team} vs {away_team})"
                
                prop = Proposition(
                    prop_id=f"{game_info.game_id}_game_outcome_{selected_team}",
                    game_info=game_info,
                    bet_type=BetType.GAME_OUTCOME,
                    metadata={'selected_team': selected_team, 'description': bet_description}
                )
                return prop, None
            elif team1:
                # Only team1 found, use it
                if team1.upper() in [home_team.upper(), away_team.upper()]:
                    selected_team = team1.upper()
                    prop = Proposition(
                        prop_id=f"{game_info.game_id}_game_outcome_{selected_team}",
                        game_info=game_info,
                        bet_type=BetType.GAME_OUTCOME,
                        metadata={'selected_team': selected_team, 'description': bet_description}
                    )
                    return prop, None
            elif team2:
                # Only team2 found, use it
                if team2.upper() in [home_team.upper(), away_team.upper()]:
                    selected_team = team2.upper()
                    prop = Proposition(
                        prop_id=f"{game_info.game_id}_game_outcome_{selected_team}",
                        game_info=game_info,
                        bet_type=BetType.GAME_OUTCOME,
                        metadata={'selected_team': selected_team, 'description': bet_description}
                    )
                    return prop, None
        
        # Pattern 4: "{team1} vs {team2} winner" or "{team1} or {team2}"
        vs_pattern = r'(\w+(?:\s+\w+)?)\s+(?:vs|or)\s+(\w+(?:\s+\w+)?)'
        match = re.search(vs_pattern, bet_lower)
        if match and 'winner' in bet_lower:
            team1 = self._normalize_team_name(match.group(1).strip())
            team2 = self._normalize_team_name(match.group(2).strip())
            
            if team1 and team2:
                # Determine which team is selected (default to first mentioned)
                if team1.upper() in [home_team.upper(), away_team.upper()]:
                    selected_team = team1.upper()
                else:
                    selected_team = team2.upper()
                
                prop = Proposition(
                    prop_id=f"{game_info.game_id}_game_outcome_{selected_team}",
                    game_info=game_info,
                    bet_type=BetType.GAME_OUTCOME,
                    metadata={'selected_team': selected_team, 'description': bet_description}
                )
                return prop, None
        
        return None, None
    
    def _parse_player_prop(
        self,
        bet_description: str,
        bet_lower: str,
        game_info: GameInfo,
        home_team: str,
        away_team: str
    ) -> Tuple[Optional[Proposition], Optional[str]]:
        """Parse player prop bets.
        
        Patterns:
        - "Will {player} {action}?" (e.g., "Will Cortland Sutton score a touchdown?")
        - "{player} {over/under} {value} {stat}" (e.g., "Sutton over 75.5 receiving yards")
        - "{player} {stat}" (e.g., "Sutton touchdown")
        """
        # Pattern 1: "Will {player} {action}?"
        will_pattern = r'will\s+([a-z\s]+?)\s+(?:get|score|have|catch|throw|rush|run)\s+(?:a\s+)?(touchdown|td|passing\s+yards?|receiving\s+yards?|rushing\s+yards?)'
        match = re.search(will_pattern, bet_lower)
        if match:
            player_name = match.group(1).strip()
            stat_desc = match.group(2).strip()
            stat_type = self._extract_stat_type(stat_desc)
            
            # For touchdowns, line_value is typically 0.5
            line_value = 0.5 if stat_type == 'touchdowns' else None
            
            prop = Proposition(
                prop_id=f"{game_info.game_id}_player_prop_{player_name.replace(' ', '_')}",
                game_info=game_info,
                bet_type=BetType.PLAYER_PROP,
                player_name=player_name.title(),
                stat_type=stat_type,
                line_value=line_value,
                metadata={'description': bet_description}
            )
            return prop, None
        
        # Pattern 2: "{player} {over/under} {value} {stat}"
        over_under_pattern = r'([a-z\s]+?)\s+(over|under)\s+(\d+\.?\d*)\s+(passing|receiving|rushing)?\s*(yards?)'
        match = re.search(over_under_pattern, bet_lower)
        if match:
            player_name = match.group(1).strip()
            direction = match.group(2).strip()
            value = float(match.group(3))
            stat_prefix = match.group(4).strip() if match.group(4) else ''
            stat_type = self._extract_stat_type(f"{stat_prefix} {match.group(5) if match.group(5) else ''}")
            
            prop = Proposition(
                prop_id=f"{game_info.game_id}_player_prop_{player_name.replace(' ', '_')}",
                game_info=game_info,
                bet_type=BetType.PLAYER_PROP,
                player_name=player_name.title(),
                stat_type=stat_type,
                line_value=value,
                metadata={'direction': direction, 'description': bet_description}
            )
            return prop, None
        
        # Pattern 3: "{player} {stat}" (simple format)
        simple_prop_pattern = r'([a-z\s]+?)\s+(touchdown|td|passing\s+yards?|receiving\s+yards?|rushing\s+yards?)'
        match = re.search(simple_prop_pattern, bet_lower)
        if match:
            player_name = match.group(1).strip()
            stat_desc = match.group(2).strip()
            stat_type = self._extract_stat_type(stat_desc)
            
            # Check if it's a simple "player touchdown" bet
            if stat_type == 'touchdowns':
                line_value = 0.5
            else:
                # Need a value for yards - can't parse without it
                return None, None
            
            prop = Proposition(
                prop_id=f"{game_info.game_id}_player_prop_{player_name.replace(' ', '_')}",
                game_info=game_info,
                bet_type=BetType.PLAYER_PROP,
                player_name=player_name.title(),
                stat_type=stat_type,
                line_value=line_value,
                metadata={'description': bet_description}
            )
            return prop, None
        
        return None, None
    
    def _parse_spread(
        self,
        bet_description: str,
        bet_lower: str,
        game_info: GameInfo,
        home_team: str,
        away_team: str
    ) -> Tuple[Optional[Proposition], Optional[str]]:
        """Parse spread bets.
        
        Patterns:
        - "{team} {+/-}{value}" (e.g., "Dolphins -3.5")
        - "{team} covers {+/-}{value}"
        """
        # Pattern: "{team} {+/-}{value}" or "{team} covers {+/-}{value}"
        spread_pattern = r'(\w+(?:\s+\w+)?)\s+(?:covers\s+)?([+-]?\d+\.?\d*)'
        match = re.search(spread_pattern, bet_lower)
        if match:
            team_name = match.group(1).strip()
            spread_value = float(match.group(2))
            team_abbr = self._normalize_team_name(team_name)
            
            if team_abbr:
                # Determine if it's home or away
                if team_abbr.upper() == home_team.upper():
                    selected_team = home_team
                elif team_abbr.upper() == away_team.upper():
                    selected_team = away_team
                else:
                    return None, f"Team '{team_name}' not found in game ({home_team} vs {away_team})"
                
                # Create betting line
                betting_line = BettingLine(
                    spread=spread_value if selected_team == home_team else -spread_value,
                    total=0.0,
                    home_ml=-110,
                    away_ml=-110,
                    source="manual"
                )
                
                prop = Proposition(
                    prop_id=f"{game_info.game_id}_spread_{selected_team}",
                    game_info=game_info,
                    bet_type=BetType.SPREAD,
                    line=betting_line,
                    line_value=spread_value,
                    metadata={'selected_team': selected_team, 'description': bet_description}
                )
                return prop, None
        
        return None, None
    
    def _parse_total(
        self,
        bet_description: str,
        bet_lower: str,
        game_info: GameInfo,
        home_team: str,
        away_team: str
    ) -> Tuple[Optional[Proposition], Optional[str]]:
        """Parse total (over/under) bets.
        
        Patterns:
        - "Over {value}" / "Under {value}"
        - "Total over {value}" / "Total under {value}"
        """
        # Pattern: "over/under {value}" or "total over/under {value}"
        total_pattern = r'(?:total\s+)?(over|under)\s+(\d+\.?\d*)'
        match = re.search(total_pattern, bet_lower)
        if match:
            direction = match.group(1).strip()
            total_value = float(match.group(2))
            
            # Create betting line
            betting_line = BettingLine(
                spread=0.0,
                total=total_value,
                home_ml=-110,
                away_ml=-110,
                source="manual"
            )
            
            prop = Proposition(
                prop_id=f"{game_info.game_id}_total_{direction}",
                game_info=game_info,
                bet_type=BetType.TOTAL,
                line=betting_line,
                line_value=total_value,
                metadata={'direction': direction, 'description': bet_description}
            )
            return prop, None
        
        return None, None
    
    def _normalize_team_name(self, team_name: str) -> Optional[str]:
        """Convert team name to abbreviation.
        
        Args:
            team_name: Team name in various formats
            
        Returns:
            Team abbreviation or None if not found
        """
        team_lower = team_name.lower().strip()
        
        # Direct lookup
        if team_lower in self.TEAM_MAPPING:
            return self.TEAM_MAPPING[team_lower]
        
        # Try with common variations
        # Remove common words
        team_clean = re.sub(r'\b(the|a|an)\b', '', team_lower).strip()
        if team_clean in self.TEAM_MAPPING:
            return self.TEAM_MAPPING[team_clean]
        
        # Try partial matches (for abbreviations)
        team_upper = team_name.upper().strip()
        if len(team_upper) <= 3:
            # Might already be an abbreviation
            return team_upper
        
        return None
    
    def _extract_stat_type(self, stat_description: str) -> str:
        """Extract stat type from description.
        
        Args:
            stat_description: Description of the stat
            
        Returns:
            Stat type string
        """
        stat_lower = stat_description.lower().strip()
        
        # Direct lookup
        if stat_lower in self.STAT_TYPE_MAPPING:
            return self.STAT_TYPE_MAPPING[stat_lower]
        
        # Try partial matches
        for key, value in self.STAT_TYPE_MAPPING.items():
            if key in stat_lower:
                return value
        
        # Default to receiving yards
        return 'receiving_yards'

