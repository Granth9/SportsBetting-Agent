# Betting Scraper Guide

This guide explains how to use the betting odds scraping system to get real-time betting options from Sleeper, Underdog, and ESPN.

## Overview

The betting scraper system allows you to:
1. Browse the 20 most popular betting options for any NFL game from supported sites
2. Select a specific bet to analyze
3. Get AI-powered recommendations for that bet

## Supported Sites

- **Sleeper** - Popular fantasy sports and betting platform
- **Underdog** - Fantasy-style betting with player props
- **ESPN** - Sports betting odds from ESPN

## Quick Start

### 1. Browse Betting Options

List the top 20 betting options for a game:

```bash
# Example: NY Giants vs NE Patriots on Sleeper
python -m src.cli browse \
    --site sleeper \
    --home-team NYG \
    --away-team NE \
    --date 2024-12-15 \
    --limit 20
```

This will display:
- Bet titles and descriptions
- Bet types (spread, total, player prop, game outcome)
- Odds (American format)
- Popularity rankings
- Player names (for props)
- Line values (for spreads/totals)

### 2. Analyze a Specific Bet

After browsing, analyze a bet by index:

```bash
python -m src.cli analyze-bet \
    --site sleeper \
    --home-team NYG \
    --away-team NE \
    --season 2024 \
    --week 15 \
    --date 2024-12-15 \
    --bet-index 5 \
    --model-dir models
```

Or use interactive mode (no `--bet-index`):

```bash
python -m src.cli analyze-bet \
    --site sleeper \
    --home-team NYG \
    --away-team NE \
    --season 2024 \
    --week 15 \
    --date 2024-12-15 \
    --model-dir models
```

This will:
1. Fetch all betting options
2. Display them for selection
3. Prompt you to choose a bet
4. Run the full AI analysis on your selection

## Command Reference

### Browse Command

```bash
python -m src.cli browse [OPTIONS]
```

**Options:**
- `--site` (required): `sleeper`, `underdog`, or `espn`
- `--home-team` (required): Home team abbreviation (e.g., `NYG`, `KC`, `BUF`)
- `--away-team` (required): Away team abbreviation
- `--date` (optional): Game date in `YYYY-MM-DD` format
- `--limit` (optional): Maximum number of bets to show (default: 20)
- `--save` (optional): Save results to JSON file

**Example:**
```bash
python -m src.cli browse \
    --site underdog \
    --home-team KC \
    --away-team BUF \
    --limit 30 \
    --save bets.json
```

### Analyze-Bet Command

```bash
python -m src.cli analyze-bet [OPTIONS]
```

**Options:**
- `--site` (required): `sleeper`, `underdog`, or `espn`
- `--home-team` (required): Home team abbreviation
- `--away-team` (required): Away team abbreviation
- `--season` (required): Season year (default: current year)
- `--week` (required): Week number (1-18)
- `--date` (optional): Game date in `YYYY-MM-DD` format
- `--bet-index` (optional): Bet number from browse (1-based index)
- `--bet-id` (optional): Specific bet ID to analyze
- `--model-dir` (optional): Directory with trained models (default: `models`)
- `--rounds` (optional): Number of debate rounds (default: 4)
- `--verbose` (optional): Show full debate transcript

**Examples:**

Analyze bet #3 from Sleeper:
```bash
python -m src.cli analyze-bet \
    --site sleeper \
    --home-team NYG \
    --away-team NE \
    --season 2024 \
    --week 15 \
    --bet-index 3 \
    --model-dir models
```

Interactive selection:
```bash
python -m src.cli analyze-bet \
    --site espn \
    --home-team KC \
    --away-team BUF \
    --season 2024 \
    --week 10 \
    --model-dir models \
    --verbose
```

## Workflow Example

Complete workflow for analyzing a bet:

```bash
# Step 1: Browse available bets
python -m src.cli browse \
    --site sleeper \
    --home-team NYG \
    --away-team NE \
    --date 2024-12-15

# Output shows:
# [1] NY Giants -3.5 vs NE Patriots
# [2] Over 42.5 Total Points
# [3] Patrick Mahomes Over 275.5 Passing Yards
# ... (20 total)

# Step 2: Analyze bet #3 (Mahomes passing yards)
python -m src.cli analyze-bet \
    --site sleeper \
    --home-team NYG \
    --away-team NE \
    --season 2024 \
    --week 15 \
    --bet-index 3 \
    --model-dir models \
    --verbose
```

## Bet Types

The system automatically classifies bets into:

- **Game Outcome**: Moneyline bets (team to win)
- **Spread**: Point spread bets
- **Total**: Over/under total points
- **Player Prop**: Player-specific statistics (yards, touchdowns, etc.)

## Output Format

### Browse Output

```
================================================================================
Found 20 betting options:
================================================================================

[#1] NY Giants -3.5 vs NE Patriots
    Type: spread | Odds: -110 | Source: sleeper
    Line: -3.5
    Description: NY Giants favored by 3.5 points...

[#2] Over 42.5 Total Points
    Type: total | Odds: -110 | Source: sleeper
    Line: 42.5
    Description: Total points over/under 42.5...

[#3] Patrick Mahomes Over 275.5 Passing Yards
    Type: player_prop | Odds: -115 | Source: sleeper
    Player: Patrick Mahomes
    Line: 275.5
    Description: Patrick Mahomes to throw over 275.5 passing yards...
```

### Analysis Output

After selecting a bet, you'll get:
- Final recommendation (BET, PASS, STRONG_BET)
- Predicted outcome
- Confidence level
- Consensus from all models
- Suggested bet size
- Expected value
- Risk assessment
- Full reasoning and debate transcript (if `--verbose`)

## Important Notes

### Website Structure Changes

These scrapers work by parsing HTML and API responses. If a website changes its structure:
- The scraper may fail to find bets
- You may see "No betting options found"
- The scraper code will need to be updated

### Rate Limiting

Be respectful of website resources:
- Don't scrape too frequently
- Some sites may block rapid requests
- Consider caching results

### Legal Compliance

- Ensure you comply with terms of service for each site
- This tool is for educational/research purposes
- Check local laws regarding sports betting data scraping

## Troubleshooting

### "No betting options found"

**Possible causes:**
1. Game not found on the site
2. Website structure changed (scraper needs update)
3. Network/API issues
4. Game date too far in future or past

**Solutions:**
- Try a different site
- Verify team abbreviations are correct
- Check if the game exists on the website manually
- Try with/without the `--date` parameter

### "Error scraping [site]"

**Possible causes:**
1. Website blocking requests
2. Network connectivity issues
3. API endpoint changed

**Solutions:**
- Check internet connection
- Try again later
- The scraper may need updates if the site changed

### Bet index out of range

**Cause:** The bet index you selected doesn't exist

**Solution:** Use `browse` command first to see available bets, then use a valid index

## Advanced Usage

### Save and Load Bets

Save bets to a file:
```bash
python -m src.cli browse \
    --site sleeper \
    --home-team NYG \
    --away-team NE \
    --save my_bets.json
```

Then analyze from saved file (future feature):
```python
import json
from src.data.collectors.betting_scraper_manager import BettingScraperManager

# Load saved bets
with open('my_bets.json', 'r') as f:
    bets_data = json.load(f)

# Convert back to BettingOption objects and analyze
```

### Programmatic Usage

```python
from src.data.collectors.betting_scraper_manager import BettingScraperManager
from datetime import datetime

manager = BettingScraperManager()

# Get bets from Sleeper
bets = manager.get_bets_for_game(
    site='sleeper',
    home_team='NYG',
    away_team='NE',
    game_date=datetime(2024, 12, 15),
    limit=20
)

# Display bets
print(manager.format_bets_for_display(bets))

# Convert to proposition and analyze
from src.pipeline.predictor import BettingCouncil
from src.utils.data_types import GameInfo

game_info = GameInfo(
    game_id="2024_15_NE_NYG",
    home_team="NYG",
    away_team="NE",
    game_date=datetime(2024, 12, 15),
    season=2024,
    week=15
)

selected_bet = bets[2]  # Select third bet
proposition = selected_bet.to_proposition(game_info)

council = BettingCouncil()
council.load_models("models")
recommendation = council.analyze(proposition)
```

## Future Enhancements

Planned improvements:
- Support for more betting sites
- Historical odds tracking
- Odds comparison across sites
- Automatic bet value detection
- Integration with live game data

