# Quick Start Guide

This guide will help you get started with the NFL Betting Agent Council.

## Prerequisites

- Python 3.8 or higher
- Anthropic API key (get one at https://console.anthropic.com/)

## Installation

1. **Clone the repository** (if you haven't already):
```bash
cd SportsBetting
```

2. **Create a virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**:
Create a `.env` file in the project root:
```bash
# Create .env file
cat > .env << EOF
ANTHROPIC_API_KEY=your_api_key_here
EOF
```

Replace `your_api_key_here` with your actual Anthropic API key.

## Training Models

Before you can analyze games, you need to train the ML models on historical data:

```bash
python scripts/train_models.py --seasons 2020 2021 2022 2023
```

This will:
- Download NFL data from the specified seasons
- Extract features from historical games
- Train 4 different ML models
- Save trained models to the `models/` directory

**Note**: Training may take 10-30 minutes depending on your hardware.

## Analyzing a Game

Once models are trained, you can analyze any upcoming or historical NFL game:

```bash
python -m src.cli analyze \
    --home-team KC \
    --away-team BUF \
    --season 2024 \
    --week 10 \
    --bet-type spread \
    --spread -2.5 \
    --total 54.5 \
    --verbose
```

### Parameters:

- `--home-team`: Home team abbreviation (e.g., KC, BUF, SF)
- `--away-team`: Away team abbreviation
- `--season`: NFL season year
- `--week`: Week number (1-18 for regular season)
- `--bet-type`: Type of bet (`game_outcome`, `spread`, `total`, `player_prop`)
- `--spread`: Point spread (optional, for spread bets)
- `--total`: Over/under total (optional, for total bets)
- `--verbose`: Show full debate transcript

### Team Abbreviations:

AFC East: BUF, MIA, NE, NYJ  
AFC North: BAL, CIN, CLE, PIT  
AFC South: HOU, IND, JAX, TEN  
AFC West: DEN, KC, LV, LAC  
NFC East: DAL, NYG, PHI, WAS  
NFC North: CHI, DET, GB, MIN  
NFC South: ATL, CAR, NO, TB  
NFC West: ARI, LAR, SF, SEA

## Understanding the Output

The analysis provides:

1. **Recommendation**: BET, STRONG_BET, or PASS
2. **Predicted Outcome**: Which team/outcome the council predicts
3. **Confidence Level**: How confident the council is (0-100%)
4. **Consensus Level**: How much the models agreed
5. **Individual Model Predictions**: Each model's independent prediction
6. **Debate Transcript** (with `--verbose`): Full debate between AI agents
7. **Reasoning**: Clear explanation of the decision
8. **Bet Sizing**: Suggested bet size (% of bankroll) using Kelly Criterion
9. **Expected Value**: Mathematical expected value of the bet

### Example Output:

```
==================================================
ANALYSIS RESULTS
==================================================

Final Recommendation: BET
Predicted Outcome: home_win
Confidence: 68.5%
Consensus Level: 75.2%
Suggested Bet Size: 3.2% of bankroll
Expected Value: 0.084

Risk Assessment: MODERATE: Good confidence with reasonable consensus.

Reasoning:
The council predicts a home win with 68.5% confidence. The Neural Analyst 
and Gradient Strategist both strongly favor the home team based on recent 
form and matchup advantages, while the Forest Evaluator and Statistical 
Conservative are more cautious but still lean home. Key factors include 
home field advantage, recent performance trends, and historical head-to-head results.
```

## Backtesting

Evaluate the system's performance on historical games:

```bash
python -m src.cli backtest \
    --start-date 2023-01-01 \
    --end-date 2023-12-31 \
    --model-dir models \
    --bankroll 1000
```

This will:
- Run the council on all games in the date range
- Track predictions vs actual outcomes
- Calculate accuracy, ROI, Sharpe ratio, etc.
- Show final bankroll after simulated betting

## Using the Python API

You can also use the system programmatically:

```python
from datetime import datetime
from src.pipeline.predictor import BettingCouncil
from src.utils.data_types import Proposition, GameInfo, BettingLine, BetType

# Initialize the council
council = BettingCouncil()
council.load_models("models")

# Create a proposition
game_info = GameInfo(
    game_id="2024_10_KC_BUF",
    home_team="KC",
    away_team="BUF",
    game_date=datetime(2024, 11, 17),
    season=2024,
    week=10
)

betting_line = BettingLine(
    spread=-2.5,
    total=54.5,
    home_ml=-135,
    away_ml=115
)

proposition = Proposition(
    prop_id="2024_10_KC_BUF_spread",
    game_info=game_info,
    bet_type=BetType.SPREAD,
    line=betting_line
)

# Analyze
recommendation = council.analyze(proposition)

# Access results
print(f"Action: {recommendation.recommended_action}")
print(f"Prediction: {recommendation.debate_result.final_prediction.value}")
print(f"Confidence: {recommendation.debate_result.final_confidence:.1%}")
```

## Jupyter Notebook

For an interactive experience, check out the example notebook:

```bash
jupyter notebook notebooks/example_usage.ipynb
```

## Configuration

Customize behavior by editing `config/config.yaml`:

- **Model parameters**: Learning rates, hidden layers, etc.
- **Debate settings**: Number of rounds, agent personalities
- **Betting thresholds**: Minimum confidence, Kelly fraction
- **Data sources**: Cache settings, data paths

## Troubleshooting

### "Model must be trained before prediction"
- Train models first using `python scripts/train_models.py`

### "ANTHROPIC_API_KEY not found"
- Make sure `.env` file exists with your API key
- Or export it: `export ANTHROPIC_API_KEY=your_key`

### "No betting lines found"
- Betting lines are optional for some bet types
- For historical games, you may need to provide lines manually

### Data download is slow
- First run downloads NFL data (can take 5-10 minutes)
- Subsequent runs use cached data from `data/raw/`

### Models predict poorly
- More training data improves performance
- Try training on 3-4 seasons: `--seasons 2020 2021 2022 2023`
- Backtest to evaluate and iterate

## Best Practices

1. **Always backtest before real betting**: Evaluate on historical data first
2. **Don't bet on every game**: PASS recommendations exist for a reason
3. **Use proper bankroll management**: Follow the suggested bet sizes
4. **Review debate transcripts**: Understand the reasoning, not just the prediction
5. **Track performance over time**: Adjust thresholds based on results
6. **Consider multiple factors**: The council's prediction is one input, not the only one

## Getting Help

- **Documentation**: See `README.md` for detailed information
- **Issues**: Report bugs or request features on GitHub
- **Examples**: Check `notebooks/` for usage examples

## Next Steps

1. Train models on historical data
2. Run backtests to evaluate performance
3. Analyze upcoming games
4. Track predictions vs outcomes
5. Refine and iterate

**Remember**: This system is for educational purposes. Sports betting involves risk. Always gamble responsibly.

