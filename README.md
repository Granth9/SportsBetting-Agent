# NFL Betting Agent Council

An advanced sports betting analysis system that combines multiple machine learning models with LLM-based agent debate to generate data-driven betting recommendations.

## Overview

This system is inspired by PewDiePie's AI council concept, where multiple AI agents with different perspectives debate to reach the best decision. In our case, each ML model is represented by an LLM agent (Claude) that argues for its prediction, leading to a consensus-driven final recommendation.

## Architecture

The system consists of four main layers:

1. **Data Collection Layer** - Scrapes and stores NFL statistics, player data, team data, and betting lines
2. **ML Models Layer** - Multiple specialized models (neural networks, gradient boosting, random forests, statistical models) that independently evaluate betting propositions
3. **Agent Debate Layer** - LLM agents representing each model debate the predictions through multiple rounds
4. **Orchestration & Output** - Coordinates the workflow and produces final recommendations

## Features

- **Multiple ML Models**: 4-6 diverse models with different approaches to prediction
- **LLM Agent Debate**: Claude-powered agents debate predictions with unique personalities
- **Comprehensive Data**: NFL game stats, player performance, team metrics, injury reports, and betting lines
- **Multiple Bet Types**: Game outcomes, spreads, totals, and player props
- **Backtesting**: Historical performance tracking and ROI calculation
- **Explainability**: Clear reasoning for each recommendation

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd SportsBetting
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

## Usage

### Command Line Interface

Analyze a specific game:
```bash
python -m src.cli analyze --home-team "Chiefs" --away-team "Bills" --bet-type spread
```

### Python API

```python
from src.pipeline.predictor import BettingCouncil
from src.utils.data_types import Proposition, BetType

# Create a betting council
council = BettingCouncil()

# Analyze a proposition
proposition = Proposition(...)
recommendation = council.analyze(proposition)

print(f"Recommendation: {recommendation.recommended_action}")
print(f"Confidence: {recommendation.debate_result.final_confidence}")
```

## Project Structure

```
SportsBetting/
├── src/
│   ├── data/              # Data collection and processing
│   ├── models/            # ML models
│   ├── agents/            # LLM agents and debate system
│   ├── pipeline/          # Main orchestration
│   └── utils/             # Utilities
├── data/                  # Data storage
├── models/                # Saved model weights
├── config/                # Configuration files
├── notebooks/             # Jupyter notebooks
└── tests/                 # Test suite
```

## Configuration

Edit `config/config.yaml` to customize:
- Model hyperparameters
- Debate settings (rounds, agent personalities)
- Data sources
- Evaluation metrics

## Development

Run tests:
```bash
pytest tests/
```

Train models:
```bash
python -m src.pipeline.train
```

Run backtesting:
```bash
python -m src.pipeline.backtest --start-date 2021-01-01 --end-date 2023-12-31
```

## License

MIT License

## Disclaimer

This system is for educational and research purposes only. Sports betting involves risk. Always gamble responsibly and never bet more than you can afford to lose.
