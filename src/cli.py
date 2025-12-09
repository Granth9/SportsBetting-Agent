"""Command-line interface for the NFL Betting Agent Council."""

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Lazy imports - only import when needed to avoid requiring torch for browse command
# from src.pipeline.predictor import BettingCouncil
# from src.pipeline.evaluator import PerformanceEvaluator
from src.utils.data_types import (
    Proposition,
    GameInfo,
    BettingLine,
    BetType
)
from src.utils.bet_parser import BetParser
from src.utils.logger import setup_logger
from src.utils.config_loader import get_config


logger = setup_logger(__name__)


def create_proposition_from_args(args) -> Proposition:
    """Create a Proposition from command-line arguments.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Proposition object
    """
    # Create game info
    game_info = GameInfo(
        game_id=f"{args.season}_{args.week:02d}_{args.away_team}_{args.home_team}",
        home_team=args.home_team,
        away_team=args.away_team,
        game_date=datetime.strptime(args.date, "%Y-%m-%d") if args.date else datetime.now(),
        season=args.season,
        week=args.week
    )
    
    # Create betting line if provided
    betting_line = None
    if args.spread is not None and args.total is not None:
        betting_line = BettingLine(
            spread=args.spread,
            total=args.total,
            home_ml=args.home_ml if args.home_ml else -110,
            away_ml=args.away_ml if args.away_ml else -110
        )
    
    # Determine bet type
    bet_type_map = {
        'game_outcome': BetType.GAME_OUTCOME,
        'spread': BetType.SPREAD,
        'total': BetType.TOTAL,
        'player_prop': BetType.PLAYER_PROP
    }
    bet_type = bet_type_map.get(args.bet_type, BetType.GAME_OUTCOME)
    
    # Create proposition
    prop = Proposition(
        prop_id=f"{game_info.game_id}_{args.bet_type}",
        game_info=game_info,
        bet_type=bet_type,
        line=betting_line,
        player_name=args.player_name if hasattr(args, 'player_name') else None
    )
    
    return prop


def analyze_command(args):
    """Handle the analyze command.
    
    Args:
        args: Parsed command-line arguments
    """
    # Lazy import to avoid requiring torch for browse command
    from src.pipeline.predictor import BettingCouncil
    
    print("\n" + "="*80)
    print("NFL BETTING AGENT COUNCIL - ANALYSIS")
    print("="*80 + "\n")
    
    # Create proposition
    prop = create_proposition_from_args(args)
    
    print(f"Analyzing: {prop.game_info.away_team} @ {prop.game_info.home_team}")
    print(f"Bet Type: {prop.bet_type.value}")
    print(f"Season: {prop.game_info.season}, Week: {prop.game_info.week}")
    
    if prop.line:
        print(f"Spread: {prop.line.spread}, Total: {prop.line.total}")
    
    print("\n" + "-"*80)
    print("Initializing models and agents...")
    print("-"*80 + "\n")
    
    # Create council
    council = BettingCouncil(debate_rounds=args.rounds if hasattr(args, 'rounds') else 4)
    
    # Load models if specified
    if args.model_dir and Path(args.model_dir).exists():
        print(f"Loading models from {args.model_dir}...")
        council.load_models(args.model_dir)
    else:
        print("Warning: No trained models loaded. Models need to be trained first.")
        print("Proceeding with untrained models for demonstration purposes.\n")
    
    # Analyze
    print("Running analysis...\n")
    
    try:
        recommendation = council.analyze(prop)
        
        # Display results
        print("\n" + "="*80)
        print("ANALYSIS RESULTS")
        print("="*80 + "\n")
        
        print(f"Final Recommendation: {recommendation.recommended_action}")
        print(f"Predicted Outcome: {recommendation.debate_result.final_prediction.value}")
        print(f"Confidence: {recommendation.debate_result.final_confidence:.1%}")
        print(f"Consensus Level: {recommendation.debate_result.consensus_level:.1%}")
        
        if recommendation.bet_size:
            print(f"Suggested Bet Size: {recommendation.bet_size:.1%} of bankroll")
        
        if recommendation.expected_value:
            print(f"Expected Value: {recommendation.expected_value:.3f}")
        
        print(f"\nRisk Assessment: {recommendation.risk_assessment}")
        
        print(f"\nReasoning:\n{recommendation.debate_result.reasoning_summary}")
        
        # Show model predictions
        print("\n" + "-"*80)
        print("INDIVIDUAL MODEL PREDICTIONS")
        print("-"*80 + "\n")
        
        for pred in recommendation.debate_result.model_predictions:
            print(f"{pred.model_name}:")
            print(f"  Prediction: {pred.prediction.value}")
            print(f"  Confidence: {pred.confidence:.1%}")
            print(f"  Reasoning: {pred.reasoning[:100]}...")
            print()
        
        # Show debate transcript if verbose
        if args.verbose:
            print("\n" + "-"*80)
            print("DEBATE TRANSCRIPT")
            print("-"*80 + "\n")
            
            current_round = 0
            for arg in recommendation.debate_result.debate_transcript:
                if arg.round_number != current_round:
                    print(f"\n--- Round {arg.round_number} ---\n")
                    current_round = arg.round_number
                
                print(f"{arg.agent_name} ({arg.confidence:.1%} confident):")
                print(f"{arg.statement}\n")
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80 + "\n")
    
    except Exception as e:
        logger.error(f"Error during analysis: {e}", exc_info=True)
        print(f"\nError: {e}")
        sys.exit(1)


def backtest_command(args):
    """Handle the backtest command.
    
    Args:
        args: Parsed command-line arguments
    """
    # Lazy import
    from src.pipeline.predictor import BettingCouncil
    
    print("\n" + "="*80)
    print("NFL BETTING AGENT COUNCIL - BACKTESTING")
    print("="*80 + "\n")
    
    print(f"Backtest period: {args.start_date} to {args.end_date}")
    print(f"Initial bankroll: ${args.bankroll:.2f}\n")
    
    # Create council
    council = BettingCouncil()
    
    # Load models
    if args.model_dir and Path(args.model_dir).exists():
        print(f"Loading models from {args.model_dir}...")
        council.load_models(args.model_dir)
    else:
        print("Error: Model directory not found or not specified.")
        print("Use --model-dir to specify the path to trained models.")
        sys.exit(1)
    
    # TODO: Load historical propositions
    # This would require implementing a data loader for historical games
    print("\nNote: Backtest functionality requires historical proposition data.")
    print("Implement data loading in the backtest_command function.")
    
    # Placeholder for actual backtest
    # propositions = load_historical_propositions(args.start_date, args.end_date)
    # evaluator = PerformanceEvaluator(council)
    # metrics = evaluator.backtest(propositions, initial_bankroll=args.bankroll)
    
    # Display results
    # print("\n" + "="*80)
    # print("BACKTEST RESULTS")
    # print("="*80 + "\n")
    # print(f"Accuracy: {metrics['accuracy']:.1%}")
    # print(f"ROI: {metrics['roi']:.1%}")
    # print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    # print(f"Max Drawdown: {metrics['max_drawdown']:.1%}")
    # print(f"Total Bets: {metrics['total_bets']}")
    # print(f"Final Bankroll: ${metrics['final_bankroll']:.2f}")


def analyze_manual_command(args):
    """Handle the analyze-manual command for natural language bet input.
    
    Args:
        args: Parsed command-line arguments
    """
    # Lazy import to avoid requiring torch for browse command
    from src.pipeline.predictor import BettingCouncil
    
    print("\n" + "="*80)
    print("NFL BETTING AGENT COUNCIL - MANUAL BET ANALYSIS")
    print("="*80 + "\n")
    
    # Create game info
    game_date = datetime.strptime(args.date, "%Y-%m-%d") if args.date else datetime.now()
    game_info = GameInfo(
        game_id=f"{args.season}_{args.week:02d}_{args.away_team}_{args.home_team}",
        home_team=args.home_team,
        away_team=args.away_team,
        game_date=game_date,
        season=args.season,
        week=args.week
    )
    
    print(f"Game: {args.away_team} @ {args.home_team}")
    print(f"Season: {args.season}, Week: {args.week}")
    print(f"Bet Description: {args.bet}")
    print("\n" + "-"*80)
    print("Parsing bet description...")
    print("-"*80 + "\n")
    
    # Parse the bet description
    parser = BetParser()
    proposition, error = parser.parse_bet_description(
        args.bet,
        game_info,
        args.home_team,
        args.away_team
    )
    
    if not proposition:
        print(f"❌ Error: {error}")
        print("\nExamples of valid bet descriptions:")
        print("  - 'Will Cortland Sutton score a touchdown?'")
        print("  - 'Dolphins will win'")
        print("  - 'Who will win: Dolphins or Jets?'")
        print("  - 'Dolphins -3.5'")
        print("  - 'Over 45.5'")
        sys.exit(1)
    
    print(f"✅ Successfully parsed bet!")
    print(f"   Bet Type: {proposition.bet_type.value}")
    if proposition.player_name:
        print(f"   Player: {proposition.player_name}")
    if proposition.line_value is not None:
        print(f"   Line Value: {proposition.line_value}")
    if proposition.stat_type:
        print(f"   Stat Type: {proposition.stat_type}")
    print()
    
    # Now run the analysis
    print("-"*80)
    print("Initializing models and agents...")
    print("-"*80 + "\n")
    
    council = BettingCouncil(debate_rounds=args.rounds)
    
    if args.model_dir and Path(args.model_dir).exists():
        print(f"Loading models from {args.model_dir}...")
        council.load_models(args.model_dir)
    else:
        print("Warning: No trained models loaded. Analysis may not be accurate.")
        print("Train models first using: python -m src.cli train --seasons 2020 2021 2022 2023")
    
    print("\nRunning analysis...\n")
    
    try:
        recommendation = council.analyze(proposition)
        
        # Display results
        print("\n" + "="*80)
        print("ANALYSIS RESULTS")
        print("="*80 + "\n")
        
        print(f"Bet: {args.bet}")
        print(f"Game: {args.away_team} @ {args.home_team}")
        print()
        
        print(f"Recommended Action: {recommendation.recommended_action.value.upper()}")
        print(f"Confidence: {recommendation.confidence:.1%}")
        print(f"Expected Value: {recommendation.expected_value:.2f}")
        print()
        
        if recommendation.debate_result:
            print("Model Consensus:")
            for prediction in recommendation.debate_result.model_predictions:
                print(f"  - {prediction.model_name}: {prediction.prediction.value} "
                      f"(confidence: {prediction.confidence:.1%})")
            print()
        
        if args.verbose and recommendation.debate_result:
            print("="*80)
            print("DEBATE TRANSCRIPT")
            print("="*80 + "\n")
            for round_num, round_data in enumerate(recommendation.debate_result.debate_rounds, 1):
                print(f"Round {round_num}:")
                for argument in round_data:
                    print(f"  [{argument.agent_name}]: {argument.argument}")
                print()
        
        print("="*80)
        
    except Exception as e:
        logger.error(f"Error during analysis: {e}", exc_info=True)
        print(f"\n❌ Error during analysis: {e}")
        sys.exit(1)


def train_command(args):
    """Handle the train command.
    
    Args:
        args: Parsed command-line arguments
    """
    print("\n" + "="*80)
    print("NFL BETTING AGENT COUNCIL - MODEL TRAINING")
    print("="*80 + "\n")
    
    print("Note: Model training requires prepared training data.")
    print("Use: python scripts/train_models.py --seasons 2020 2021 2022 2023")
    print("\nSteps:")
    print("1. Collect historical NFL data")
    print("2. Engineer features")
    print("3. Train each model")
    print("4. Save trained models")


def predict_command(args):
    """Handle the predict command for quick natural language predictions.
    
    Args:
        args: Parsed command-line arguments
    """
    from src.pipeline.quick_predict import QuickPredictor
    
    # Join query parts
    query = ' '.join(args.query)
    
    print("\n" + "="*60)
    print("NFL BETTING PREDICTION")
    print("="*60)
    
    # Create predictor and make prediction
    predictor = QuickPredictor(args.model_dir)
    result = predictor.predict(query)
    
    if not result['success']:
        print(f"\n❌ {result['error']}")
        if 'hint' in result:
            print(f"💡 {result['hint']}")
        return
    
    # Check if this is a player prop
    if result.get('query_type') == 'player_prop':
        print(f"\n🏈 PLAYER PROP: {result['player_name']}")
        print(f"📍 {result['team']} | {result['position']}")
        print("-"*60)
        
        # Check if this is a TD prediction
        if result.get('is_td_prop'):
            print(f"\n📊 {result['stat_type']}")
            print(f"\n🎯 PREDICTION: {result['prediction']} (scores TD)")
            print(f"📈 Confidence: {result['confidence']:.0%}")
            print()
            print(f"🏈 TD Rate: {result['td_rate']:.0%} ({result['games_with_td']} TDs in {result['games_analyzed']} games)")
            print(f"📊 Total TDs This Season: {result['total_tds']}")
            print(f"📈 Avg TDs/Game: {result['avg_tds_per_game']:.2f}")
            print()
            # Handle both string format ("W1: 1 TD") and numeric format
            last_5 = result['last_5_games']
            if last_5 and isinstance(last_5[0], str):
                print(f"📅 Last 5 Games: {', '.join(last_5)}")
            else:
                print(f"📅 Last 5 Games: {[f'{t} TD' for t in last_5]}")
        else:
            # Yards prediction
            print(f"\n📊 {result['stat_type']}: {result['line']} yards")
            print(f"\n🎯 PREDICTION: {result['prediction']}")
            print(f"📈 Confidence: {result['confidence']:.0%}")
            print()
            print(f"📈 Season Average: {result['avg_yards']:.1f} yards")
            print(f"📊 Median: {result['median_yards']:.1f} yards")
            print(f"📋 Games Analyzed: {result['games_analyzed']}")
            print(f"✓ Hit Rate: {result['hit_rate']:.0%} over this line")
            print()
            # Handle both string format ("W1: 16") and numeric format
            last_5 = result['last_5_games']
            if last_5 and isinstance(last_5[0], str):
                print(f"📅 Last 5 Games: {', '.join(last_5)}")
            else:
                print(f"📅 Last 5 Games: {[f'{y:.0f}' for y in last_5]}")
        
        # Recommendation
        print()
        if result['confidence'] >= 0.65:
            print("✅ RECOMMENDED BET")
        else:
            print("⚠️  MARGINAL - Consider passing")
    else:
        # Game outcome prediction
        home_full = result.get('home_team_full', result.get('home_team', 'Unknown'))
        away_full = result.get('away_team_full', result.get('away_team', 'Unknown'))
        
        print(f"\n{away_full} @ {home_full}")
        print(f"Week {result['week']} | {result.get('gameday', 'TBD')}")
        print("-"*60)
        
        if result.get('completed'):
            print(f"\n📊 FINAL: {result['score']}")
            print(f"🏆 {result['result']}")
        else:
            print(f"\n🎯 PREDICTION: {result['prediction_full']} WIN")
            print(f"📈 Confidence: {result['confidence']:.0%}")
            print(f"📊 Spread: {result['spread']:+.1f}")
            print()
            
            if result['meets_criteria']:
                print("✅ HIGH CONFIDENCE PICK")
                print("   Meets selective 75% strategy criteria")
            else:
                print("⚠️  LOWER CONFIDENCE")
                print("   Consider skipping this game")
    
    print("\n" + "="*60)


def parlay_command(args):
    """Handle the parlay command for building multi-leg parlays.
    
    Args:
        args: Parsed command-line arguments
    """
    from src.pipeline.parlay_builder import ParlayBuilder
    
    # Join query parts
    query = ' '.join(args.query)
    
    print("\n" + "="*60)
    print("🎰 PARLAY BUILDER")
    print("="*60)
    
    # Create builder and parse query
    builder = ParlayBuilder(args.model_dir)
    params = builder.parse_parlay_query(query)
    
    print(f"\n📋 Building {params['num_legs']}-leg parlay...")
    
    # Generate options based on query type
    if params['is_single_game'] and len(params['teams']) == 2:
        team1, team2 = params['teams']
        print(f"🏈 Game: {builder.TEAM_FULL_NAMES.get(team1, team1)} vs {builder.TEAM_FULL_NAMES.get(team2, team2)}")
        print("\n⏳ Generating betting options...")
        options = builder.generate_game_options(team1, team2)
    elif params['is_week_parlay'] and params['week']:
        print(f"📅 Week {params['week']} parlay")
        print("\n⏳ Generating betting options for all games...")
        options = builder.generate_week_options(params['week'])
    else:
        print("\n❌ Could not parse parlay query")
        print("💡 Try: '4 leg parlay Eagles vs Chargers' or '5 leg parlay week 14'")
        return
    
    if not options:
        print("\n❌ No betting options generated. Please check team names or week number.")
        return
    
    # Auto-select top N options by confidence
    selected_options = options[:params['num_legs']]
    parlay = builder.build_parlay(selected_options)
    
    if parlay['success']:
        print(builder.display_parlay(parlay))
        
        # Show all available options for reference
        print("\n" + "-"*60)
        print("📋 ALL AVAILABLE OPTIONS (if you want to swap):")
        print(builder.display_options(options, max_display=15))
    else:
        print(f"\n❌ Error building parlay: {parlay.get('error', 'Unknown error')}")
    
    print("\n" + "="*60)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="NFL Betting Agent Council - AI-powered betting analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze a betting proposition')
    analyze_parser.add_argument('--home-team', required=True, help='Home team abbreviation (e.g., KC)')
    analyze_parser.add_argument('--away-team', required=True, help='Away team abbreviation (e.g., BUF)')
    analyze_parser.add_argument('--season', type=int, default=datetime.now().year, help='Season year')
    analyze_parser.add_argument('--week', type=int, required=True, help='Week number')
    analyze_parser.add_argument('--date', help='Game date (YYYY-MM-DD)')
    analyze_parser.add_argument('--bet-type', choices=['game_outcome', 'spread', 'total', 'player_prop'],
                                default='game_outcome', help='Type of bet')
    analyze_parser.add_argument('--spread', type=float, help='Point spread (positive for home team)')
    analyze_parser.add_argument('--total', type=float, help='Over/under total')
    analyze_parser.add_argument('--home-ml', type=int, help='Home team moneyline')
    analyze_parser.add_argument('--away-ml', type=int, help='Away team moneyline')
    analyze_parser.add_argument('--player-name', help='Player name (for player props)')
    analyze_parser.add_argument('--rounds', type=int, default=4, help='Number of debate rounds')
    analyze_parser.add_argument('--model-dir', default='models', help='Directory with trained models')
    analyze_parser.add_argument('--verbose', '-v', action='store_true', help='Show full debate transcript')
    
    # Backtest command
    backtest_parser = subparsers.add_parser('backtest', help='Backtest on historical data')
    backtest_parser.add_argument('--start-date', required=True, help='Start date (YYYY-MM-DD)')
    backtest_parser.add_argument('--end-date', required=True, help='End date (YYYY-MM-DD)')
    backtest_parser.add_argument('--bankroll', type=float, default=1000.0, help='Initial bankroll')
    backtest_parser.add_argument('--model-dir', required=True, help='Directory with trained models')
    backtest_parser.add_argument('--output', help='Output file for results (CSV)')
    
    # Analyze-manual command
    analyze_manual_parser = subparsers.add_parser('analyze-manual', help='Analyze a manually entered bet using natural language')
    analyze_manual_parser.add_argument('--bet', required=True, help='Bet description in natural language (e.g., "Will Cortland Sutton score a touchdown?" or "Dolphins will win")')
    analyze_manual_parser.add_argument('--home-team', required=True, help='Home team abbreviation (e.g., NYJ)')
    analyze_manual_parser.add_argument('--away-team', required=True, help='Away team abbreviation (e.g., MIA)')
    analyze_manual_parser.add_argument('--season', type=int, default=datetime.now().year, help='Season year')
    analyze_manual_parser.add_argument('--week', type=int, required=True, help='Week number')
    analyze_manual_parser.add_argument('--date', help='Game date (YYYY-MM-DD)')
    analyze_manual_parser.add_argument('--rounds', type=int, default=4, help='Number of debate rounds')
    analyze_manual_parser.add_argument('--model-dir', default='models', help='Directory with trained models')
    analyze_manual_parser.add_argument('--verbose', '-v', action='store_true', help='Show full debate transcript')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the models')
    train_parser.add_argument('--seasons', nargs='+', type=int, required=True, help='Seasons to train on')
    train_parser.add_argument('--output-dir', default='models/trained', help='Directory to save trained models')
    
    # Predict command (simple natural language)
    predict_parser = subparsers.add_parser('predict', help='Quick prediction from natural language query')
    predict_parser.add_argument('query', nargs='+', help='Natural language query (e.g., "Chiefs vs Raiders")')
    predict_parser.add_argument('--model-dir', default='models/trained', help='Directory with trained models')
    
    # Parlay builder command
    parlay_parser = subparsers.add_parser('parlay', help='Build a parlay from betting options')
    parlay_parser.add_argument('query', nargs='+', help='Parlay query (e.g., "4 leg parlay Eagles vs Chargers" or "5 leg parlay week 14")')
    parlay_parser.add_argument('--model-dir', default='models/trained', help='Directory with trained models')
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Route to appropriate command
    if args.command == 'analyze':
        analyze_command(args)
    elif args.command == 'backtest':
        backtest_command(args)
    elif args.command == 'train':
        train_command(args)
    elif args.command == 'analyze-manual':
        analyze_manual_command(args)
    elif args.command == 'predict':
        predict_command(args)
    elif args.command == 'parlay':
        parlay_command(args)


if __name__ == '__main__':
    main()

