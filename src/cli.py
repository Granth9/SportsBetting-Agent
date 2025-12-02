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
from src.data.collectors.betting_scraper_manager import BettingScraperManager
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


def browse_command(args):
    """Handle the browse command - list betting options from a site.
    
    Args:
        args: Parsed command-line arguments
    """
    print("\n" + "="*80)
    print("NFL BETTING AGENT COUNCIL - BROWSE BETS")
    print("="*80 + "\n")
    
    print(f"Site: {args.site}")
    print(f"Game: {args.away_team} @ {args.home_team}")
    if args.date:
        print(f"Date: {args.date}")
    print()
    
    # Initialize scraper manager
    manager = BettingScraperManager()
    
    # Parse date if provided
    game_date = None
    if args.date:
        try:
            game_date = datetime.strptime(args.date, "%Y-%m-%d")
        except ValueError:
            print(f"Warning: Invalid date format. Use YYYY-MM-DD. Proceeding without date filter.")
    
    # Get bets
    print(f"Fetching betting options from {args.site}...")
    bets = manager.get_bets_for_game(
        site=args.site,
        home_team=args.home_team,
        away_team=args.away_team,
        game_date=game_date,
        limit=args.limit
    )
    
    if not bets:
        print(f"\nNo betting options found for {args.away_team} @ {args.home_team} on {args.site}.")
        print("\nPossible reasons:")
        print("  - Game not found on this site")
        print("  - Site structure may have changed (scraper needs update)")
        print("  - Network/API issues")
        return
    
    # Display bets
    display_text = manager.format_bets_for_display(bets)
    print(display_text)
    
    # Save to file if requested
    if args.save:
        output_path = Path(args.save)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        import json
        from dataclasses import asdict
        
        bets_data = [asdict(bet) for bet in bets]
        with open(output_path, 'w') as f:
            json.dump(bets_data, f, indent=2, default=str)
        
        print(f"\nBetting options saved to {output_path}")


def analyze_bet_command(args):
    """Handle the analyze-bet command - analyze a specific bet option.
    
    Args:
        args: Parsed command-line arguments
    """
    # Lazy import
    from src.pipeline.predictor import BettingCouncil
    
    print("\n" + "="*80)
    print("NFL BETTING AGENT COUNCIL - ANALYZE BET")
    print("="*80 + "\n")
    
    # Initialize scraper manager
    manager = BettingScraperManager()
    
    # Parse date if provided
    game_date = None
    if args.date:
        try:
            game_date = datetime.strptime(args.date, "%Y-%m-%d")
        except ValueError:
            print(f"Warning: Invalid date format. Use YYYY-MM-DD.")
            sys.exit(1)
    
    # Get bets
    print(f"Fetching betting options from {args.site}...")
    bets = manager.get_bets_for_game(
        site=args.site,
        home_team=args.home_team,
        away_team=args.away_team,
        game_date=game_date,
        limit=50  # Get more options to find the one we want
    )
    
    if not bets:
        print(f"\nNo betting options found.")
        sys.exit(1)
    
    # Find the selected bet
    selected_bet = None
    
    if args.bet_index:
        # Select by index
        try:
            index = int(args.bet_index) - 1
            if 0 <= index < len(bets):
                selected_bet = bets[index]
            else:
                print(f"Error: Bet index {args.bet_index} out of range (1-{len(bets)})")
                sys.exit(1)
        except ValueError:
            print(f"Error: Invalid bet index '{args.bet_index}'")
            sys.exit(1)
    elif args.bet_id:
        # Select by ID
        selected_bet = next((bet for bet in bets if bet.option_id == args.bet_id), None)
        if not selected_bet:
            print(f"Error: Bet with ID '{args.bet_id}' not found")
            sys.exit(1)
    else:
        # Interactive selection
        print("\nAvailable betting options:")
        print(manager.format_bets_for_display(bets))
        
        while True:
            try:
                choice = input(f"\nSelect a bet to analyze (1-{len(bets)}) or 'q' to quit: ").strip()
                if choice.lower() == 'q':
                    sys.exit(0)
                
                index = int(choice) - 1
                if 0 <= index < len(bets):
                    selected_bet = bets[index]
                    break
                else:
                    print(f"Please enter a number between 1 and {len(bets)}")
            except ValueError:
                print("Please enter a valid number or 'q' to quit")
            except KeyboardInterrupt:
                print("\nCancelled.")
                sys.exit(0)
    
    # Convert bet option to proposition
    game_info = GameInfo(
        game_id=f"{args.season}_{args.week:02d}_{args.away_team}_{args.home_team}",
        home_team=args.home_team,
        away_team=args.away_team,
        game_date=game_date or datetime.now(),
        season=args.season,
        week=args.week
    )
    
    proposition = selected_bet.to_proposition(game_info)
    
    print(f"\n{'='*80}")
    print(f"Selected Bet: {selected_bet.title}")
    print(f"Type: {selected_bet.bet_type.value} | Odds: {selected_bet.odds:+.0f}")
    print(f"{'='*80}\n")
    
    # Now analyze using the existing analyze functionality
    print("Initializing models and agents...")
    council = BettingCouncil(debate_rounds=args.rounds if hasattr(args, 'rounds') else 4)
    
    if args.model_dir and Path(args.model_dir).exists():
        print(f"Loading models from {args.model_dir}...")
        council.load_models(args.model_dir)
    else:
        print("Warning: No trained models loaded.")
    
    print("\nRunning analysis...\n")
    
    try:
        recommendation = council.analyze(proposition)
        
        # Display results (reuse analyze_command display logic)
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
    
    except Exception as e:
        logger.error(f"Error during analysis: {e}", exc_info=True)
        print(f"\nError: {e}")
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
    
    # Browse command
    browse_parser = subparsers.add_parser('browse', help='Browse betting options from a site')
    browse_parser.add_argument('--site', choices=['sleeper', 'underdog', 'espn'], required=True,
                              help='Betting site to scrape')
    browse_parser.add_argument('--home-team', required=True, help='Home team abbreviation (e.g., NYG)')
    browse_parser.add_argument('--away-team', required=True, help='Away team abbreviation (e.g., NE)')
    browse_parser.add_argument('--date', help='Game date (YYYY-MM-DD)')
    browse_parser.add_argument('--limit', type=int, default=20, help='Maximum number of bets to show')
    browse_parser.add_argument('--save', help='Save bets to JSON file')
    
    # Analyze-bet command
    analyze_bet_parser = subparsers.add_parser('analyze-bet', help='Analyze a specific bet from a site')
    analyze_bet_parser.add_argument('--site', choices=['sleeper', 'underdog', 'espn'], required=True,
                                    help='Betting site')
    analyze_bet_parser.add_argument('--home-team', required=True, help='Home team abbreviation')
    analyze_bet_parser.add_argument('--away-team', required=True, help='Away team abbreviation')
    analyze_bet_parser.add_argument('--season', type=int, default=datetime.now().year, help='Season year')
    analyze_bet_parser.add_argument('--week', type=int, required=True, help='Week number')
    analyze_bet_parser.add_argument('--date', help='Game date (YYYY-MM-DD)')
    analyze_bet_parser.add_argument('--bet-index', help='Bet index to analyze (from browse command)')
    analyze_bet_parser.add_argument('--bet-id', help='Bet ID to analyze')
    analyze_bet_parser.add_argument('--rounds', type=int, default=4, help='Number of debate rounds')
    analyze_bet_parser.add_argument('--model-dir', default='models', help='Directory with trained models')
    analyze_bet_parser.add_argument('--verbose', '-v', action='store_true', help='Show full debate transcript')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the models')
    train_parser.add_argument('--seasons', nargs='+', type=int, required=True, help='Seasons to train on')
    train_parser.add_argument('--output-dir', default='models', help='Directory to save trained models')
    
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
    elif args.command == 'browse':
        browse_command(args)
    elif args.command == 'analyze-bet':
        analyze_bet_command(args)


if __name__ == '__main__':
    main()

