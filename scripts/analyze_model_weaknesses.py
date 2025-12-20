"""Comprehensive analysis script to identify model weaknesses and areas for improvement."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json
import joblib
from typing import Dict, List, Any, Optional
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.base_model import BaseModel
from src.data.collectors.nfl_data_collector import NFLDataCollector
from src.data.processors.feature_engineer import FeatureEngineer
from src.data.processors.data_preprocessor import DataPreprocessor
from src.utils.logger import setup_logger
from src.utils.data_types import Outcome
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

logger = setup_logger(__name__)


class ModelWeaknessAnalyzer:
    """Analyze model weaknesses and identify improvement areas."""
    
    def __init__(self, model_dir: str = 'models/trained'):
        """Initialize analyzer.
        
        Args:
            model_dir: Directory containing trained models
        """
        self.model_dir = Path(model_dir)
        self.models: Dict[str, BaseModel] = {}
        self.metadata = None
        self.preprocessor = None
        
        # Load models and metadata
        self._load_models()
    
    def _load_models(self):
        """Load trained models and metadata."""
        logger.info(f"Loading models from {self.model_dir}")
        
        # Load metadata
        metadata_path = self.model_dir / 'metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            logger.info(f"Loaded metadata: {len(self.metadata.get('model_accuracies', {}))} models")
        
        # Load preprocessor
        preprocessor_path = self.model_dir / 'preprocessor.pkl'
        if preprocessor_path.exists():
            self.preprocessor = joblib.load(preprocessor_path)
            logger.info("Loaded preprocessor")
        
        # Load models
        model_files = list(self.model_dir.glob('*.pkl'))
        for model_file in model_files:
            if model_file.name in ['preprocessor.pkl', 'scaler.pkl', 'metadata.json']:
                continue
            
            try:
                model_data = joblib.load(model_file)
                
                # Models are saved as dicts with BaseModel data
                if isinstance(model_data, dict) and 'model' in model_data:
                    model_name = model_data.get('name', model_file.stem.replace('_', ' ').title())
                    model_type = model_data.get('model_type', 'unknown')
                    
                    # Reconstruct model based on type
                    try:
                        if 'neural' in model_type:
                            from src.models.neural_nets.deep_predictor import DeepPredictor
                            temp_model = DeepPredictor(name=model_name)
                        elif 'gradient' in model_type:
                            from src.models.traditional.gradient_boost_model import GradientBoostModel
                            temp_model = GradientBoostModel(name=model_name)
                        elif 'random' in model_type or 'forest' in model_type:
                            from src.models.traditional.random_forest_model import RandomForestModel
                            temp_model = RandomForestModel(name=model_name)
                        elif 'statistical' in model_type:
                            from src.models.traditional.statistical_model import StatisticalModel
                            temp_model = StatisticalModel(name=model_name)
                        elif 'lightgbm' in model_type:
                            from src.models.traditional.lightgbm_model import LightGBMModel
                            temp_model = LightGBMModel(name=model_name)
                        elif 'svm' in model_type:
                            from src.models.traditional.svm_model import SVMModel
                            temp_model = SVMModel(name=model_name)
                        elif 'stacking' in model_type:
                            from src.models.ensemble.stacking_model import StackingModel
                            temp_model = StackingModel(name=model_name)
                        elif 'ensemble' in model_type:
                            from src.models.ensemble.ensemble_model import EnsembleModel
                            temp_model = EnsembleModel(name=model_name)
                        else:
                            # Try to infer from filename
                            if 'statistical' in model_file.name.lower():
                                from src.models.traditional.statistical_model import StatisticalModel
                                temp_model = StatisticalModel(name=model_name)
                            elif 'svm' in model_file.name.lower():
                                from src.models.traditional.svm_model import SVMModel
                                temp_model = SVMModel(name=model_name)
                            elif 'stacking' in model_file.name.lower():
                                from src.models.ensemble.stacking_model import StackingModel
                                temp_model = StackingModel(name=model_name)
                            else:
                                logger.warning(f"Unknown model type for {model_file.name}, skipping")
                                continue
                        
                        # Load the model using BaseModel.load
                        temp_model.load(str(model_file))
                        self.models[temp_model.name] = temp_model
                        logger.info(f"Loaded model: {temp_model.name}")
                    except Exception as e2:
                        logger.warning(f"Could not reconstruct {model_name} from {model_file.name}: {e2}")
                elif hasattr(model_data, 'name'):
                    # Direct model instance
                    self.models[model_data.name] = model_data
                    logger.info(f"Loaded model: {model_data.name}")
                else:
                    logger.warning(f"Unknown format in {model_file.name}")
            except Exception as e:
                logger.warning(f"Could not load {model_file}: {e}")
        
        logger.info(f"Loaded {len(self.models)} models")
    
    def analyze_performance(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Analyze performance metrics for all models.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary with performance metrics for each model
        """
        logger.info("="*60)
        logger.info("Performance Analysis")
        logger.info("="*60)
        
        results = {}
        
        for model_name, model in self.models.items():
            if not model.is_trained:
                logger.warning(f"{model_name} is not trained, skipping")
                continue
            
            try:
                # Get predictions
                predictions = []
                probabilities = []
                
                for i in range(len(X_test)):
                    try:
                        pred = model.predict(X_test[i:i+1])
                        proba = model.predict_proba(X_test[i:i+1])
                        
                        predictions.append(1 if pred.prediction == Outcome.HOME_WIN else 0)
                        probabilities.append(proba.get(Outcome.HOME_WIN, 0.5))
                    except Exception as e:
                        logger.warning(f"Error predicting sample {i} with {model_name}: {e}")
                        predictions.append(0)
                        probabilities.append(0.5)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, predictions)
                precision = precision_score(y_test, predictions, zero_division=0)
                recall = recall_score(y_test, predictions, zero_division=0)
                f1 = f1_score(y_test, predictions, zero_division=0)
                
                try:
                    roc_auc = roc_auc_score(y_test, probabilities)
                except:
                    roc_auc = 0.0
                
                # Confusion matrix
                cm = confusion_matrix(y_test, predictions)
                if cm.size == 4:
                    tn, fp, fn, tp = cm.ravel()
                else:
                    tn, fp, fn, tp = 0, 0, 0, 0
                
                results[model_name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'roc_auc': roc_auc,
                    'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)},
                    'predictions': predictions,
                    'probabilities': probabilities
                }
                
                logger.info(f"\n{model_name}:")
                logger.info(f"  Accuracy:  {accuracy:.1%}")
                logger.info(f"  Precision: {precision:.1%}")
                logger.info(f"  Recall:    {recall:.1%}")
                logger.info(f"  F1 Score:  {f1:.1%}")
                logger.info(f"  ROC-AUC:   {roc_auc:.3f}")
                logger.info(f"  Confusion: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
                
            except Exception as e:
                logger.error(f"Error analyzing {model_name}: {e}", exc_info=True)
                results[model_name] = {'error': str(e)}
        
        return results
    
    def analyze_error_patterns(self, X_test: np.ndarray, y_test: np.ndarray, 
                              test_features_df: pd.DataFrame,
                              performance_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze error patterns and misclassification types.
        
        Args:
            X_test: Test features
            y_test: Test labels
            test_features_df: DataFrame with test features (for analysis)
            performance_results: Results from analyze_performance
            
        Returns:
            Dictionary with error pattern analysis
        """
        logger.info("\n" + "="*60)
        logger.info("Error Pattern Analysis")
        logger.info("="*60)
        
        error_analysis = {}
        
        for model_name, results in performance_results.items():
            if 'error' in results or 'predictions' not in results:
                continue
            
            predictions = results['predictions']
            errors = np.array(predictions) != np.array(y_test)
            
            error_analysis[model_name] = {
                'total_errors': int(errors.sum()),
                'error_rate': float(errors.mean()),
                'false_positives': int(results['confusion_matrix']['fp']),
                'false_negatives': int(results['confusion_matrix']['fn']),
                'error_by_feature': {}
            }
            
            # Analyze errors by various game characteristics
            if len(test_features_df) == len(errors):
                # Analyze by spread (if available)
                if 'spread' in test_features_df.columns:
                    spread_errors = test_features_df.loc[errors, 'spread']
                    spread_correct = test_features_df.loc[~errors, 'spread']
                    
                    error_analysis[model_name]['errors_by_spread'] = {
                        'mean_error_spread': float(spread_errors.mean()) if len(spread_errors) > 0 else 0,
                        'mean_correct_spread': float(spread_correct.mean()) if len(spread_correct) > 0 else 0,
                        'close_games_error_rate': float(errors[test_features_df['spread'].abs() < 3].mean()) if 'spread' in test_features_df.columns else None,
                        'blowout_error_rate': float(errors[test_features_df['spread'].abs() > 10].mean()) if 'spread' in test_features_df.columns else None
                    }
                
                # Analyze by season/week
                if 'season' in test_features_df.columns:
                    season_errors = test_features_df.loc[errors, 'season'].value_counts()
                    error_analysis[model_name]['errors_by_season'] = season_errors.to_dict()
                
                if 'week' in test_features_df.columns:
                    week_errors = test_features_df.loc[errors, 'week'].value_counts()
                    error_analysis[model_name]['errors_by_week'] = week_errors.to_dict()
            
            logger.info(f"\n{model_name} Error Analysis:")
            logger.info(f"  Total Errors: {error_analysis[model_name]['total_errors']}")
            logger.info(f"  Error Rate: {error_analysis[model_name]['error_rate']:.1%}")
            logger.info(f"  False Positives: {error_analysis[model_name]['false_positives']}")
            logger.info(f"  False Negatives: {error_analysis[model_name]['false_negatives']}")
        
        return error_analysis
    
    def analyze_features(self) -> Dict[str, Any]:
        """Analyze feature importance and quality.
        
        Returns:
            Dictionary with feature analysis
        """
        logger.info("\n" + "="*60)
        logger.info("Feature Analysis")
        logger.info("="*60)
        
        feature_analysis = {
            'feature_importance': {},
            'common_features': [],
            'missing_features': []
        }
        
        # Get feature importance from tree-based models
        for model_name, model in self.models.items():
            if not model.is_trained:
                continue
            
            try:
                if hasattr(model, 'get_feature_importance'):
                    importance = model.get_feature_importance()
                    if importance:
                        feature_analysis['feature_importance'][model_name] = importance
                        logger.info(f"\n{model_name} Top 10 Features:")
                        top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
                        for feat, imp in top_features:
                            logger.info(f"  {feat}: {imp:.4f}")
            except Exception as e:
                logger.debug(f"Could not get feature importance from {model_name}: {e}")
        
        # Analyze feature overlap
        if feature_analysis['feature_importance']:
            all_features = set()
            for model_importance in feature_analysis['feature_importance'].values():
                all_features.update(model_importance.keys())
            
            feature_analysis['total_unique_features'] = len(all_features)
            logger.info(f"\nTotal unique features across models: {len(all_features)}")
        
        return feature_analysis
    
    def analyze_model_specific_weaknesses(self, performance_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze model-specific weaknesses.
        
        Args:
            performance_results: Results from analyze_performance
            
        Returns:
            Dictionary with model-specific analysis
        """
        logger.info("\n" + "="*60)
        logger.info("Model-Specific Weakness Analysis")
        logger.info("="*60)
        
        weaknesses = {}
        
        for model_name, results in performance_results.items():
            if 'error' in results:
                continue
            
            model_weaknesses = []
            
            # Check for overfitting (would need train performance, skip for now)
            
            # Check precision/recall imbalance
            if results['precision'] < 0.5:
                model_weaknesses.append("Very low precision - many false positives")
            if results['recall'] < 0.5:
                model_weaknesses.append("Very low recall - many false negatives")
            
            # Check ROC-AUC
            if results['roc_auc'] < 0.6:
                model_weaknesses.append("Poor discrimination (ROC-AUC < 0.6)")
            
            # Check confusion matrix patterns
            cm = results['confusion_matrix']
            total = cm['tn'] + cm['fp'] + cm['fn'] + cm['tp']
            if total > 0:
                fp_rate = cm['fp'] / total
                fn_rate = cm['fn'] / total
                
                if fp_rate > 0.3:
                    model_weaknesses.append(f"High false positive rate ({fp_rate:.1%})")
                if fn_rate > 0.3:
                    model_weaknesses.append(f"High false negative rate ({fn_rate:.1%})")
            
            weaknesses[model_name] = {
                'weaknesses': model_weaknesses,
                'accuracy': results['accuracy'],
                'needs_improvement': len(model_weaknesses) > 0
            }
            
            if model_weaknesses:
                logger.info(f"\n{model_name} Weaknesses:")
                for weakness in model_weaknesses:
                    logger.info(f"  - {weakness}")
            else:
                logger.info(f"\n{model_name}: No major weaknesses identified")
        
        return weaknesses
    
    def generate_report(self, performance_results: Dict[str, Any],
                       error_analysis: Dict[str, Any],
                       feature_analysis: Dict[str, Any],
                       model_weaknesses: Dict[str, Any],
                       output_path: Path) -> None:
        """Generate comprehensive analysis report.
        
        Args:
            performance_results: Performance metrics
            error_analysis: Error pattern analysis
            feature_analysis: Feature analysis
            model_weaknesses: Model-specific weaknesses
            output_path: Path to save report
        """
        logger.info("\n" + "="*60)
        logger.info("Generating Analysis Report")
        logger.info("="*60)
        
        report_lines = []
        report_lines.append("# Model Weakness Analysis Report\n")
        report_lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Performance Summary
        report_lines.append("## Performance Summary\n\n")
        report_lines.append("| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |\n")
        report_lines.append("|-------|----------|-----------|--------|----|---------|\n")
        
        for model_name, results in sorted(performance_results.items(), 
                                          key=lambda x: x[1].get('accuracy', 0), reverse=True):
            if 'error' in results:
                continue
            report_lines.append(
                f"| {model_name} | {results['accuracy']:.1%} | {results['precision']:.1%} | "
                f"{results['recall']:.1%} | {results['f1']:.1%} | {results['roc_auc']:.3f} |\n"
            )
        
        # Best/Worst Models
        report_lines.append("\n## Best and Worst Performing Models\n\n")
        sorted_models = sorted([(name, res) for name, res in performance_results.items() 
                               if 'error' not in res],
                              key=lambda x: x[1]['accuracy'], reverse=True)
        
        if sorted_models:
            best = sorted_models[0]
            worst = sorted_models[-1]
            report_lines.append(f"**Best Model**: {best[0]} ({best[1]['accuracy']:.1%} accuracy)\n\n")
            report_lines.append(f"**Worst Model**: {worst[0]} ({worst[1]['accuracy']:.1%} accuracy)\n\n")
        
        # Model Weaknesses
        report_lines.append("## Model-Specific Weaknesses\n\n")
        for model_name, weakness_data in model_weaknesses.items():
            report_lines.append(f"### {model_name}\n\n")
            if weakness_data['weaknesses']:
                for weakness in weakness_data['weaknesses']:
                    report_lines.append(f"- {weakness}\n")
            else:
                report_lines.append("- No major weaknesses identified\n")
            report_lines.append("\n")
        
        # Error Patterns
        report_lines.append("## Error Pattern Analysis\n\n")
        for model_name, error_data in error_analysis.items():
            report_lines.append(f"### {model_name}\n\n")
            report_lines.append(f"- Total Errors: {error_data['total_errors']}\n")
            report_lines.append(f"- Error Rate: {error_data['error_rate']:.1%}\n")
            report_lines.append(f"- False Positives: {error_data['false_positives']}\n")
            report_lines.append(f"- False Negatives: {error_data['false_negatives']}\n\n")
        
        # Feature Analysis
        report_lines.append("## Feature Analysis\n\n")
        if feature_analysis.get('feature_importance'):
            report_lines.append("### Top Features by Model\n\n")
            for model_name, importance in feature_analysis['feature_importance'].items():
                report_lines.append(f"#### {model_name}\n\n")
                top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
                for feat, imp in top_features:
                    report_lines.append(f"- {feat}: {imp:.4f}\n")
                report_lines.append("\n")
        
        # Recommendations
        report_lines.append("## Recommendations for Improvement\n\n")
        
        # Identify common issues
        low_accuracy_models = [name for name, res in performance_results.items() 
                              if 'error' not in res and res['accuracy'] < 0.65]
        if low_accuracy_models:
            report_lines.append(f"### Models with Low Accuracy (<65%)\n\n")
            report_lines.append(f"The following models need significant improvement: {', '.join(low_accuracy_models)}\n\n")
            report_lines.append("**Recommendations:**\n")
            report_lines.append("- Increase training data\n")
            report_lines.append("- Improve feature engineering\n")
            report_lines.append("- Tune hyperparameters\n")
            report_lines.append("- Consider ensemble methods\n\n")
        
        # Feature recommendations
        report_lines.append("### Feature Engineering Recommendations\n\n")
        report_lines.append("- Analyze top features and create similar features\n")
        report_lines.append("- Remove or fix low-importance features\n")
        report_lines.append("- Check for feature correlation issues\n")
        report_lines.append("- Add domain-specific features (injuries, weather, etc.)\n\n")
        
        # Model-specific recommendations
        report_lines.append("### Model-Specific Recommendations\n\n")
        for model_name, weakness_data in model_weaknesses.items():
            if weakness_data['needs_improvement']:
                report_lines.append(f"**{model_name}**:\n")
                for weakness in weakness_data['weaknesses']:
                    if 'precision' in weakness.lower():
                        report_lines.append("- Increase regularization to reduce false positives\n")
                    elif 'recall' in weakness.lower():
                        report_lines.append("- Reduce threshold or adjust class weights\n")
                    elif 'ROC-AUC' in weakness:
                        report_lines.append("- Improve feature selection and engineering\n")
                report_lines.append("\n")
        
        # Write report
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.writelines(report_lines)
        
        logger.info(f"Report saved to {output_path}")


def load_test_data_from_training(seasons: List[int], include_2025_week15: bool = True):
    """Load test data using the same preparation as training script.
    
    Args:
        seasons: List of seasons
        include_2025_week15: Whether to filter 2025 to week 15
        
    Returns:
        Tuple of (X_test, y_test, test_features_df)
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from scripts.train_models import prepare_training_data
    
    logger.info("Preparing test data using training script logic...")
    try:
        X_train, y_train, X_val, y_val, X_test, y_test, feature_names, preprocessor, sample_weights = prepare_training_data(
            seasons,
            include_2025_week15=include_2025_week15,
            temporal_weighting=False  # Don't need weights for test
        )
        
        # Reconstruct test features DataFrame for analysis
        test_features_df = pd.DataFrame(X_test, columns=feature_names)
        
        logger.info(f"Loaded test data: {len(X_test)} samples, {len(feature_names)} features")
        return X_test, y_test, test_features_df
    except Exception as e:
        logger.error(f"Error loading test data: {e}", exc_info=True)
        raise


def main():
    """Main analysis function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze model weaknesses")
    parser.add_argument('--model-dir', default='models/trained',
                       help='Directory containing trained models')
    parser.add_argument('--seasons', nargs='+', type=int, default=None,
                       help='Seasons used for training (to reconstruct test data)')
    parser.add_argument('--include-2025-week15', action='store_true', default=True,
                       help='Include 2025 data through Week 15')
    parser.add_argument('--test-data', default=None,
                       help='Path to test data (if not provided, will use training data split)')
    parser.add_argument('--output', default='reports/model_analysis_report.md',
                       help='Output path for analysis report')
    parser.add_argument('--skip-test-analysis', action='store_true',
                       help='Skip test data analysis (faster, less comprehensive)')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = ModelWeaknessAnalyzer(args.model_dir)
    
    # Determine seasons from metadata if not provided
    if args.seasons is None and analyzer.metadata:
        args.seasons = analyzer.metadata.get('seasons', [2020, 2021, 2022, 2023, 2024])
    
    # Load test data if requested
    X_test = None
    y_test = None
    test_features_df = None
    
    if not args.skip_test_analysis and args.seasons:
        try:
            X_test, y_test, test_features_df = load_test_data_from_training(
                args.seasons, 
                include_2025_week15=args.include_2025_week15
            )
            logger.info(f"Loaded test data: {len(X_test)} samples")
        except Exception as e:
            logger.warning(f"Could not load test data: {e}")
            logger.info("Continuing with metadata-only analysis...")
    
    # If we have metadata, analyze what we have
    if analyzer.metadata:
        logger.info("\nMetadata Analysis:")
        logger.info(f"Trained on seasons: {analyzer.metadata.get('seasons', 'Unknown')}")
        logger.info(f"Number of features: {analyzer.metadata.get('num_features', 'Unknown')}")
        
        if 'model_accuracies' in analyzer.metadata:
            logger.info("\nModel Accuracies from Training:")
            for model_name, accuracy in analyzer.metadata['model_accuracies'].items():
                logger.info(f"  {model_name}: {accuracy:.1%}")
    
    # Feature analysis (doesn't require test data)
    feature_analysis = analyzer.analyze_features()
    
    # Create basic performance results from metadata
    performance_results = {}
    error_analysis = {}
    model_weaknesses = {}
    
    if analyzer.metadata and 'model_accuracies' in analyzer.metadata:
        for model_name, accuracy in analyzer.metadata['model_accuracies'].items():
            performance_results[model_name] = {
                'accuracy': accuracy,
                'precision': 0.0,  # Not available from metadata
                'recall': 0.0,
                'f1': 0.0,
                'roc_auc': 0.0,
                'confusion_matrix': {'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0}
            }
        model_weaknesses = analyzer.analyze_model_specific_weaknesses(performance_results)
    
    # Generate report
    output_path = Path(args.output)
    analyzer.generate_report(
        performance_results,
        error_analysis,
        feature_analysis,
        model_weaknesses,
        output_path
    )
    
    logger.info(f"\n✅ Analysis complete! Report saved to {output_path}")


if __name__ == "__main__":
    main()

