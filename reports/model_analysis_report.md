# Model Weakness Analysis Report
Generated: 2025-12-17 18:34:28

## Performance Summary

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|----|---------|
| SVM Strategist | 69.8% | 0.0% | 0.0% | 0.0% | 0.000 |
| Statistical Conservative | 67.7% | 0.0% | 0.0% | 0.0% | 0.000 |
| Stacking Meta-Learner | 67.4% | 0.0% | 0.0% | 0.0% | 0.000 |

## Best and Worst Performing Models

**Best Model**: SVM Strategist (69.8% accuracy)

**Worst Model**: Stacking Meta-Learner (67.4% accuracy)

## Model-Specific Weaknesses

### Statistical Conservative

- Very low precision - many false positives
- Very low recall - many false negatives
- Poor discrimination (ROC-AUC < 0.6)

### SVM Strategist

- Very low precision - many false positives
- Very low recall - many false negatives
- Poor discrimination (ROC-AUC < 0.6)

### Stacking Meta-Learner

- Very low precision - many false positives
- Very low recall - many false negatives
- Poor discrimination (ROC-AUC < 0.6)

## Error Pattern Analysis

## Feature Analysis

### Top Features by Model

#### Statistical Conservative

- away_rush_yards_per_game: 0.0340
- spread: 0.0303
- spread_x_home_offense: 0.0292
- implied_home_prob: 0.0241
- away_ml: 0.0241
- home_off_epa_x_away_def_epa: 0.0240
- is_primetime: 0.0230
- implied_away_prob: 0.0225
- away_away_record: 0.0212
- away_rush_off_vs_home_rush_def: 0.0208

## Recommendations for Improvement

### Feature Engineering Recommendations

- Analyze top features and create similar features
- Remove or fix low-importance features
- Check for feature correlation issues
- Add domain-specific features (injuries, weather, etc.)

### Model-Specific Recommendations

**Statistical Conservative**:
- Increase regularization to reduce false positives
- Reduce threshold or adjust class weights
- Improve feature selection and engineering

**SVM Strategist**:
- Increase regularization to reduce false positives
- Reduce threshold or adjust class weights
- Improve feature selection and engineering

**Stacking Meta-Learner**:
- Increase regularization to reduce false positives
- Reduce threshold or adjust class weights
- Improve feature selection and engineering

