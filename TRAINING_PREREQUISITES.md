# Training Prerequisites Checklist

This document lists all items that need to be completed before you can successfully train the models.

## ✅ Completed Items

### Model Implementation
- ✅ All 9 models implemented:
  1. DeepPredictor (Neural Network)
  2. GradientBoostModel (XGBoost)
  3. RandomForestModel
  4. StatisticalModel (Logistic Regression)
  5. LightGBMModel
  6. SVMModel
  7. CatBoostModel
  8. EnsembleModel (Ensemble Council)
  9. StackingModel (Stacking Meta-Learner)
- ✅ All models follow BaseModel interface
- ✅ Feature preparation moved to BaseModel (eliminates duplication)
- ✅ DeepPredictor bugs fixed (outcome mapping, output layer, in-place operations)
- ✅ Early stopping added to neural network
- ✅ XGBoost and LightGBM segmentation faults fixed

### Training Infrastructure
- ✅ DataPreprocessor integrated into training script
- ✅ Parallel training support added
- ✅ Comprehensive evaluation metrics (accuracy, precision, recall, F1, ROC-AUC)
- ✅ Data validation function added
- ✅ Feature caching implemented in FeatureEngineer

### Configuration
- ✅ Config.yaml updated with all 6 model configurations
- ✅ Agent personalities added for new models
- ✅ BettingCouncil updated to include all 6 models

## ⚠️ Items to Complete Before Training

### 1. Environment Setup
- [ ] Python 3.8+ installed
- [ ] Virtual environment created and activated
- [ ] All dependencies installed: `pip install -r requirements.txt`
- [ ] Verify LightGBM is installed (already in requirements.txt)

### 2. Data Collection Verification
- [ ] Test NFL data collector works:
  ```python
  from src.data.collectors.nfl_data_collector import NFLDataCollector
  collector = NFLDataCollector()
  schedule = collector.get_schedule([2023])
  print(f"Collected {len(schedule)} games")
  ```
- [ ] Verify data quality (no missing critical fields)
- [ ] Check internet connection for data downloads
- [ ] Ensure sufficient disk space (~1-2 GB for multiple seasons)

### 3. Data Quality Checks
- [ ] Verify feature engineering works on sample data
- [ ] Check for edge cases:
  - First games of season (no historical data)
  - New teams (limited history)
  - Missing player stats
- [ ] Test data validation function with sample data

### 4. Configuration Verification
- [ ] Review `config/config.yaml` settings
- [ ] Adjust model hyperparameters if needed
- [ ] Set ensemble.enabled to true if you want ensemble model
- [ ] Verify all model configs are present

### 5. Training Script Testing
- [ ] Test with small dataset first (1 season):
  ```bash
  python scripts/train_models.py --seasons 2023 --output-dir models/test
  ```
- [ ] Verify all 6 models train successfully
- [ ] Check that models are saved correctly
- [ ] Verify preprocessor is saved

### 6. Resource Requirements
- [ ] Sufficient RAM (8GB+ recommended for parallel training)
- [ ] Sufficient disk space for models (~500MB-1GB)
- [ ] Training time estimate: 10-30 minutes for 3-4 seasons
- [ ] GPU optional but recommended for neural network (CUDA/MPS)

### 7. Error Handling
- [ ] Test error handling for:
  - Missing data files
  - Network failures during data download
  - Insufficient memory
  - Invalid season years

## Quick Start Training Command

Once prerequisites are met:

```bash
# Train on multiple seasons (recommended)
python scripts/train_models.py --seasons 2020 2021 2022 2023 --output-dir models

# Train sequentially (if parallel causes issues)
python scripts/train_models.py --seasons 2020 2021 2022 2023 --output-dir models --no-parallel

# Train on single season (for testing)
python scripts/train_models.py --seasons 2023 --output-dir models/test
```

## Expected Output

After successful training, you should see:
- 6 model files in the output directory (`.pkl` files)
- `preprocessor.pkl` file
- Training summary with metrics for each model
- No errors in the logs

## Troubleshooting

### Common Issues

1. **Import Errors**
   - Ensure you're in the project root directory
   - Activate virtual environment
   - Run: `pip install -r requirements.txt`

2. **Data Collection Fails**
   - Check internet connection
   - Verify `nfl-data-py` package is installed
   - Check if NFL API is accessible

3. **Memory Errors**
   - Reduce number of seasons
   - Train models sequentially: `--no-parallel`
   - Reduce batch size in config

4. **Model Training Fails**
   - Check data quality (run validation)
   - Verify feature engineering produces valid features
   - Check logs for specific error messages

5. **SVM Training Slow**
   - SVM is slower than other models
   - Consider training it separately if needed
   - Or reduce training data size for SVM

## Next Steps After Training

1. Verify models loaded correctly:
   ```python
   from src.pipeline.predictor import BettingCouncil
   council = BettingCouncil()
   council.load_models("models")
   ```

2. Test predictions on sample games

3. Run backtesting to evaluate performance

4. Set up Anthropic API key for debate functionality

## Model Count Verification

You should have **9 models total**:
1. Neural Analyst (DeepPredictor)
2. Gradient Strategist (GradientBoostModel)
3. Forest Evaluator (RandomForestModel)
4. Statistical Conservative (StatisticalModel)
5. LightGBM Optimizer (LightGBMModel)
6. SVM Strategist (SVMModel)
7. CatBoost Optimizer (CatBoostModel)
8. Ensemble Council (EnsembleModel)
9. Stacking Meta-Learner (StackingModel)

## Efficiency Improvements Made

1. ✅ Eliminated code duplication (feature preparation in BaseModel)
2. ✅ Added caching to feature engineering
3. ✅ Parallel training support
4. ✅ Early stopping for neural network
5. ✅ Comprehensive evaluation metrics
6. ✅ Data validation before training
7. ✅ Feature standardization via DataPreprocessor

