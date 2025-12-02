# Quick Training Checklist

Before running training, verify these items:

## ✅ Pre-Flight Checks (5 minutes)

### 1. Environment Setup
```bash
# Activate virtual environment
source venv/bin/activate  # or: venv\Scripts\activate on Windows

# Verify Python version (need 3.8+)
python --version

# Install/update dependencies
pip install -r requirements.txt
```

### 2. Quick Data Test
```bash
# Test data collection (should complete without errors)
python -c "
from src.data.collectors.nfl_data_collector import NFLDataCollector
collector = NFLDataCollector()
schedule = collector.get_schedule([2023])
print(f'✓ Collected {len(schedule)} games for 2023')
"
```

### 3. Verify Models Can Import
```bash
# Test all model imports
python -c "
from src.models.neural_nets.deep_predictor import DeepPredictor
from src.models.traditional.gradient_boost_model import GradientBoostModel
from src.models.traditional.random_forest_model import RandomForestModel
from src.models.traditional.statistical_model import StatisticalModel
from src.models.traditional.lightgbm_model import LightGBMModel
from src.models.traditional.svm_model import SVMModel
print('✓ All 6 models imported successfully')
"
```

## 🚀 Ready to Train

If all checks pass, you're ready to train:

```bash
# Start with a small test (1 season)
python scripts/train_models.py --seasons 2023 --output-dir models/test

# If successful, train on full dataset
python scripts/train_models.py --seasons 2020 2021 2022 2023 --output-dir models
```

## ⚠️ Common Issues & Quick Fixes

### Issue: "ModuleNotFoundError"
**Fix:** Make sure you're in the project root and virtual environment is activated

### Issue: "nfl-data-py not found"
**Fix:** `pip install nfl-data-py`

### Issue: "LightGBM import error"
**Fix:** `pip install lightgbm`

### Issue: "Out of memory"
**Fix:** Use `--no-parallel` flag or train on fewer seasons

### Issue: "No features extracted"
**Fix:** Check internet connection and verify NFL data API is accessible

## Expected Training Time

- **1 season**: ~3-5 minutes
- **4 seasons**: ~15-30 minutes (parallel) or ~30-45 minutes (sequential)

## What Success Looks Like

After training completes, you should see:
- ✅ 6 model files (`.pkl`) in output directory
- ✅ `preprocessor.pkl` file
- ✅ Training summary with metrics for each model
- ✅ No error messages

## Next Steps After Training

1. Verify models loaded:
   ```python
   from src.pipeline.predictor import BettingCouncil
   council = BettingCouncil()
   council.load_models("models")
   print(f"Loaded {len(council.models)} models")
   ```

2. Test a prediction (optional - requires API key for debate)

3. Review training metrics in the logs

