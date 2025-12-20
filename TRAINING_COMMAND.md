# Training Command - Ready to Run

## Fixed Issues

✅ **Neural Network validation error** - Fixed tensor scalar conversion  
✅ **Parallel training silent failures** - Added better error handling  
✅ **Sequential training by default** - More reliable, shows progress  
✅ **Better logging** - Shows which model is training and progress  

## Run This Command

In your terminal, run:

```bash
cd /Users/granthernandez/Documents/code/SportsBetting

PYTHONPATH=/Users/granthernandez/Documents/code/SportsBetting \
python scripts/train_models.py \
  --seasons 2020 2021 2022 2023 2024 \
  --output-dir models/trained \
  2>&1 | tee training_run_final.log
```

## What Will Happen

1. **Data Preparation** (~5-10 min)
   - Load schedule, team stats, player stats, rosters
   - Extract features from 1408 games
   - Feature selection (top 75 features)

2. **Model Training** (Sequential - ~1-2 hours total)
   - Neural Analyst (deep network) - ~10-20 min
   - Statistical Conservative - ~1 min
   - SVM Strategist - ~2-5 min
   - Gradient Strategist (XGBoost) - ~5-10 min
   - Forest Evaluator (Random Forest) - ~5-10 min
   - LightGBM Optimizer - ~5-10 min
   - Ensemble Council - ~2-5 min
   - Stacking Meta-Learner - ~10-20 min

3. **Final Summary**
   - Shows accuracy for all models
   - Saves all models to `models/trained/`

## Monitor Progress

While training, you can check progress:

```bash
# See latest activity
tail -f training_run_final.log

# Check which models completed
grep -E "completed|Accuracy|Saved" training_run_final.log

# Check for errors
grep -i "error" training_run_final.log
```

## Expected Results

- All models should train successfully
- Sequential training is more reliable
- Better error messages if something fails
- Progress logging shows which model is training

## When You Return

Check completion:

```bash
tail -100 training_run_final.log | grep -E "TRAINING SUMMARY|Successfully trained|Failed"
ls -lh models/trained/*.pkl
```

If you see "TRAINING SUMMARY" and multiple `.pkl` files, training completed successfully!

