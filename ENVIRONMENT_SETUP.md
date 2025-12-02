# Environment Setup Guide

## Setting Up Your Anthropic API Key

The NFL Betting Agent Council uses Claude (via Anthropic's API) for the agent debate system. You need an API key to use the system.

### Step 1: Get an API Key

1. Go to https://console.anthropic.com/
2. Sign up or log in
3. Navigate to API Keys section
4. Create a new API key
5. Copy the key (you won't be able to see it again)

### Step 2: Configure the Environment

You have several options for providing the API key:

#### Option A: Using .env file (Recommended)

Create a `.env` file in the project root:

```bash
# Create .env file
touch .env

# Add your API key
echo "ANTHROPIC_API_KEY=your_api_key_here" >> .env
```

Replace `your_api_key_here` with your actual API key.

#### Option B: Export as Environment Variable

For a single session:

```bash
export ANTHROPIC_API_KEY=your_api_key_here
```

To make it permanent, add to your shell profile:

**For bash (~/.bashrc or ~/.bash_profile):**
```bash
echo 'export ANTHROPIC_API_KEY=your_api_key_here' >> ~/.bashrc
source ~/.bashrc
```

**For zsh (~/.zshrc):**
```bash
echo 'export ANTHROPIC_API_KEY=your_api_key_here' >> ~/.zshrc
source ~/.zshrc
```

#### Option C: System-wide Environment Variable (macOS/Linux)

Add to `/etc/environment`:
```bash
ANTHROPIC_API_KEY=your_api_key_here
```

### Step 3: Verify Setup

Test that your API key is configured correctly:

```python
from src.utils.config_loader import get_config

try:
    config = get_config()
    api_key = config.anthropic_api_key
    print("✓ API key found!")
except ValueError as e:
    print(f"✗ Error: {e}")
```

Or use the command line:

```bash
python -c "from src.utils.config_loader import get_config; print('API Key:', get_config().anthropic_api_key[:10] + '...')"
```

## Python Environment

### Recommended Setup

1. **Python Version**: 3.8 or higher (3.10+ recommended)

2. **Virtual Environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

### Optional: Using Conda

If you prefer conda:

```bash
conda create -n betting-council python=3.10
conda activate betting-council
pip install -r requirements.txt
```

## GPU Support (Optional)

For faster neural network training, you can use GPU acceleration:

### CUDA (NVIDIA GPUs)

```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### MPS (Apple Silicon)

PyTorch automatically uses Metal Performance Shaders on M1/M2 Macs. No additional setup needed.

## Configuration Files

### config/config.yaml

This file controls system behavior. Key sections:

```yaml
# LLM Configuration
anthropic:
  model: "claude-3-5-sonnet-20241022"
  temperature: 0.7

# Model Training
models:
  neural_net:
    hidden_layers: [256, 128, 64]
    learning_rate: 0.001

# Debate Settings
debate:
  rounds: 4
  temperature: 0.7

# Betting Parameters
betting:
  confidence_threshold: 0.65
  kelly_criterion_fraction: 0.25
```

You can modify these values without changing code.

## Data Storage

The system stores data in these locations:

- `data/raw/`: Downloaded NFL data (cached)
- `data/processed/`: Processed features
- `models/`: Trained model weights
- `logs/`: System and debate logs

Make sure you have sufficient disk space (~1-2 GB for a few seasons of data).

## Troubleshooting

### Import Errors

If you get import errors:

```bash
# Make sure you're in the project root
cd /path/to/SportsBetting

# Run with -m to ensure proper imports
python -m src.cli analyze ...
```

### API Rate Limits

Anthropic has rate limits. If you hit them:

1. Add delays between analyses
2. Reduce debate rounds in config
3. Upgrade your API plan

### Memory Issues

If training fails due to memory:

1. Reduce batch size in models config
2. Train on fewer seasons at once
3. Use smaller neural network architectures

### Permission Errors

If you get permission errors on data files:

```bash
chmod -R u+w data/
chmod -R u+w models/
chmod -R u+w logs/
```

## Security Notes

1. **Never commit `.env` to Git**: The `.env` file is in `.gitignore` by default
2. **Rotate API keys regularly**: Especially if shared or exposed
3. **Use read-only keys if available**: Anthropic doesn't have separate read/write keys, but be careful with key sharing

## Development Setup

For development work:

```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest pytest-asyncio black flake8 mypy

# Run tests
pytest tests/

# Format code
black src/ tests/

# Type checking
mypy src/
```

## Next Steps

After setup:

1. Verify API key: `python -c "from src.utils.config_loader import get_config; get_config().anthropic_api_key"`
2. Train models: `python scripts/train_models.py --seasons 2020 2021 2022 2023`
3. Run analysis: `python -m src.cli analyze --help`

## Getting Help

- Check `QUICKSTART.md` for usage examples
- See `README.md` for architecture overview
- Review `notebooks/example_usage.ipynb` for interactive examples

