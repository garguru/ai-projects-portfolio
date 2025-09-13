# 📊 Data Analysis AI Project

> Professional cryptocurrency data analysis pipeline with AI-powered insights

## 🏗️ Project Structure

```
data-analysis-ai/
│
├── src/                      # Source code
│   ├── ai_data_analyst.py
│   ├── crypto_ai_analysis.py
│   ├── coingecko_api.py
│   └── top100_*.py
│
├── data/                     # Data directory
│   ├── raw/                 # Original data (read-only)
│   └── processed/           # Cleaned/transformed data
│
├── results/                  # Output directory
│   ├── figures/             # Generated charts
│   └── data/                # Analysis results
│
├── tests/                    # Test files
│   └── test_analysis.py
│
├── configs/                  # Configuration
│   └── config.py
│
├── docs/                     # Documentation
│
├── run_pipeline.py          # Master pipeline script
├── requirements.txt         # Python dependencies
├── setup.py                 # Installation script
└── README.md               # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd data-analysis-ai

# Create virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Complete Pipeline

```bash
# Run all analysis steps
python run_pipeline.py

# Or run specific steps
python run_pipeline.py --steps collect crypto top100
python run_pipeline.py --steps report
```

### 3. Pipeline Steps

The pipeline executes in this order:

1. **Data Collection** - Fetches latest cryptocurrency data
2. **Crypto Analysis** - Analyzes selected cryptocurrencies
3. **Top 100 Analysis** - Processes top 100 crypto market
4. **Report Generation** - Creates summary report

## 📋 Features

### ✅ Implemented Best Practices

- **Organized Structure**: Clean separation of code, data, and results
- **Reproducibility**: Fixed random seeds, requirements.txt
- **Pipeline Design**: Master script orchestrates all steps
- **Configuration**: Centralized settings in config.py
- **Logging**: Complete execution tracking
- **Documentation**: Clear instructions and structure

### 🎯 Analysis Capabilities

- Real-time cryptocurrency data fetching
- Technical indicator calculations (MA, RSI, Volatility)
- AI-powered trading recommendations
- Top 100 cryptocurrency market analysis
- Interactive visualizations and dashboards
- Automated report generation

## 🔧 Configuration

Edit `configs/config.py` to customize:

- Analysis parameters (timeframes, indicators)
- Data sources and API keys
- Output directories
- Random seeds for reproducibility

## 📊 Output Files

After running the pipeline, find your results in:

- `results/figures/` - Charts and visualizations
- `results/data/` - CSV files with analysis results
- `analysis.log` - Detailed execution log
- `analysis_report.md` - Summary report

## 🧪 Testing

Run tests to verify functionality:

```bash
# Run all tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_analysis.py
```

## 🔄 Reproducibility

This project ensures reproducible results through:

1. **Version Control**: All code tracked in Git
2. **Dependency Management**: Pinned versions in requirements.txt
3. **Random Seeds**: Consistent randomization (seed=42)
4. **Automated Pipeline**: Single command execution
5. **No Manual Steps**: Everything scripted

To reproduce results:
```bash
# Clean previous results
rm -rf results/*

# Run pipeline with same seed
python run_pipeline.py --seed 42
```

## 📈 Example Usage

### Basic Analysis
```python
from src.crypto_ai_analysis import CryptoAIAnalyzer

analyzer = CryptoAIAnalyzer()
analyzer.fetch_crypto_data(['BTC-USD'], days=30)
analyzer.calculate_indicators('BTC-USD')
recommendation = analyzer.ai_trading_recommendation('BTC-USD')
```

### Custom Pipeline
```python
import sys
sys.path.insert(0, 'src')
sys.path.insert(0, 'configs')

from config import ANALYSIS_CONFIG
from coingecko_api import fetch_top_100_crypto_data

# Your custom analysis here
data = fetch_top_100_crypto_data()
# Process data...
```

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Module not found | Ensure you're in project root and src is in Python path |
| API errors | Check internet connection and API keys in config |
| Missing dependencies | Run `pip install -r requirements.txt` |
| Permission errors | Check file permissions in data directories |

## 📝 Development Workflow

1. Make changes to source files in `src/`
2. Test changes: `python -m pytest tests/`
3. Run pipeline: `python run_pipeline.py`
4. Check results in `results/`
5. Commit changes to Git

## 🎓 Learning Path

This project demonstrates:
- **Clean Code**: Modular, documented, testable
- **Data Science Best Practices**: Reproducible analysis
- **Project Organization**: Professional structure
- **Pipeline Design**: Automated workflows
- **Version Control**: Git for tracking changes

## 📚 Next Steps

- [ ] Add more cryptocurrencies to analysis
- [ ] Implement machine learning predictions
- [ ] Create real-time dashboard
- [ ] Add more technical indicators
- [ ] Integrate additional data sources

## 👤 Author

**Garrita** - AI Developer & Former Chef
- Focus: Cryptocurrency analysis and AI systems
- Location: Uruguay

---

*"From mise en place to clean code - organizing data like a kitchen"*