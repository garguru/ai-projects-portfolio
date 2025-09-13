# AI Projects Portfolio — Data Analysis AI

> Professional cryptocurrency analysis and AI-powered data science projects

## 🚀 Quickstart (Python 3.11+)

```bash
# Clone and setup
git clone https://github.com/garguru/ai-projects-portfolio.git
cd ai-projects-portfolio

# Install UV (modern Python package manager)
pip install uv

# Setup environment
uv venv
uv pip install -r data-analysis-ai/requirements.txt

# Run a demo
python data-analysis-ai/scripts/run_demo.py
```

## 📊 Project Map

```
ai-projects-portfolio/
├── data-analysis-ai/          # Cryptocurrency analysis pipeline
│   ├── src/                   # Reusable code (analysis, plotting, data I/O)
│   ├── scripts/               # run_demo.py, data ingestion scripts
│   ├── notebooks/             # EDA & interactive analysis
│   ├── data/                  # raw/, processed/ (sample data included)
│   ├── results/               # Generated charts and reports
│   ├── tests/                 # Unit tests and validation
│   └── run_pipeline.py        # Complete analysis pipeline
│
├── deeplearningai_rag/        # RAG system with ChromaDB
│   ├── backend/               # FastAPI server with AI integration
│   ├── frontend/              # Web interface
│   └── docs/                  # Course materials for RAG processing
│
└── notebooks/                 # Jupyter demonstrations
```

## 🎯 Learning Path

### For Data Analysis:
1. **Start here**: `data-analysis-ai/notebooks/00_intro.ipynb`
2. **Run the demo**: `python data-analysis-ai/scripts/run_demo.py`
3. **Explore modules**: Read comments in `data-analysis-ai/src/`
4. **Full pipeline**: `python data-analysis-ai/run_pipeline.py`

### For RAG Systems:
1. **Setup**: `cd deeplearningai_rag && uv run uvicorn app:app --reload`
2. **Web interface**: http://localhost:8000
3. **Query documents**: Ask questions about the course materials

## 🛠️ Features

### Data Analysis AI
- **Real-time crypto data** fetching and analysis
- **Technical indicators** (MA, RSI, volatility)
- **AI-powered insights** and trading recommendations
- **Automated reporting** with visualizations
- **Reproducible pipeline** (single command execution)

### RAG System
- **ChromaDB vector storage** for semantic search
- **Course material processing** from multiple formats
- **Anthropic Claude integration** for AI responses
- **Web interface** for interactive queries

## 🔧 Quick Commands

```bash
# Data analysis demo (5 minutes)
python data-analysis-ai/scripts/run_demo.py

# Full crypto analysis pipeline
python data-analysis-ai/run_pipeline.py

# Run tests
python -m pytest data-analysis-ai/tests/

# Start RAG system
cd deeplearningai_rag && uv run uvicorn app:app --reload
```

## 📈 Sample Outputs

After running the demo, you'll get:
- `results/sample_analysis.png` - Market visualization
- `results/crypto_summary.csv` - Analysis metrics
- Console output with key insights

## 👤 Author

**Garrita** - AI Developer & Former Chef
- 🌍 Location: Uruguay
- 🎯 Focus: AI systems, cryptocurrency analysis, data science
- 🏆 Journey: 3-Michelin-star chef → AI developer

## 🎓 What You'll Learn

This portfolio demonstrates:
- **Professional project structure** and organization
- **Reproducible data science** workflows
- **AI integration** patterns (RAG, analysis, insights)
- **Modern Python tooling** (uv, pytest, FastAPI)
- **Clean code practices** for data science

---

*"From mise en place to clean code - organizing data like a professional kitchen"*