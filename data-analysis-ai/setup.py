# AI-Enhanced Data Analysis Setup for Garrita
# This script sets up your complete data analysis environment

import subprocess
import sys

def install_packages():
    """Install essential data analysis packages"""
    packages = [
        # Core data analysis
        'pandas',
        'numpy',
        'scipy',
        
        # Visualization
        'matplotlib',
        'seaborn',
        'plotly',
        
        # Machine Learning
        'scikit-learn',
        'xgboost',
        
        # AI/LLM integration
        'openai',
        'langchain',
        'chromadb',
        
        # Data sources
        'yfinance',  # Financial data
        'pandas-datareader',
        'openpyxl',  # Excel support
        
        # Jupyter for interactive analysis
        'jupyter',
        'notebook',
        'ipywidgets'
    ]
    
    for package in packages:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
    
    print("\n✅ All packages installed successfully!")

if __name__ == "__main__":
    install_packages()