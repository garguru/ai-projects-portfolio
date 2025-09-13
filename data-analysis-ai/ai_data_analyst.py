"""
AI-Enhanced Data Analysis Assistant
Built for Garrita - Combining Chef precision with Data Science
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Any, Dict, List
import json

class AIDataAnalyst:
    """Your AI-powered data analysis assistant"""
    
    def __init__(self):
        """Initialize the analyst with chef-like mise en place"""
        self.data = None
        self.insights = []
        self.setup_style()
    
    def setup_style(self):
        """Set up beautiful visualizations"""
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load data from various formats"""
        if filepath.endswith('.csv'):
            self.data = pd.read_csv(filepath)
        elif filepath.endswith('.xlsx'):
            self.data = pd.read_excel(filepath)
        elif filepath.endswith('.json'):
            self.data = pd.read_json(filepath)
        
        print(f"✅ Data loaded: {self.data.shape[0]} rows, {self.data.shape[1]} columns")
        return self.data
    
    def auto_analyze(self) -> Dict[str, Any]:
        """Automatically analyze data and generate insights"""
        if self.data is None:
            return {"error": "No data loaded"}
        
        analysis = {
            "shape": self.data.shape,
            "columns": list(self.data.columns),
            "dtypes": self.data.dtypes.to_dict(),
            "missing_values": self.data.isnull().sum().to_dict(),
            "statistics": self.data.describe().to_dict()
        }
        
        # Find patterns (like finding flavor combinations in cooking)
        correlations = self.data.select_dtypes(include=[np.number]).corr()
        high_corr = []
        for i in range(len(correlations.columns)):
            for j in range(i+1, len(correlations.columns)):
                if abs(correlations.iloc[i, j]) > 0.7:
                    high_corr.append({
                        "var1": correlations.columns[i],
                        "var2": correlations.columns[j],
                        "correlation": correlations.iloc[i, j]
                    })
        
        analysis["high_correlations"] = high_corr
        
        # Generate insights
        self.generate_insights(analysis)
        
        return analysis
    
    def generate_insights(self, analysis: Dict):
        """Generate human-readable insights"""
        self.insights = []
        
        # Data size insight
        rows, cols = analysis["shape"]
        self.insights.append(f"📊 Your dataset contains {rows:,} records across {cols} variables")
        
        # Missing data insight
        missing = analysis["missing_values"]
        high_missing = [col for col, count in missing.items() if count > rows * 0.1]
        if high_missing:
            self.insights.append(f"⚠️ Columns with >10% missing data: {', '.join(high_missing)}")
        
        # Correlation insights
        if analysis["high_correlations"]:
            for corr in analysis["high_correlations"][:3]:
                self.insights.append(
                    f"🔗 Strong correlation ({corr['correlation']:.2f}) between "
                    f"{corr['var1']} and {corr['var2']}"
                )
    
    def visualize_distributions(self):
        """Create distribution plots for numeric columns"""
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        
        n_cols = min(len(numeric_cols), 6)
        if n_cols == 0:
            print("No numeric columns to visualize")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, col in enumerate(numeric_cols[:6]):
            self.data[col].hist(ax=axes[i], bins=30, edgecolor='black')
            axes[i].set_title(f'Distribution of {col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')
        
        # Hide empty subplots
        for i in range(n_cols, 6):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('distributions.png')
        plt.show()
        print("📈 Distribution plots saved as 'distributions.png'")
    
    def smart_query(self, question: str) -> str:
        """Answer questions about the data using AI-like logic"""
        question_lower = question.lower()
        
        if "average" in question_lower or "mean" in question_lower:
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            means = self.data[numeric_cols].mean()
            return f"Averages:\n{means.to_string()}"
        
        elif "correlation" in question_lower:
            corr_matrix = self.data.select_dtypes(include=[np.number]).corr()
            return f"Correlation Matrix:\n{corr_matrix.to_string()}"
        
        elif "missing" in question_lower:
            missing = self.data.isnull().sum()
            return f"Missing Values:\n{missing[missing > 0].to_string()}"
        
        else:
            return "I can help with: averages, correlations, missing values, and more. Try asking!"
    
    def export_report(self, filename: str = "analysis_report.html"):
        """Export a beautiful HTML report"""
        html = f"""
        <html>
        <head>
            <title>Data Analysis Report - Garrita</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #2c3e50; }}
                .insight {{ background: #f0f0f0; padding: 10px; margin: 10px 0; border-radius: 5px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #4CAF50; color: white; }}
            </style>
        </head>
        <body>
            <h1>AI-Enhanced Data Analysis Report</h1>
            <h2>Key Insights</h2>
        """
        
        for insight in self.insights:
            html += f'<div class="insight">{insight}</div>'
        
        html += """
            <h2>Data Preview</h2>
        """
        
        if self.data is not None:
            html += self.data.head(10).to_html()
        
        html += """
        </body>
        </html>
        """
        
        with open(filename, 'w') as f:
            f.write(html)
        
        print(f"📄 Report exported to {filename}")

# Example usage
if __name__ == "__main__":
    print("🎯 AI Data Analyst Ready!")
    print("=" * 50)
    
    analyst = AIDataAnalyst()
    
    # Create sample data for demonstration
    sample_data = pd.DataFrame({
        'sales': np.random.randn(100) * 1000 + 5000,
        'temperature': np.random.randn(100) * 10 + 20,
        'customers': np.random.randint(50, 200, 100),
        'satisfaction': np.random.uniform(3, 5, 100)
    })
    
    analyst.data = sample_data
    
    # Run analysis
    results = analyst.auto_analyze()
    
    print("\n🔍 Insights Generated:")
    for insight in analyst.insights:
        print(f"  {insight}")
    
    # Create visualizations
    analyst.visualize_distributions()
    
    # Export report
    analyst.export_report()