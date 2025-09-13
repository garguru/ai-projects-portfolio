#!/usr/bin/env python
"""
5-Minute Demo Script for Data Analysis AI
Shows the complete workflow in a simple, runnable example
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "configs"))

print("🚀 Data Analysis AI - Quick Demo")
print("=" * 50)

def create_sample_data():
    """Create sample crypto data for demo purposes"""
    print("📊 Creating sample cryptocurrency data...")

    # Generate 30 days of sample crypto price data
    dates = pd.date_range(start=datetime.now() - timedelta(days=30),
                         end=datetime.now(), freq='D')

    # Simulate realistic crypto price movements
    np.random.seed(42)  # For reproducibility
    base_price = 45000  # Starting BTC price
    prices = []

    for i in range(len(dates)):
        # Random walk with slight upward trend
        change = np.random.normal(0.001, 0.03)  # 0.1% mean, 3% std
        base_price *= (1 + change)
        prices.append(base_price)

    # Create sample data
    sample_data = pd.DataFrame({
        'date': dates,
        'price': prices,
        'volume': np.random.lognormal(15, 0.5, len(dates)),  # Realistic volume
        'market_cap': np.array(prices) * 19500000,  # Approximate BTC supply
        'symbol': 'BTC-USD'
    })

    # Save sample data
    data_file = project_root / "data" / "raw" / "sample_crypto.csv"
    sample_data.to_csv(data_file, index=False)
    print(f"✅ Sample data saved to {data_file}")

    return sample_data

def analyze_data(df):
    """Simple but realistic data analysis"""
    print("\n🔍 Running analysis...")

    # Basic statistics
    stats = {
        'total_days': len(df),
        'avg_price': df['price'].mean(),
        'price_volatility': df['price'].std() / df['price'].mean(),
        'total_volume': df['volume'].sum(),
        'price_change': (df['price'].iloc[-1] - df['price'].iloc[0]) / df['price'].iloc[0],
        'max_price': df['price'].max(),
        'min_price': df['price'].min()
    }

    # Technical indicators (simplified)
    df['ma_7'] = df['price'].rolling(7).mean()
    df['ma_21'] = df['price'].rolling(21).mean()
    df['returns'] = df['price'].pct_change()
    df['volatility'] = df['returns'].rolling(7).std()

    return stats, df

def create_visualization(df, stats):
    """Create a professional-looking chart"""
    print("📈 Creating visualization...")

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Cryptocurrency Analysis Dashboard - Demo', fontsize=16, fontweight='bold')

    # Price chart with moving averages
    axes[0, 0].plot(df['date'], df['price'], label='Price', linewidth=2, color='orange')
    axes[0, 0].plot(df['date'], df['ma_7'], label='7-day MA', alpha=0.7, color='blue')
    axes[0, 0].plot(df['date'], df['ma_21'], label='21-day MA', alpha=0.7, color='red')
    axes[0, 0].set_title('Price with Moving Averages')
    axes[0, 0].legend()
    axes[0, 0].tick_params(axis='x', rotation=45)

    # Volume chart
    axes[0, 1].bar(df['date'], df['volume'], alpha=0.6, color='green')
    axes[0, 1].set_title('Trading Volume')
    axes[0, 1].tick_params(axis='x', rotation=45)

    # Returns distribution
    axes[1, 0].hist(df['returns'].dropna(), bins=20, alpha=0.7, color='purple')
    axes[1, 0].set_title('Daily Returns Distribution')
    axes[1, 0].set_xlabel('Daily Return')
    axes[1, 0].set_ylabel('Frequency')

    # Key metrics text
    metrics_text = f"""Key Metrics:

Average Price: ${stats['avg_price']:,.0f}
Price Change: {stats['price_change']:.1%}
Volatility: {stats['price_volatility']:.1%}
Max Price: ${stats['max_price']:,.0f}
Min Price: ${stats['min_price']:,.0f}
Total Volume: {stats['total_volume']:,.0f}"""

    axes[1, 1].text(0.1, 0.9, metrics_text, transform=axes[1, 1].transAxes,
                   fontsize=10, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    axes[1, 1].set_title('Analysis Summary')
    axes[1, 1].axis('off')

    plt.tight_layout()

    # Save chart
    chart_file = project_root / "results" / "figures" / "sample_analysis.png"
    chart_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(chart_file, dpi=100, bbox_inches='tight')
    plt.show()

    print(f"✅ Chart saved to {chart_file}")
    return chart_file

def save_results(stats, df):
    """Save analysis results"""
    print("💾 Saving results...")

    # Save summary statistics
    results_dir = project_root / "results" / "data"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Convert stats to DataFrame for easy CSV export
    summary_df = pd.DataFrame([stats])
    summary_file = results_dir / "crypto_summary.csv"
    summary_df.to_csv(summary_file, index=False)

    # Save processed data with indicators
    processed_file = results_dir / "processed_sample_data.csv"
    df.to_csv(processed_file, index=False)

    print(f"✅ Summary saved to {summary_file}")
    print(f"✅ Processed data saved to {processed_file}")

    return summary_file, processed_file

def print_insights(stats):
    """Print AI-style insights"""
    print("\n🤖 AI Analysis Insights:")
    print("-" * 30)

    # Trend analysis
    if stats['price_change'] > 0.1:
        trend = "📈 Strong upward trend"
    elif stats['price_change'] > 0:
        trend = "📊 Slight upward trend"
    elif stats['price_change'] > -0.1:
        trend = "📉 Slight downward trend"
    else:
        trend = "📉 Strong downward trend"

    # Volatility analysis
    if stats['price_volatility'] > 0.05:
        volatility_desc = "High volatility - risky but potential for gains"
    elif stats['price_volatility'] > 0.03:
        volatility_desc = "Moderate volatility - normal market behavior"
    else:
        volatility_desc = "Low volatility - stable price movement"

    print(f"• Price Movement: {trend} ({stats['price_change']:.1%})")
    print(f"• Volatility: {volatility_desc} ({stats['price_volatility']:.1%})")
    print(f"• Price Range: ${stats['min_price']:,.0f} - ${stats['max_price']:,.0f}")

    # Trading recommendation (simplified)
    if stats['price_change'] > 0 and stats['price_volatility'] < 0.05:
        recommendation = "🟢 HOLD - Positive trend with stable volatility"
    elif stats['price_change'] > 0.05:
        recommendation = "🟡 CONSIDER SELLING - Take profits on strong gains"
    elif stats['price_change'] < -0.05:
        recommendation = "🟡 CONSIDER BUYING - Potential oversold condition"
    else:
        recommendation = "🟡 NEUTRAL - Monitor for clear signals"

    print(f"• Recommendation: {recommendation}")

def main():
    """Run the complete demo"""
    try:
        # Step 1: Create sample data
        df = create_sample_data()

        # Step 2: Analyze data
        stats, df_analyzed = analyze_data(df)

        # Step 3: Create visualization
        chart_file = create_visualization(df_analyzed, stats)

        # Step 4: Save results
        summary_file, processed_file = save_results(stats, df_analyzed)

        # Step 5: Show insights
        print_insights(stats)

        # Demo completion summary
        print("\n" + "=" * 50)
        print("🎉 Demo completed successfully!")
        print("=" * 50)
        print("📁 Generated files:")
        print(f"  • Chart: {chart_file}")
        print(f"  • Summary: {summary_file}")
        print(f"  • Data: {processed_file}")
        print("\n💡 Next steps:")
        print("  • Run full pipeline: python run_pipeline.py")
        print("  • Explore notebooks: notebooks/00_intro.ipynb")
        print("  • Run tests: python -m pytest tests/")

    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure you're in the project root directory and have installed dependencies.")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())