"""
Top 100 Cryptocurrencies Analysis with Technical Indicators
For Garrita - January 31, 2025 Snapshot
"""

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import time
import warnings
warnings.filterwarnings('ignore')

class Top100CryptoAnalyzer:
    """Analyze top 100 cryptocurrencies with technical indicators"""
    
    def __init__(self):
        self.base_url = "https://api.coingecko.com/api/v3"
        self.data = None
        
    def fetch_top_100(self):
        """Fetch top 100 cryptocurrencies by market cap"""
        print("Fetching top 100 cryptocurrencies...")
        
        # We'll need to make 2 API calls (50 each) due to pagination
        all_coins = []
        
        for page in [1, 2]:
            params = {
                'vs_currency': 'usd',
                'order': 'market_cap_desc',
                'per_page': 50,
                'page': page,
                'sparkline': True,  # Get 7-day price data for indicators
                'price_change_percentage': '1h,24h,7d,14d,30d'
            }
            
            try:
                response = requests.get(f"{self.base_url}/coins/markets", params=params)
                response.raise_for_status()
                coins = response.json()
                all_coins.extend(coins)
                print(f"  Fetched page {page}/2")
                time.sleep(2)  # Be nice to the API
            except Exception as e:
                print(f"Error fetching page {page}: {e}")
        
        # Convert to DataFrame
        self.data = pd.DataFrame(all_coins)
        print(f"Successfully fetched {len(self.data)} cryptocurrencies")
        return self.data
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI for given price series"""
        if len(prices) < period:
            return np.nan
            
        deltas = np.diff(prices)
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        
        if down == 0:
            return 100
        
        rs = up / down
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_macd(self, prices):
        """Calculate MACD (12-day EMA - 26-day EMA)"""
        if len(prices) < 26:
            return np.nan
            
        # Simple implementation using available data
        ema_12 = pd.Series(prices).ewm(span=12, adjust=False).mean().iloc[-1]
        ema_26 = pd.Series(prices).ewm(span=26, adjust=False).mean().iloc[-1]
        
        macd = ema_12 - ema_26
        signal = pd.Series([macd]).ewm(span=9, adjust=False).mean().iloc[-1]
        
        return macd - signal  # Return MACD histogram
    
    def calculate_stochastic(self, prices, period=14):
        """Calculate Stochastic Oscillator"""
        if len(prices) < period:
            return np.nan
            
        lowest_low = min(prices[-period:])
        highest_high = max(prices[-period:])
        
        if highest_high == lowest_low:
            return 50  # Middle value if no range
            
        current_close = prices[-1]
        k_percent = 100 * ((current_close - lowest_low) / (highest_high - lowest_low))
        
        return k_percent
    
    def process_indicators(self):
        """Calculate all technical indicators for each cryptocurrency"""
        print("\nCalculating technical indicators...")
        
        rsi_values = []
        macd_values = []
        stoch_values = []
        
        for idx, row in self.data.iterrows():
            # Get sparkline data (7-day hourly prices)
            sparkline = row.get('sparkline_in_7d', {})
            prices = sparkline.get('price', [])
            
            if prices and len(prices) > 0:
                # Calculate indicators
                rsi = self.calculate_rsi(prices)
                macd = self.calculate_macd(prices)
                stoch = self.calculate_stochastic(prices)
            else:
                rsi = macd = stoch = np.nan
            
            rsi_values.append(rsi)
            macd_values.append(macd)
            stoch_values.append(stoch)
            
            if (idx + 1) % 20 == 0:
                print(f"  Processed {idx + 1}/100 coins")
        
        # Add to dataframe
        self.data['RSI'] = rsi_values
        self.data['MACD'] = macd_values
        self.data['Stochastic'] = stoch_values
        
        print("Technical indicators calculated successfully")
        return self.data
    
    def create_comprehensive_chart(self):
        """Create a multi-panel chart with all metrics"""
        # Prepare data
        df = self.data.head(100).copy()  # Ensure we have top 100
        df['rank'] = range(1, len(df) + 1)
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 4, figsize=(20, 24))
        fig.suptitle('Top 100 Cryptocurrencies by Market Cap - Technical Analysis\nJanuary 31, 2025', 
                     fontsize=16, fontweight='bold')
        
        # Prepare y-axis labels (coin symbols)
        y_labels = [f"{row['rank']}. {row['symbol'].upper()}" for _, row in df.iterrows()]
        y_positions = range(len(df))
        
        # Panel 1: Price (log scale for better visualization)
        ax1 = axes[0]
        prices = df['current_price'].values
        colors1 = ['green' if p > df['current_price'].median() else 'red' for p in prices]
        bars1 = ax1.barh(y_positions, prices, color=colors1, alpha=0.7)
        ax1.set_xscale('log')
        ax1.set_xlabel('Price (USD, log scale)', fontsize=12, fontweight='bold')
        ax1.set_title('Current Price', fontsize=14)
        ax1.set_yticks(y_positions[::5])  # Show every 5th label
        ax1.set_yticklabels(y_labels[::5], fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # Add price values on bars
        for i, (bar, price) in enumerate(zip(bars1, prices)):
            if i < 20:  # Only show values for top 20
                ax1.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
                        f'${price:.2f}' if price > 1 else f'${price:.4f}',
                        ha='left', va='center', fontsize=7)
        
        # Panel 2: RSI
        ax2 = axes[1]
        rsi_values = df['RSI'].fillna(50).values
        colors2 = ['darkred' if r > 70 else 'darkgreen' if r < 30 else 'gray' for r in rsi_values]
        bars2 = ax2.barh(y_positions, rsi_values, color=colors2, alpha=0.7)
        ax2.axvline(x=30, color='green', linestyle='--', alpha=0.5, label='Oversold')
        ax2.axvline(x=70, color='red', linestyle='--', alpha=0.5, label='Overbought')
        ax2.set_xlim(0, 100)
        ax2.set_xlabel('RSI', fontsize=12, fontweight='bold')
        ax2.set_title('Relative Strength Index', fontsize=14)
        ax2.set_yticks([])
        ax2.legend(loc='upper right', fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        # Panel 3: MACD
        ax3 = axes[2]
        macd_values = df['MACD'].fillna(0).values
        # Normalize MACD for better visualization
        macd_norm = np.clip(macd_values, -10, 10)
        colors3 = ['green' if m > 0 else 'red' for m in macd_norm]
        bars3 = ax3.barh(y_positions, macd_norm, color=colors3, alpha=0.7)
        ax3.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        ax3.set_xlabel('MACD Histogram', fontsize=12, fontweight='bold')
        ax3.set_title('MACD (Normalized)', fontsize=14)
        ax3.set_yticks([])
        ax3.grid(True, alpha=0.3)
        
        # Panel 4: Stochastic Oscillator
        ax4 = axes[3]
        stoch_values = df['Stochastic'].fillna(50).values
        colors4 = ['darkred' if s > 80 else 'darkgreen' if s < 20 else 'gray' for s in stoch_values]
        bars4 = ax4.barh(y_positions, stoch_values, color=colors4, alpha=0.7)
        ax4.axvline(x=20, color='green', linestyle='--', alpha=0.5, label='Oversold')
        ax4.axvline(x=80, color='red', linestyle='--', alpha=0.5, label='Overbought')
        ax4.set_xlim(0, 100)
        ax4.set_xlabel('Stochastic %K', fontsize=12, fontweight='bold')
        ax4.set_title('Stochastic Oscillator', fontsize=14)
        ax4.set_yticks([])
        ax4.legend(loc='upper right', fontsize=8)
        ax4.grid(True, alpha=0.3)
        
        # Add coin names on the right
        for i, label in enumerate(y_labels):
            ax4.text(102, i, label, fontsize=7, va='center')
        
        plt.tight_layout()
        
        # Save the chart
        filename = 'top100_crypto_analysis.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"\nChart saved as '{filename}'")
        
        # Also create a summary DataFrame
        summary = df[['rank', 'symbol', 'name', 'current_price', 'market_cap', 
                     'price_change_percentage_24h', 'RSI', 'MACD', 'Stochastic']].copy()
        summary.columns = ['Rank', 'Symbol', 'Name', 'Price', 'Market Cap', 
                          '24h Change %', 'RSI', 'MACD', 'Stochastic']
        
        # Format the summary
        summary['Price'] = summary['Price'].apply(lambda x: f'${x:,.2f}' if x > 1 else f'${x:.6f}')
        summary['Market Cap'] = summary['Market Cap'].apply(lambda x: f'${x/1e9:.2f}B' if x > 1e9 else f'${x/1e6:.2f}M')
        summary['24h Change %'] = summary['24h Change %'].apply(lambda x: f'{x:.2f}%' if pd.notna(x) else 'N/A')
        summary['RSI'] = summary['RSI'].apply(lambda x: f'{x:.1f}' if pd.notna(x) else 'N/A')
        summary['MACD'] = summary['MACD'].apply(lambda x: f'{x:.2f}' if pd.notna(x) else 'N/A')
        summary['Stochastic'] = summary['Stochastic'].apply(lambda x: f'{x:.1f}' if pd.notna(x) else 'N/A')
        
        # Save summary to CSV
        summary.to_csv('top100_crypto_summary.csv', index=False)
        print(f"Summary saved as 'top100_crypto_summary.csv'")
        
        # Display top 10
        print("\nTop 10 Cryptocurrencies Summary:")
        print(summary.head(10).to_string(index=False))
        
        plt.show()
        
        return summary

# Main execution
if __name__ == "__main__":
    print("=" * 60)
    print("TOP 100 CRYPTOCURRENCIES TECHNICAL ANALYSIS")
    print("Date: January 31, 2025")
    print("=" * 60)
    
    analyzer = Top100CryptoAnalyzer()
    
    # Fetch data
    analyzer.fetch_top_100()
    
    # Calculate indicators
    analyzer.process_indicators()
    
    # Create visualization
    summary = analyzer.create_comprehensive_chart()
    
    # Analysis insights
    print("\n" + "=" * 60)
    print("KEY INSIGHTS:")
    print("=" * 60)
    
    # Find oversold coins (RSI < 30)
    oversold = analyzer.data[analyzer.data['RSI'] < 30]['symbol'].tolist()
    if oversold:
        print(f"Oversold (RSI < 30): {', '.join(oversold[:5])}")
    
    # Find overbought coins (RSI > 70)
    overbought = analyzer.data[analyzer.data['RSI'] > 70]['symbol'].tolist()
    if overbought:
        print(f"Overbought (RSI > 70): {', '.join(overbought[:5])}")
    
    # Bullish MACD
    bullish = analyzer.data[analyzer.data['MACD'] > 0]['symbol'].tolist()
    print(f"Bullish MACD signals: {len(bullish)} coins")
    
    # Stochastic extremes
    stoch_oversold = analyzer.data[analyzer.data['Stochastic'] < 20]['symbol'].tolist()
    if stoch_oversold:
        print(f"Stochastic oversold: {', '.join(stoch_oversold[:5])}")
    
    print("\nAnalysis complete!")