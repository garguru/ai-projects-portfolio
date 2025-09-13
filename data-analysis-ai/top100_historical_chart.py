"""
Top 100 Cryptocurrencies Historical Data - January 31, 2025 at 12:00 PM
Historical snapshot with technical indicators
"""

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime, timezone
import time
import json

class HistoricalCryptoAnalyzer:
    """Fetch historical data for specific date and time"""
    
    def __init__(self):
        self.base_url = "https://api.coingecko.com/api/v3"
        # January 31, 2025 at 12:00 PM UTC
        self.target_date = "31-01-2025"
        self.target_timestamp = int(datetime(2025, 1, 31, 12, 0, 0, tzinfo=timezone.utc).timestamp())
        
    def fetch_historical_top_100(self):
        """Fetch historical data for top 100 coins on specific date"""
        print(f"Fetching historical data for January 31, 2025 at 12:00 PM...")
        print("Note: Fetching historical prices for each coin...")
        
        # First get the list of top 100 coins by current market cap
        # (CoinGecko doesn't provide historical rankings easily)
        coins_url = f"{self.base_url}/coins/markets"
        params = {
            'vs_currency': 'usd',
            'order': 'market_cap_desc',
            'per_page': 100,
            'page': 1
        }
        
        try:
            response = requests.get(coins_url, params=params)
            response.raise_for_status()
            coins = response.json()
            
            # Now fetch historical data for each coin
            historical_data = []
            
            for i, coin in enumerate(coins):
                if i % 10 == 0:
                    print(f"  Fetching historical data: {i}/100 coins...")
                    time.sleep(1)  # Rate limiting
                
                # Get historical data for this coin
                hist_url = f"{self.base_url}/coins/{coin['id']}/history"
                hist_params = {'date': self.target_date}
                
                try:
                    hist_response = requests.get(hist_url, params=hist_params)
                    if hist_response.status_code == 200:
                        hist_data = hist_response.json()
                        
                        # Extract relevant data
                        market_data = hist_data.get('market_data', {})
                        
                        coin_info = {
                            'id': coin['id'],
                            'symbol': coin['symbol'],
                            'name': coin['name'],
                            'current_price': market_data.get('current_price', {}).get('usd', 0),
                            'market_cap': market_data.get('market_cap', {}).get('usd', 0),
                            'total_volume': market_data.get('total_volume', {}).get('usd', 0),
                            'price_change_24h': market_data.get('price_change_percentage_24h_in_currency', {}).get('usd', 0)
                        }
                        
                        # Get price history for technical indicators
                        # Fetch 30-day market chart ending on target date
                        chart_url = f"{self.base_url}/coins/{coin['id']}/market_chart/range"
                        chart_params = {
                            'vs_currency': 'usd',
                            'from': self.target_timestamp - (30 * 24 * 3600),  # 30 days before
                            'to': self.target_timestamp
                        }
                        
                        chart_response = requests.get(chart_url, params=chart_params)
                        if chart_response.status_code == 200:
                            chart_data = chart_response.json()
                            coin_info['price_history'] = [p[1] for p in chart_data.get('prices', [])]
                        else:
                            coin_info['price_history'] = []
                        
                        historical_data.append(coin_info)
                        
                    time.sleep(0.5)  # Rate limiting
                    
                except Exception as e:
                    print(f"    Error fetching {coin['id']}: {e}")
                    continue
            
            df = pd.DataFrame(historical_data)
            print(f"Successfully fetched historical data for {len(df)} coins")
            print(f"Date: January 31, 2025 at 12:00 PM UTC")
            return df
            
        except Exception as e:
            print(f"Error: {e}")
            return None
    
    def calculate_indicators(self, df):
        """Calculate technical indicators from historical price data"""
        print("Calculating technical indicators...")
        
        rsi_list = []
        macd_list = []
        stoch_list = []
        
        for idx, row in df.iterrows():
            prices = row.get('price_history', [])
            
            if prices and len(prices) > 26:
                # RSI calculation
                price_series = pd.Series(prices)
                delta = price_series.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs.iloc[-1]))
                
                # MACD calculation
                ema12 = price_series.ewm(span=12, adjust=False).mean().iloc[-1]
                ema26 = price_series.ewm(span=26, adjust=False).mean().iloc[-1]
                macd = (ema12 - ema26) / prices[-1] * 100 if prices[-1] > 0 else 0
                
                # Stochastic calculation
                low_14 = min(prices[-14:])
                high_14 = max(prices[-14:])
                if high_14 != low_14:
                    stoch = 100 * ((prices[-1] - low_14) / (high_14 - low_14))
                else:
                    stoch = 50
            else:
                rsi = 50
                macd = 0
                stoch = 50
            
            rsi_list.append(rsi if not pd.isna(rsi) else 50)
            macd_list.append(macd if not pd.isna(macd) else 0)
            stoch_list.append(stoch if not pd.isna(stoch) else 50)
        
        df['RSI'] = rsi_list
        df['MACD_norm'] = macd_list
        df['Stochastic'] = stoch_list
        
        print("Technical indicators calculated")
        return df
    
    def create_historical_chart(self, df):
        """Create the comprehensive historical chart"""
        print("Creating historical data visualization...")
        
        # Sort by market cap and take top 100
        df = df.sort_values('market_cap', ascending=False).head(100)
        df['rank'] = range(1, len(df) + 1)
        
        # Create figure
        fig = plt.figure(figsize=(20, 24))
        gs = gridspec.GridSpec(1, 4, width_ratios=[2.5, 1, 1, 1], wspace=0.25)
        
        # Title with specific date and time
        fig.suptitle('Top 100 Cryptocurrencies - Historical Snapshot\nJanuary 31, 2025 at 12:00 PM UTC', 
                     fontsize=18, fontweight='bold', y=0.99)
        
        # Prepare labels
        labels = [f"{i}. {row['symbol'].upper()}" for i, (_, row) in enumerate(df.iterrows(), 1)]
        y_pos = np.arange(len(labels))
        
        # Color functions
        def get_price_color(val, median):
            return '#27ae60' if val > median else '#e74c3c'
        
        def get_rsi_color(val):
            if val > 70: return '#c0392b'  # Overbought - dark red
            elif val < 30: return '#27ae60'  # Oversold - green
            else: return '#7f8c8d'  # Neutral - gray
        
        def get_macd_color(val):
            if val > 1: return '#27ae60'  # Strong bullish
            elif val < -1: return '#c0392b'  # Strong bearish
            else: return '#95a5a6'  # Neutral
        
        def get_stoch_color(val):
            if val > 80: return '#c0392b'  # Overbought
            elif val < 20: return '#27ae60'  # Oversold
            else: return '#7f8c8d'  # Neutral
        
        # Panel 1: Historical Price (log scale)
        ax1 = plt.subplot(gs[0])
        prices = df['current_price'].values
        price_median = np.median(prices[prices > 0])  # Exclude zeros
        colors1 = [get_price_color(p, price_median) for p in prices]
        
        bars1 = ax1.barh(y_pos, prices, color=colors1, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax1.set_xscale('log')
        ax1.set_xlabel('Price (USD) on Jan 31, 2025 12:00 PM - Log Scale', fontsize=12, fontweight='bold')
        ax1.set_title('Historical Price', fontsize=14, fontweight='bold')
        ax1.set_yticks(y_pos[::2])  # Show every 2nd label
        ax1.set_yticklabels(labels[::2], fontsize=6)
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Add top 10 price labels
        for i in range(min(10, len(bars1))):
            price = prices[i]
            if price > 0:
                ax1.text(price * 1.05, i, f'${price:,.2f}' if price > 1 else f'${price:.6f}',
                        va='center', fontsize=7, fontweight='bold')
        
        # Panel 2: RSI
        ax2 = plt.subplot(gs[1])
        rsi_values = df['RSI'].values
        colors2 = [get_rsi_color(r) for r in rsi_values]
        
        ax2.barh(y_pos, rsi_values, color=colors2, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax2.axvline(x=30, color='green', linestyle='--', alpha=0.5, linewidth=1.5, label='Oversold')
        ax2.axvline(x=50, color='gray', linestyle='-', alpha=0.3, linewidth=1)
        ax2.axvline(x=70, color='red', linestyle='--', alpha=0.5, linewidth=1.5, label='Overbought')
        ax2.set_xlim(0, 100)
        ax2.set_xlabel('RSI', fontsize=12, fontweight='bold')
        ax2.set_title('RSI\n(Historical)', fontsize=11, fontweight='bold')
        ax2.set_yticks([])
        ax2.legend(loc='upper right', fontsize=8)
        ax2.grid(True, alpha=0.3, axis='x')
        
        # Panel 3: MACD
        ax3 = plt.subplot(gs[2])
        macd_values = np.clip(df['MACD_norm'].values, -5, 5)
        colors3 = [get_macd_color(m) for m in macd_values]
        
        ax3.barh(y_pos, macd_values, color=colors3, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax3.axvline(x=0, color='black', linestyle='-', alpha=0.5, linewidth=2)
        ax3.set_xlabel('MACD %', fontsize=12, fontweight='bold')
        ax3.set_title('MACD\n(Historical)', fontsize=11, fontweight='bold')
        ax3.set_xlim(-5, 5)
        ax3.set_yticks([])
        ax3.grid(True, alpha=0.3, axis='x')
        
        # Panel 4: Stochastic
        ax4 = plt.subplot(gs[3])
        stoch_values = df['Stochastic'].values
        colors4 = [get_stoch_color(s) for s in stoch_values]
        
        ax4.barh(y_pos, stoch_values, color=colors4, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax4.axvline(x=20, color='green', linestyle='--', alpha=0.5, linewidth=1.5, label='Oversold')
        ax4.axvline(x=50, color='gray', linestyle='-', alpha=0.3, linewidth=1)
        ax4.axvline(x=80, color='red', linestyle='--', alpha=0.5, linewidth=1.5, label='Overbought')
        ax4.set_xlim(0, 100)
        ax4.set_xlabel('Stochastic %K', fontsize=12, fontweight='bold')
        ax4.set_title('Stochastic\n(Historical)', fontsize=11, fontweight='bold')
        ax4.set_yticks(y_pos[::2])
        ax4.set_yticklabels(labels[::2], fontsize=6)
        ax4.yaxis.tick_right()
        ax4.legend(loc='upper left', fontsize=8)
        ax4.grid(True, alpha=0.3, axis='x')
        
        # Add timestamp
        timestamp_text = "Data: January 31, 2025 at 12:00 PM UTC (Historical Snapshot)"
        fig.text(0.5, 0.01, timestamp_text, ha='center', fontsize=10, style='italic')
        
        plt.tight_layout()
        
        # Save chart
        filename = 'top100_historical_jan31_2025.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Historical chart saved as '{filename}'")
        
        # Create CSV with historical data
        summary = df[['rank', 'symbol', 'name', 'current_price', 'market_cap', 
                      'price_change_24h', 'RSI', 'MACD_norm', 'Stochastic']].copy()
        summary.columns = ['Rank', 'Symbol', 'Name', 'Price (Jan 31)', 'Market Cap', 
                          '24h Change %', 'RSI', 'MACD %', 'Stochastic']
        
        # Format for readability
        summary['Price (Jan 31)'] = summary['Price (Jan 31)'].apply(
            lambda x: f'${x:,.2f}' if x > 1 else f'${x:.6f}' if x > 0 else 'N/A'
        )
        summary['Market Cap'] = summary['Market Cap'].apply(
            lambda x: f'${x/1e9:.2f}B' if x > 1e9 else f'${x/1e6:.2f}M' if x > 0 else 'N/A'
        )
        
        summary.to_csv('top100_historical_jan31_2025.csv', index=False)
        print(f"Historical data saved as 'top100_historical_jan31_2025.csv'")
        
        return summary

# Main execution
if __name__ == "__main__":
    print("=" * 70)
    print("TOP 100 CRYPTOCURRENCIES - HISTORICAL ANALYSIS")
    print("Target Date: January 31, 2025 at 12:00 PM UTC")
    print("=" * 70)
    
    analyzer = HistoricalCryptoAnalyzer()
    
    # Fetch historical data
    df = analyzer.fetch_historical_top_100()
    
    if df is not None and not df.empty:
        # Calculate indicators
        df = analyzer.calculate_indicators(df)
        
        # Create chart
        summary = analyzer.create_historical_chart(df)
        
        # Print top 10
        print("\n" + "=" * 70)
        print("TOP 10 CRYPTOCURRENCIES ON JANUARY 31, 2025 AT 12:00 PM UTC:")
        print("=" * 70)
        
        for i, row in df.head(10).iterrows():
            print(f"{row['rank']}. {row['symbol'].upper()}: ${row['current_price']:,.2f}")
            print(f"   RSI: {row['RSI']:.1f} | MACD: {row['MACD_norm']:.2f}% | Stoch: {row['Stochastic']:.1f}")
        
        print("\nHistorical analysis complete!")
    else:
        print("Failed to fetch historical data.")