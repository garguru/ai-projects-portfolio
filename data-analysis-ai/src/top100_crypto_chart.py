"""
Top 100 Cryptocurrencies Chart with Technical Indicators
Optimized version for January 31, 2025
"""

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime
import time

def fetch_top_cryptos(limit=100):
    """Fetch top cryptocurrencies in one efficient call"""
    print(f"Fetching top {limit} cryptocurrencies...")
    
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        'vs_currency': 'usd',
        'order': 'market_cap_desc',
        'per_page': limit,
        'page': 1,
        'sparkline': True,  # 7-day price data
        'price_change_percentage': '24h,7d'
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        print(f"Successfully fetched {len(data)} cryptocurrencies")
        return pd.DataFrame(data)
    except Exception as e:
        print(f"Error: {e}")
        return None

def calculate_indicators(df):
    """Calculate technical indicators from sparkline data"""
    print("Calculating technical indicators...")
    
    rsi_list = []
    macd_list = []
    stoch_list = []
    
    for idx, row in df.iterrows():
        sparkline = row.get('sparkline_in_7d', {})
        prices = sparkline.get('price', [])
        
        if prices and len(prices) > 26:
            # RSI calculation (simplified)
            price_series = pd.Series(prices)
            delta = price_series.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs.iloc[-1]))
            
            # MACD calculation
            ema12 = price_series.ewm(span=12, adjust=False).mean().iloc[-1]
            ema26 = price_series.ewm(span=26, adjust=False).mean().iloc[-1]
            macd = (ema12 - ema26) / prices[-1] * 100  # Normalize as percentage
            
            # Stochastic calculation
            low_14 = min(prices[-14:]) if len(prices) >= 14 else min(prices)
            high_14 = max(prices[-14:]) if len(prices) >= 14 else max(prices)
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
    
    print("Indicators calculated")
    return df

def create_chart(df):
    """Create the comprehensive chart"""
    print("Creating visualization...")
    
    # Prepare data
    df = df.head(100)  # Ensure top 100
    df['rank'] = range(1, len(df) + 1)
    
    # Create figure
    fig = plt.figure(figsize=(18, 20))
    gs = gridspec.GridSpec(1, 4, width_ratios=[2, 1, 1, 1], wspace=0.3)
    
    # Title
    fig.suptitle('Top 100 Cryptocurrencies - Technical Analysis Dashboard\nJanuary 31, 2025', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Prepare labels
    labels = [f"{i}. {row['symbol'].upper()}" for i, (_, row) in enumerate(df.iterrows(), 1)]
    y_pos = np.arange(len(labels))
    
    # Color schemes
    def get_price_color(val, median):
        return '#2ecc71' if val > median else '#e74c3c'
    
    def get_rsi_color(val):
        if val > 70: return '#e74c3c'  # Overbought
        elif val < 30: return '#2ecc71'  # Oversold
        else: return '#95a5a6'  # Neutral
    
    def get_macd_color(val):
        return '#2ecc71' if val > 0 else '#e74c3c'
    
    def get_stoch_color(val):
        if val > 80: return '#e74c3c'  # Overbought
        elif val < 20: return '#2ecc71'  # Oversold
        else: return '#95a5a6'  # Neutral
    
    # Panel 1: Price (log scale)
    ax1 = plt.subplot(gs[0])
    prices = df['current_price'].values
    price_median = np.median(prices)
    colors1 = [get_price_color(p, price_median) for p in prices]
    
    bars1 = ax1.barh(y_pos, prices, color=colors1, alpha=0.6, edgecolor='black', linewidth=0.5)
    ax1.set_xscale('log')
    ax1.set_xlabel('Price (USD) - Log Scale', fontsize=11, fontweight='bold')
    ax1.set_title('Current Price', fontsize=12, fontweight='bold')
    ax1.set_yticks(y_pos[::5])
    ax1.set_yticklabels(labels[::5], fontsize=7)
    ax1.grid(True, alpha=0.2, axis='x')
    
    # Add top 10 price labels
    for i in range(min(10, len(bars1))):
        price = prices[i]
        ax1.text(price * 1.1, i, f'${price:,.2f}' if price > 1 else f'${price:.4f}',
                va='center', fontsize=7, fontweight='bold')
    
    # Panel 2: RSI
    ax2 = plt.subplot(gs[1])
    rsi_values = df['RSI'].values
    colors2 = [get_rsi_color(r) for r in rsi_values]
    
    ax2.barh(y_pos, rsi_values, color=colors2, alpha=0.6, edgecolor='black', linewidth=0.5)
    ax2.axvline(x=30, color='green', linestyle='--', alpha=0.4, linewidth=1)
    ax2.axvline(x=70, color='red', linestyle='--', alpha=0.4, linewidth=1)
    ax2.set_xlim(0, 100)
    ax2.set_xlabel('RSI', fontsize=11, fontweight='bold')
    ax2.set_title('RSI\n(<30 Oversold, >70 Overbought)', fontsize=10)
    ax2.set_yticks([])
    ax2.grid(True, alpha=0.2, axis='x')
    
    # Panel 3: MACD
    ax3 = plt.subplot(gs[2])
    macd_values = np.clip(df['MACD_norm'].values, -5, 5)  # Clip for visualization
    colors3 = [get_macd_color(m) for m in macd_values]
    
    ax3.barh(y_pos, macd_values, color=colors3, alpha=0.6, edgecolor='black', linewidth=0.5)
    ax3.axvline(x=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
    ax3.set_xlabel('MACD %', fontsize=11, fontweight='bold')
    ax3.set_title('MACD\n(Normalized)', fontsize=10)
    ax3.set_xlim(-5, 5)
    ax3.set_yticks([])
    ax3.grid(True, alpha=0.2, axis='x')
    
    # Panel 4: Stochastic
    ax4 = plt.subplot(gs[3])
    stoch_values = df['Stochastic'].values
    colors4 = [get_stoch_color(s) for s in stoch_values]
    
    ax4.barh(y_pos, stoch_values, color=colors4, alpha=0.6, edgecolor='black', linewidth=0.5)
    ax4.axvline(x=20, color='green', linestyle='--', alpha=0.4, linewidth=1)
    ax4.axvline(x=80, color='red', linestyle='--', alpha=0.4, linewidth=1)
    ax4.set_xlim(0, 100)
    ax4.set_xlabel('Stochastic %K', fontsize=11, fontweight='bold')
    ax4.set_title('Stochastic\n(<20 Oversold, >80 Overbought)', fontsize=10)
    ax4.set_yticks(y_pos[::5])
    ax4.set_yticklabels(labels[::5], fontsize=7)
    ax4.yaxis.tick_right()
    ax4.grid(True, alpha=0.2, axis='x')
    
    # Add legend
    legend_elements = [
        plt.Rectangle((0,0),1,1, fc='#2ecc71', alpha=0.6, label='Bullish/Above Average'),
        plt.Rectangle((0,0),1,1, fc='#e74c3c', alpha=0.6, label='Bearish/Below Average'),
        plt.Rectangle((0,0),1,1, fc='#95a5a6', alpha=0.6, label='Neutral')
    ]
    ax1.legend(handles=legend_elements, loc='lower right', fontsize=8)
    
    plt.tight_layout()
    
    # Save chart
    filename = 'top100_crypto_chart.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Chart saved as '{filename}'")
    
    # Create summary CSV
    summary = df[['rank', 'symbol', 'name', 'current_price', 'market_cap', 
                  'price_change_percentage_24h', 'RSI', 'MACD_norm', 'Stochastic']].copy()
    summary.columns = ['Rank', 'Symbol', 'Name', 'Price', 'Market Cap', 
                      '24h %', 'RSI', 'MACD %', 'Stochastic']
    summary.to_csv('top100_crypto_data.csv', index=False)
    print(f"Data saved as 'top100_crypto_data.csv'")
    
    return summary

def main():
    print("=" * 60)
    print("TOP 100 CRYPTOCURRENCIES - TECHNICAL INDICATORS")
    print("Date: January 31, 2025")
    print("=" * 60)
    
    # Fetch data
    df = fetch_top_cryptos(100)
    
    if df is not None:
        # Calculate indicators
        df = calculate_indicators(df)
        
        # Create chart
        summary = create_chart(df)
        
        # Print insights
        print("\n" + "=" * 60)
        print("KEY INSIGHTS:")
        print("=" * 60)
        
        # Top 10 by market cap
        print("\nTop 10 by Market Cap:")
        for i, row in df.head(10).iterrows():
            print(f"{i+1}. {row['symbol'].upper()}: ${row['current_price']:,.2f} "
                  f"(RSI: {row['RSI']:.1f}, MACD: {row['MACD_norm']:.2f}%)")
        
        # Find extremes
        oversold_rsi = df[df['RSI'] < 30]
        if not oversold_rsi.empty:
            print(f"\nOversold (RSI < 30): {', '.join(oversold_rsi['symbol'].str.upper().tolist()[:5])}")
        
        overbought_rsi = df[df['RSI'] > 70]
        if not overbought_rsi.empty:
            print(f"Overbought (RSI > 70): {', '.join(overbought_rsi['symbol'].str.upper().tolist()[:5])}")
        
        bullish_macd = df[df['MACD_norm'] > 1]
        print(f"\nStrong Bullish MACD (>1%): {len(bullish_macd)} coins")
        
        bearish_macd = df[df['MACD_norm'] < -1]
        print(f"Strong Bearish MACD (<-1%): {len(bearish_macd)} coins")
        
        print("\nChart generation complete!")
        print("Files created:")
        print("  - top100_crypto_chart.png (visualization)")
        print("  - top100_crypto_data.csv (data export)")
    else:
        print("Failed to fetch data. Please try again.")

if __name__ == "__main__":
    main()