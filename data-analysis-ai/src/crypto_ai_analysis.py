"""
Crypto Market Analysis with AI Enhancement
Perfect for your crypto trading interests, Garrita!
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.subplots as sp
from plotly.offline import plot
import logging

logger = logging.getLogger(__name__)

class CryptoAIAnalyzer:
    """AI-powered crypto market analyzer"""
    
    def __init__(self):
        self.data = {}
        self.signals = []
        
    def fetch_crypto_data(self, symbols=['BTC-USD', 'ETH-USD'], days=30):
        """Fetch cryptocurrency data"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        for symbol in symbols:
            print(f"📥 Fetching {symbol} data...")
            self.data[symbol] = yf.download(symbol, start=start_date, end=end_date, progress=False)
            
        return self.data
    
    def calculate_indicators(self, symbol='BTC-USD'):
        """Calculate technical indicators with AI insights"""
        df = self.data[symbol].copy()
        
        # Moving averages (like timing in cooking - everything has its moment)
        df['MA7'] = df['Close'].rolling(window=7).mean()
        df['MA21'] = df['Close'].rolling(window=21).mean()
        
        # RSI - Relative Strength Index
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Volatility
        df['Volatility'] = df['Close'].pct_change().rolling(window=7).std() * np.sqrt(365)

        # Bollinger Bands
        bb_period = 20
        bb_std = 2
        df['BB_Middle'] = df['Close'].rolling(window=bb_period).mean()
        bb_std_dev = df['Close'].rolling(window=bb_period).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std_dev * bb_std)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std_dev * bb_std)

        # MACD (Moving Average Convergence Divergence)
        exp1 = df['Close'].ewm(span=12).mean()
        exp2 = df['Close'].ewm(span=26).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']

        # AI-like pattern detection with enhanced signals
        df['Signal'] = 'HOLD'
        
        # Enhanced Buy signals using multiple indicators
        buy_condition = (
            (df['MA7'] > df['MA21']) &  # Uptrend
            (df['RSI'] < 40) &  # Not overbought (relaxed from 30)
            (df['Close'] < df['BB_Lower']) &  # Below lower Bollinger Band (oversold)
            (df['MACD'] > df['MACD_Signal'])  # MACD bullish crossover
        )
        df.loc[buy_condition, 'Signal'] = 'BUY'

        # Strong buy for multiple confirmations
        strong_buy_condition = (
            (df['MA7'] > df['MA21']) &
            (df['RSI'] < 30) &
            (df['Close'] < df['BB_Lower']) &
            (df['MACD'] > df['MACD_Signal']) &
            (df['MACD_Histogram'] > 0)
        )
        df.loc[strong_buy_condition, 'Signal'] = 'STRONG_BUY'

        # Enhanced Sell signals
        sell_condition = (
            (df['MA7'] < df['MA21']) &  # Downtrend
            (df['RSI'] > 60) &  # Overbought (relaxed from 70)
            (df['Close'] > df['BB_Upper']) &  # Above upper Bollinger Band (overbought)
            (df['MACD'] < df['MACD_Signal'])  # MACD bearish crossover
        )
        df.loc[sell_condition, 'Signal'] = 'SELL'
        
        self.data[symbol] = df
        return df
    
    def generate_ai_insights(self, symbol='BTC-USD'):
        """Generate AI-powered trading insights"""
        df = self.data[symbol]
        latest = df.iloc[-1]
        
        insights = []
        
        # Price trend insight
        price_change = (latest['Close'] - df.iloc[-7]['Close']) / df.iloc[-7]['Close'] * 100
        if abs(price_change) > 5:
            direction = "up 📈" if price_change > 0 else "down 📉"
            insights.append(f"{symbol} is {direction} {abs(price_change):.1f}% this week")
        
        # RSI insight
        if latest['RSI'] < 30:
            insights.append(f"🟢 {symbol} is oversold (RSI: {latest['RSI']:.1f}) - potential buy opportunity")
        elif latest['RSI'] > 70:
            insights.append(f"🔴 {symbol} is overbought (RSI: {latest['RSI']:.1f}) - consider taking profits")
        
        # Volatility insight
        if latest['Volatility'] > 0.8:
            insights.append(f"⚠️ High volatility detected ({latest['Volatility']:.1%} annualized)")
        
        # Moving average crossover
        if latest['MA7'] > latest['MA21'] and df.iloc[-2]['MA7'] <= df.iloc[-2]['MA21']:
            insights.append("🎯 Golden cross detected - bullish signal!")
        elif latest['MA7'] < latest['MA21'] and df.iloc[-2]['MA7'] >= df.iloc[-2]['MA21']:
            insights.append("⚡ Death cross detected - bearish signal!")
        
        return insights
    
    def visualize_analysis(self, symbol='BTC-USD'):
        """Create comprehensive visualization"""
        df = self.data[symbol]
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Price and moving averages
        axes[0].plot(df.index, df['Close'], label='Close Price', linewidth=2)
        axes[0].plot(df.index, df['MA7'], label='MA7', alpha=0.7)
        axes[0].plot(df.index, df['MA21'], label='MA21', alpha=0.7)
        axes[0].set_title(f'{symbol} Price Analysis')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # RSI
        axes[1].plot(df.index, df['RSI'], color='purple', linewidth=2)
        axes[1].axhline(y=70, color='r', linestyle='--', alpha=0.5)
        axes[1].axhline(y=30, color='g', linestyle='--', alpha=0.5)
        axes[1].fill_between(df.index, 30, 70, alpha=0.1)
        axes[1].set_title('RSI Indicator')
        axes[1].set_ylabel('RSI')
        axes[1].grid(True, alpha=0.3)
        
        # Volume
        axes[2].bar(df.index, df['Volume'], alpha=0.5)
        axes[2].set_title('Trading Volume')
        axes[2].set_ylabel('Volume')
        
        plt.tight_layout()
        plt.savefig(f'{symbol}_analysis.png')
        plt.show()

        print(f"📊 Analysis chart saved as '{symbol}_analysis.png'")

    def create_interactive_dashboard(self, symbol='BTC-USD'):
        """Create interactive Plotly dashboard"""
        if symbol not in self.data:
            logger.error(f"No data available for {symbol}")
            return

        df = self.data[symbol]

        # Create subplots
        fig = sp.make_subplots(
            rows=3, cols=1,
            subplot_titles=['Price & Moving Averages', 'RSI Indicator', 'Volume'],
            vertical_spacing=0.05,
            shared_xaxes=True,
            row_heights=[0.5, 0.25, 0.25]
        )

        # Price and moving averages
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['Close'],
                name='Close Price',
                line=dict(color='#1f77b4', width=2),
                hovertemplate='Price: $%{y:.2f}<br>Date: %{x}<extra></extra>'
            ), row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['MA7'],
                name='MA7',
                line=dict(color='orange', width=1),
                opacity=0.7
            ), row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['MA21'],
                name='MA21',
                line=dict(color='red', width=1),
                opacity=0.7
            ), row=1, col=1
        )

        # RSI
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['RSI'],
                name='RSI',
                line=dict(color='purple', width=2),
                hovertemplate='RSI: %{y:.1f}<br>Date: %{x}<extra></extra>'
            ), row=2, col=1
        )

        # RSI reference lines
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)

        # Volume
        fig.add_trace(
            go.Bar(
                x=df.index, y=df['Volume'],
                name='Volume',
                marker_color='lightblue',
                opacity=0.6
            ), row=3, col=1
        )

        # Update layout
        fig.update_layout(
            title=f'{symbol} Interactive Analysis Dashboard',
            height=800,
            showlegend=True,
            template='plotly_white'
        )

        # Update x-axes
        fig.update_xaxes(title_text="Date", row=3, col=1)

        # Update y-axes
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
        fig.update_yaxes(title_text="Volume", row=3, col=1)

        # Save as HTML
        filename = f'{symbol}_interactive_dashboard.html'
        plot(fig, filename=filename, auto_open=False)
        logger.info(f"Interactive dashboard saved as {filename}")

        return fig
    
    def ai_trading_recommendation(self, symbol='BTC-USD', investment=1000):
        """Generate AI trading recommendation"""
        latest = self.data[symbol].iloc[-1]
        
        print(f"\n🤖 AI Trading Recommendation for {symbol}")
        print("=" * 50)
        
        # Calculate position size (like portioning in cooking)
        risk_level = "LOW" if latest['Volatility'] < 0.5 else "MEDIUM" if latest['Volatility'] < 0.8 else "HIGH"
        
        if latest['Signal'] == 'BUY':
            position_size = investment * (0.3 if risk_level == "HIGH" else 0.5 if risk_level == "MEDIUM" else 0.7)
            print(f"✅ RECOMMENDATION: BUY")
            print(f"💰 Suggested position: ${position_size:.2f} ({position_size/investment*100:.0f}% of capital)")
            print(f"⚠️ Risk level: {risk_level}")
            print(f"🎯 Entry price: ${latest['Close']:.2f}")
            print(f"🛡️ Stop loss: ${latest['Close'] * 0.95:.2f} (-5%)")
            print(f"🎁 Take profit: ${latest['Close'] * 1.10:.2f} (+10%)")
            
        elif latest['Signal'] == 'SELL':
            print(f"❌ RECOMMENDATION: SELL/AVOID")
            print(f"📉 Current price: ${latest['Close']:.2f}")
            print(f"⚠️ Risk level: {risk_level}")
            print("💡 Wait for better entry point")
            
        else:
            print(f"⏸️ RECOMMENDATION: HOLD")
            print(f"📊 Current price: ${latest['Close']:.2f}")
            print(f"⚠️ Risk level: {risk_level}")
            print("👀 Monitor for clearer signals")

# Example usage
if __name__ == "__main__":
    print("🚀 Crypto AI Analyzer Starting...")
    print("=" * 50)
    
    analyzer = CryptoAIAnalyzer()
    
    # Fetch data
    analyzer.fetch_crypto_data(['BTC-USD', 'ETH-USD'], days=30)
    
    # Analyze Bitcoin
    analyzer.calculate_indicators('BTC-USD')
    
    # Get AI insights
    insights = analyzer.generate_ai_insights('BTC-USD')
    print("\n💡 AI Insights:")
    for insight in insights:
        print(f"  {insight}")
    
    # Create interactive dashboard
    analyzer.create_interactive_dashboard('BTC-USD')

    # Also create static visualization for comparison
    analyzer.visualize_analysis('BTC-USD')

    # Get trading recommendation
    analyzer.ai_trading_recommendation('BTC-USD', investment=1000)