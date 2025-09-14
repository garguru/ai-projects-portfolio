"""
CoinGecko API Integration for AI-Enhanced Crypto Analysis
Built for Garrita - Professional Crypto Data Analysis
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
import logging
from typing import Dict, List, Optional
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from pydantic import BaseModel, field_validator

# Simple data validation models
class CoinData(BaseModel):
    """Validate basic coin data structure"""
    name: str
    symbol: str
    current_price: float
    market_cap: float
    volume_24h: float

    @field_validator('current_price', 'market_cap', 'volume_24h')
    @classmethod
    def validate_positive_numbers(cls, v):
        if v is None or v < 0:
            raise ValueError('Financial data must be positive')
        return v

# Set up logging
logger = logging.getLogger(__name__)

class CoinGeckoAnalyzer:
    """
    Professional CoinGecko API integration with AI analysis capabilities
    Free tier: 10-30 calls/minute (we'll handle rate limiting)
    """
    
    def __init__(self):
        self.base_url = "https://api.coingecko.com/api/v3"
        self.session = requests.Session()
        self.data_cache = {}
        
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((requests.exceptions.RequestException, requests.exceptions.Timeout))
    )
    def _make_request(self, endpoint: str, params: dict = None) -> dict:
        """Make API request with robust error handling and retry logic"""
        url = f"{self.base_url}/{endpoint}"
        try:
            logger.info(f"Making request to: {endpoint}")
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()

            # Rate limiting (be nice to free API)
            time.sleep(1)

            data = response.json()
            logger.info(f"Successfully fetched data from {endpoint}")
            return data

        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed for {endpoint}: {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response from {endpoint}: {e}")
            return {}

    def validate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Simple data validation and cleaning"""
        if df.empty:
            return df

        # Remove rows with null prices or negative values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df = df.dropna(subset=numeric_cols)
        df = df[(df[numeric_cols] >= 0).all(axis=1)]

        logger.info(f"Data validation complete. {len(df)} valid rows remaining")
        return df

    def get_coin_list(self, limit: int = 100) -> pd.DataFrame:
        """Get list of top cryptocurrencies by market cap"""
        params = {
            'vs_currency': 'usd',
            'order': 'market_cap_desc',
            'per_page': limit,
            'page': 1,
            'sparkline': False
        }
        
        data = self._make_request('coins/markets', params)
        
        if data:
            df = pd.DataFrame(data)
            df = self.validate_dataframe(df)
            logger.info(f"Fetched {len(df)} valid cryptocurrencies")
            return df
        return pd.DataFrame()
    
    def get_coin_data(self, coin_id: str = 'bitcoin', days: int = 30) -> pd.DataFrame:
        """Get historical price data for a specific coin"""
        params = {
            'vs_currency': 'usd',
            'days': days,
            'interval': 'daily' if days > 1 else 'hourly'
        }
        
        data = self._make_request(f'coins/{coin_id}/market_chart', params)
        
        if data:
            # Convert to DataFrame
            prices = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
            volumes = pd.DataFrame(data['total_volumes'], columns=['timestamp', 'volume'])
            market_caps = pd.DataFrame(data['market_caps'], columns=['timestamp', 'market_cap'])
            
            # Merge all data
            df = prices.merge(volumes, on='timestamp').merge(market_caps, on='timestamp')
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Cache the data
            self.data_cache[coin_id] = df
            
            print(f"[OK] Fetched {len(df)} days of data for {coin_id}")
            return df
        return pd.DataFrame()
    
    def get_coin_details(self, coin_id: str = 'bitcoin') -> dict:
        """Get detailed information about a coin"""
        params = {
            'localization': False,
            'tickers': False,
            'market_data': True,
            'community_data': True,
            'developer_data': True
        }
        
        data = self._make_request(f'coins/{coin_id}', params)
        
        if data:
            details = {
                'name': data.get('name'),
                'symbol': data.get('symbol'),
                'current_price': data.get('market_data', {}).get('current_price', {}).get('usd'),
                'market_cap': data.get('market_data', {}).get('market_cap', {}).get('usd'),
                'volume_24h': data.get('market_data', {}).get('total_volume', {}).get('usd'),
                'price_change_24h': data.get('market_data', {}).get('price_change_percentage_24h'),
                'price_change_7d': data.get('market_data', {}).get('price_change_percentage_7d'),
                'price_change_30d': data.get('market_data', {}).get('price_change_percentage_30d'),
                'ath': data.get('market_data', {}).get('ath', {}).get('usd'),
                'ath_date': data.get('market_data', {}).get('ath_date', {}).get('usd'),
                'from_ath': data.get('market_data', {}).get('ath_change_percentage', {}).get('usd'),
                'circulating_supply': data.get('market_data', {}).get('circulating_supply'),
                'max_supply': data.get('market_data', {}).get('max_supply'),
            }
            return details
        return {}
    
    def get_trending_coins(self) -> List[Dict]:
        """Get trending coins (most searched in last 24h)"""
        data = self._make_request('search/trending')
        
        if data:
            trending = []
            for coin in data.get('coins', []):
                trending.append({
                    'name': coin['item']['name'],
                    'symbol': coin['item']['symbol'],
                    'market_cap_rank': coin['item']['market_cap_rank'],
                    'price_btc': coin['item']['price_btc']
                })
            print(f"[TRENDING] Found {len(trending)} trending coins")
            return trending
        return []
    
    def get_defi_data(self) -> dict:
        """Get global DeFi statistics"""
        data = self._make_request('global/decentralized_finance_defi')
        
        if data:
            return {
                'defi_market_cap': data.get('defi_market_cap'),
                'defi_to_eth_ratio': data.get('defi_to_eth_ratio'),
                'trading_volume_24h': data.get('trading_volume_24h'),
                'defi_dominance': data.get('defi_dominance'),
                'top_coin': data.get('top_coin_name')
            }
        return {}
    
    def analyze_coin_ai(self, coin_id: str = 'bitcoin') -> Dict:
        """AI-enhanced analysis of a cryptocurrency"""
        # Get coin data
        details = self.get_coin_details(coin_id)
        historical = self.get_coin_data(coin_id, days=30)
        
        if not details or historical.empty:
            return {"error": "Could not fetch data"}
        
        # Calculate technical indicators
        historical['returns'] = historical['price'].pct_change()
        historical['volatility'] = historical['returns'].rolling(7).std() * np.sqrt(365)
        historical['MA7'] = historical['price'].rolling(7).mean()
        historical['MA30'] = historical['price'].rolling(30).mean()
        
        # Generate AI insights
        insights = []
        
        # Price trend analysis
        current_price = details['current_price']
        ma7 = historical['MA7'].iloc[-1]
        ma30 = historical['MA30'].iloc[-1]
        
        if current_price > ma7 > ma30:
            insights.append("[UP] Strong uptrend - price above both moving averages")
        elif current_price < ma7 < ma30:
            insights.append("[DOWN] Downtrend - price below both moving averages")
        else:
            insights.append("[NEUTRAL] Consolidation phase - mixed signals")
        
        # Volatility insight
        current_vol = historical['volatility'].iloc[-1]
        if current_vol > 1.0:
            insights.append(f"[WARNING] High volatility: {current_vol:.1%} annualized")
        elif current_vol < 0.5:
            insights.append(f"[CALM] Low volatility: {current_vol:.1%} annualized")
        
        # Market position
        from_ath = details.get('from_ath', 0)
        if from_ath < -50:
            insights.append(f"[VALUE] Potential value: {abs(from_ath):.1f}% below ATH")
        elif from_ath > -10:
            insights.append(f"[HOT] Near all-time high: only {abs(from_ath):.1f}% below")
        
        # Volume analysis
        volume_change = (historical['volume'].iloc[-1] - historical['volume'].mean()) / historical['volume'].mean()
        if volume_change > 0.5:
            insights.append(f"[VOLUME+] High volume: {volume_change:.0%} above average")
        elif volume_change < -0.5:
            insights.append(f"[VOLUME-] Low volume: {abs(volume_change):.0%} below average")
        
        return {
            'coin': details['name'],
            'symbol': details['symbol'],
            'current_price': current_price,
            'market_cap': details['market_cap'],
            'insights': insights,
            'recommendation': self._generate_recommendation(details, historical),
            'risk_score': self._calculate_risk_score(historical)
        }
    
    def _generate_recommendation(self, details: dict, historical: pd.DataFrame) -> str:
        """Generate trading recommendation based on data"""
        score = 0
        
        # Price momentum
        if details['price_change_24h'] > 0:
            score += 1
        if details['price_change_7d'] > 0:
            score += 1
        if details['price_change_30d'] > 0:
            score += 1
        
        # Technical indicators
        if historical['price'].iloc[-1] > historical['MA7'].iloc[-1]:
            score += 2
        if historical['MA7'].iloc[-1] > historical['MA30'].iloc[-1]:
            score += 1
        
        # Volume
        if historical['volume'].iloc[-1] > historical['volume'].mean():
            score += 1
        
        if score >= 5:
            return "[STRONG BUY]"
        elif score >= 3:
            return "[BUY]"
        elif score >= 2:
            return "[HOLD]"
        else:
            return "[SELL/WAIT]"
    
    def _calculate_risk_score(self, historical: pd.DataFrame) -> int:
        """Calculate risk score (1-10, 10 being highest risk)"""
        volatility = historical['volatility'].iloc[-1]
        
        if volatility > 1.5:
            return 10
        elif volatility > 1.2:
            return 8
        elif volatility > 0.9:
            return 6
        elif volatility > 0.6:
            return 4
        else:
            return 2
    
    def compare_coins(self, coins: List[str]) -> pd.DataFrame:
        """Compare multiple cryptocurrencies"""
        comparison = []
        
        for coin in coins:
            print(f"Analyzing {coin}...")
            details = self.get_coin_details(coin)
            
            if details:
                comparison.append({
                    'Coin': details['name'],
                    'Symbol': details['symbol'].upper(),
                    'Price': f"${details['current_price']:,.2f}",
                    '24h Change': f"{details['price_change_24h']:.2f}%",
                    '7d Change': f"{details['price_change_7d']:.2f}%",
                    '30d Change': f"{details['price_change_30d']:.2f}%",
                    'Market Cap': f"${details['market_cap']/1e9:.2f}B",
                    'From ATH': f"{details['from_ath']:.1f}%"
                })
        
        return pd.DataFrame(comparison)

# Example usage
if __name__ == "__main__":
    print("CoinGecko AI Analyzer")
    print("=" * 50)
    
    analyzer = CoinGeckoAnalyzer()
    
    # Get trending coins
    print("\nTrending Coins:")
    trending = analyzer.get_trending_coins()
    for coin in trending[:5]:
        print(f"  - {coin['name']} ({coin['symbol']})")
    
    # Analyze Bitcoin
    print("\nBitcoin Analysis:")
    btc_analysis = analyzer.analyze_coin_ai('bitcoin')
    print(f"Price: ${btc_analysis['current_price']:,.2f}")
    print(f"Recommendation: {btc_analysis['recommendation']}")
    print(f"Risk Score: {btc_analysis['risk_score']}/10")
    print("\nInsights:")
    for insight in btc_analysis['insights']:
        print(f"  {insight}")
    
    # Compare top coins
    print("\nComparison of Top Coins:")
    comparison = analyzer.compare_coins(['bitcoin', 'ethereum', 'binancecoin', 'solana'])
    print(comparison.to_string(index=False))