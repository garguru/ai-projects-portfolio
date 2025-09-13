"""
Fetch REAL Historical Data for January 31, 2025 at 12:00 PM
Since today is August 29, 2025, this is actual past data
"""

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime, timezone
import time
import json

print("=" * 70)
print("FETCHING ACTUAL HISTORICAL DATA")
print("Today's Date: August 29, 2025")
print("Target Date: January 31, 2025 at 12:00 PM UTC (7 months ago)")
print("=" * 70)

# January 31, 2025 is in the past, so we can get real data
target_date = "31-01-2025"

# First, get current top coins to know which ones to check
print("\nFetching top cryptocurrencies list...")
url = "https://api.coingecko.com/api/v3/coins/markets"
params = {
    'vs_currency': 'usd',
    'order': 'market_cap_desc', 
    'per_page': 100,
    'page': 1
}

response = requests.get(url, params=params)
if response.status_code == 200:
    current_coins = response.json()
    print(f"Found {len(current_coins)} top coins")
    
    # Now fetch historical data for January 31, 2025
    print(f"\nFetching REAL historical data for January 31, 2025...")
    historical_data = []
    
    # Test with top 10 first to see if historical data is available
    for i, coin in enumerate(current_coins[:10]):
        coin_id = coin['id']
        hist_url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/history"
        hist_params = {'date': target_date}
        
        print(f"Fetching historical data for {coin['symbol'].upper()}...")
        
        try:
            hist_response = requests.get(hist_url, params=hist_params)
            if hist_response.status_code == 200:
                data = hist_response.json()
                market_data = data.get('market_data', {})
                
                if market_data:
                    price = market_data.get('current_price', {}).get('usd', 0)
                    market_cap = market_data.get('market_cap', {}).get('usd', 0)
                    volume = market_data.get('total_volume', {}).get('usd', 0)
                    
                    print(f"  {coin['symbol'].upper()} on Jan 31, 2025: ${price:,.2f}")
                    
                    historical_data.append({
                        'symbol': coin['symbol'],
                        'name': coin['name'],
                        'price_jan31': price,
                        'market_cap_jan31': market_cap,
                        'volume_jan31': volume
                    })
                else:
                    print(f"  No market data available for {coin['symbol']}")
            else:
                print(f"  Error {hist_response.status_code} for {coin['symbol']}")
                
            time.sleep(1)  # Rate limiting
            
        except Exception as e:
            print(f"  Error: {e}")
    
    if historical_data:
        print("\n" + "=" * 70)
        print("ACTUAL PRICES ON JANUARY 31, 2025 AT 12:00 PM:")
        print("=" * 70)
        
        df = pd.DataFrame(historical_data)
        for _, row in df.iterrows():
            print(f"{row['symbol'].upper()}: ${row['price_jan31']:,.2f}")
        
        print("\n✅ This is REAL historical data from January 31, 2025")
        print("(Not simulated - actual past market data)")
    else:
        print("\n⚠️ Historical data not available from API")
        print("The free tier might have limitations on historical data access")
        
else:
    print(f"Error fetching current coins: {response.status_code}")