import os
from dotenv import load_dotenv
import requests
import pandas as pd
load_dotenv()

def sentiment_data(ticker, date_from, date_to):
    rapid_key = os.getenv('rapid_key')
    url = f'https://us-stocks-news-sentiment-data.p.rapidapi.com/{ticker}'
    queries = {'dateTo': date_to, 'dateFrom': date_from}
    headers = {
        'x-rapidapi-key': rapid_key,
        'x-rapidapi-host': 'us-stocks-news-sentiment-data.p.rapidapi.com'
    }
    response = requests.get(url, headers=headers, params=queries)
    df = pd.DataFrame(response.json()['sentiments'])
    df.index = pd.to_datetime(df['date'])
    df.drop(columns='date', inplace=True)
    df = df.iloc[::-1]
    return df

def price_data(ticker, date_from, date_to):
    av_key = os.getenv('av_key')
    base_url = 'https://learn-api.wqu.edu/1/data-services/alpha-vantage/query?'
    sub_url = f'function=TIME_SERIES_DAILY&symbol=AAPL&apikey={av_key}&outputsize=full'
    response = requests.get(base_url + sub_url)
    df = pd.DataFrame(response.json()['Time Series (Daily)']).T
    df.index = pd.to_datetime(df.index)
    df = df.loc[pd.to_datetime(date_to):pd.to_datetime(date_from)]
    df.columns = ['open', 'high', 'low', 'close', 'volume']
    df = df.apply(lambda x: pd.to_numeric(x))
    df = df.iloc[::-1]
    return df

def technical_indicators(price_data, window_short=20, window_long=50):
    df = pd.DataFrame(index=price_data.index)

    df['returns'] = price_data['close'].pct_change()
    
    df['SMA_short'] = price_data['close'].rolling(window=window_short).mean()
    df['SMA_long'] = price_data['close'].rolling(window=window_long).mean()
    df['EMA_short'] = price_data['close'].ewm(span=window_short, adjust=False).mean()
    df['EMA_long'] = price_data['close'].ewm(span=window_long, adjust=False).mean()

    df['std_dev'] = price_data['close'].rolling(window=window_short).std()
    df['upper_band'] = df['SMA_short'] + (df['std_dev'] * 2)
    df['lower_band'] = df['SMA_short'] - (df['std_dev'] * 2)

    delta = price_data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window_short).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window_short).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    df['MACD'] = df['EMA_short'] - df['EMA_long']
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_histogram'] = df['MACD'] - df['MACD_signal']

    tr1 = price_data['high'] - price_data['low']
    tr2 = abs(price_data['high'] - price_data['close'].shift())
    tr3 = abs(price_data['low'] - price_data['close'].shift())
    df['ATR'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(window=window_short).mean()

    return df

def wrangle_data(*dfs):
    df = pd.concat(dfs, axis=1)
    df.dropna(inplace=True)
    return df