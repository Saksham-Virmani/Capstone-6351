"""utils.py - Module for fetching financial news sentiment and stock community data."""

import os
import json
import requests
import pandas as pd

from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

av_key = os.getenv("av_key")

def get_news_sentiment_alpha_vantage(ticker):
    """Fetches news sentiment data from Alpha Vantage for a given stock ticker.

    Args:
        ticker (str): The stock ticker symbol.

    Returns:
        pd.DataFrame: A DataFrame containing news summaries, sources, and relevance scores,
            indexed by the time the news was published.
    """
    base_url = "https://learn-api.wqu.edu/1/data-services/alpha-vantage/query?"
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": ticker,
        "apikey": av_key,
    }

    response = requests.get(base_url, params=params)

    if response.status_code == 200:
        data = response.json()["feed"]

        df = pd.DataFrame(
            {
                "time_published": item["time_published"],
                "summary": item["summary"],
                "source": item["source"],
                "relevance": item["ticker_sentiment"][0]["relevance_score"],
            }
            for item in data
        )

        df.index = pd.DatetimeIndex(df["time_published"])
        df.drop(columns="time_published", inplace=True)

        return df

    else:
        print(f"Error fetching news sentiment: {response.status_code} - {response.text}")
        return None

def get_stock_community_data(ticker):
    """Fetches stock community comments data from Yahoo Finance.

    Args:
        ticker (str): The stock ticker symbol.

    Returns:
        pd.DataFrame: A DataFrame containing comments, user IDs, timestamps, and other
            relevant information, indexed by the time the comment was posted.
    """
    base_url = f"https://finance.yahoo.com/quote/{ticker}/community/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/110.0"
    }

    response = requests.get(base_url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    try:
        config_data = json.loads(
            soup.select_one("#spotim-config").get_text(strip=True)
        )["config"]
    except (AttributeError, json.JSONDecodeError) as e:
        print(f"Error extracting or parsing config data: {e}")
        return None

    payload = json.dumps(
        {
            "conversation_id": config_data["spotId"]
            + config_data["uuid"].replace("_", "$"),
            "count": 250,
            "offset": 0,
        }
    )
    api_headers = {
        "User-Agent": headers["User-Agent"],
        "Content-Type": "application/json",
        "x-spot-id": config_data["spotId"],
        "x-post-id": config_data["uuid"].replace("_", "$"),
    }

    api_url = "https://api-2-0.spot.im/v1.0.0/conversation/read"
    response = requests.post(api_url, headers=api_headers, data=payload)

    if response.status_code != 200:
        print(f"API request failed with status code: {response.status_code}")
        return None
   
    data = response.json()
    comments_data = data["conversation"]["comments"]

    extracted_data = []
    for comment in comments_data:
        user_info = comment["user_id"]
        comment_info = next(
            (
                content["text"]
                for content in comment["content"]
                if "text" in content
            ),
            "",
        )
        time_info = pd.to_datetime(comment["written_at"], unit="s")
        replies_count = comment["replies_count"]
        rank_up = comment["rank"]["ranks_up"]
        rank_down = comment["rank"]["ranks_down"]
        rank_score = comment["rank_score"]

        extracted_data.append(
            {
                "User ID": user_info,
                "Comment": comment_info,
                "Time": time_info,
                "Replies Count": replies_count,
                "Rank Up": rank_up,
                "Rank Down": rank_down,
                "Rank Score": rank_score,
            }
        )

    df = pd.DataFrame(extracted_data)
    df.set_index("Time", inplace=True)

    return df
