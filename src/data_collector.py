import os
import requests
import logging
from typing import List, Dict, Any
from datetime import datetime, timedelta
from dotenv import load_dotenv
import yfinance as yf

from .utils import get_date_range, format_ticker, save_json

load_dotenv()

logger = logging.getLogger(__name__)


class NewsCollector:
    """Collect financial news from multiple sources"""

    def __init__(self):
        """Initialize NewsCollector with API keys from environment"""
        self.news_api_key = os.getenv("NEWS_API_KEY")
        self.alpha_vantage_key = os.getenv("ALPHA_VANTAGE_KEY")
        self.finnhub_key = os.getenv("FINNHUB_API_KEY")

        if not any([self.news_api_key, self.finnhub_key]):
            logger.warning("No API keys found. Please set up .env file.")

    def collect_news_api(self, ticker: str, days: int = 7) -> List[Dict[str, Any]]:
        """
        Collect news from NewsAPI
        Args:
            ticker: Stock ticker symbol
            days: Number of days to look back
        Returns:
            List of news articles
        """
        if not self.news_api_key:
            logger.error("NEWS_API_KEY not found in environment")
            return []

        ticker = format_ticker(ticker)
        start_date, end_date = get_date_range(days)

        url = "https://newsapi.org/v2/everything"
        params = {
            "q": f"{ticker} stock OR {ticker} shares",
            "from": start_date,
            "to": end_date,
            "language": "en",
            "sortBy": "publishedAt",
            "apiKey": self.news_api_key,
            "pageSize": 100,
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            articles = []
            for article in data.get("articles", []):
                articles.append(
                    {
                        "source": "newsapi",
                        "title": article.get("title", ""),
                        "description": article.get("description", ""),
                        "content": article.get("content", ""),
                        "url": article.get("url", ""),
                        "published_at": article.get("publishedAt", ""),
                        "source_name": article.get("source", {}).get("name", "Unknown"),
                    }
                )

            logger.info(f"Collected {len(articles)} articles from NewsAPI for {ticker}")
            return articles

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching from NewsAPI: {e}")
            return []

    def collect_finnhub(self, ticker: str, days: int = 7) -> List[Dict[str, Any]]:
        """
        Collect news from Finnhub API
        Args:
            ticker: Stock ticker symbol
            days: Number of days to look back
        Returns:
            List of news articles
        """
        if not self.finnhub_key:
            logger.error("FINNHUB_API_KEY not found in environment")
            return []

        ticker = format_ticker(ticker)
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        end_date = datetime.now().strftime("%Y-%m-%d")

        url = "https://finnhub.io/api/v1/company-news"
        params = {"symbol": ticker, "from": start_date, "to": end_date, "token": self.finnhub_key}

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            articles = []
            for item in data:
                articles.append(
                    {
                        "source": "finnhub",
                        "title": item.get("headline", ""),
                        "description": item.get("summary", ""),
                        "content": item.get("summary", ""),
                        "url": item.get("url", ""),
                        "published_at": datetime.fromtimestamp(item.get("datetime", 0)).isoformat(),
                        "source_name": item.get("source", "Finnhub"),
                    }
                )

            logger.info(f"Collected {len(articles)} articles from Finnhub for {ticker}")
            return articles

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching from Finnhub: {e}")
            return []

    def collect_yfinance_news(self, ticker: str) -> List[Dict[str, Any]]:
        """
        Collect news from Yahoo Finance via yfinance
        Args:
            ticker: Stock ticker symbol
        Returns:
            List of news articles
        """
        ticker = format_ticker(ticker)

        try:
            stock = yf.Ticker(ticker)
            news = stock.news

            articles = []
            for item in news:
                articles.append(
                    {
                        "source": "yahoo_finance",
                        "title": item.get("title", ""),
                        "description": item.get("summary", ""),
                        "content": item.get("summary", ""),
                        "url": item.get("link", ""),
                        "published_at": datetime.fromtimestamp(
                            item.get("providerPublishTime", 0)
                        ).isoformat(),
                        "source_name": item.get("publisher", "Yahoo Finance"),
                    }
                )

            logger.info(f"Collected {len(articles)} articles from Yahoo Finance for {ticker}")
            return articles

        except Exception as e:
            logger.error(f"Error fetching from Yahoo Finance: {e}")
            return []

    def collect_all(
        self, ticker: str, days: int = 7, max_articles: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Collect news from all available sources
        Args:
            ticker: Stock ticker symbol
            days: Number of days to look back
            max_articles: Maximum number of articles to return
        Returns:
            Combined list of news articles
        """
        ticker = format_ticker(ticker)
        logger.info(f"Collecting news for {ticker} from all sources...")

        all_articles = []

        # Collect from all sources
        all_articles.extend(self.collect_news_api(ticker, days))
        all_articles.extend(self.collect_finnhub(ticker, days))
        all_articles.extend(self.collect_yfinance_news(ticker))

        # Remove duplicates based on title
        seen_titles = set()
        unique_articles = []
        for article in all_articles:
            title = article.get("title", "").lower()
            if title and title not in seen_titles:
                seen_titles.add(title)
                unique_articles.append(article)

        # Sort by date (newest first)
        unique_articles.sort(key=lambda x: x.get("published_at", ""), reverse=True)

        # Limit to max_articles
        result = unique_articles[:max_articles]

        logger.info(f"Total unique articles collected: {len(result)}")
        return result

    def save_news(self, ticker: str, articles: List[Dict[str, Any]], output_dir: str = "data/raw"):
        """
        Save collected news to JSON file
        Args:
            ticker: Stock ticker symbol
            articles: List of news articles
            output_dir: Output directory
        """
        ticker = format_ticker(ticker)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{output_dir}/{ticker}_news_{timestamp}.json"

        data = {
            "ticker": ticker,
            "collected_at": datetime.now().isoformat(),
            "article_count": len(articles),
            "articles": articles,
        }

        save_json(data, filename)
        logger.info(f"Saved {len(articles)} articles to {filename}")
