import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any
from pathlib import Path
import pandas as pd
import yfinance as yf


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler("logs/stocksentinel.log"), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


def ensure_directories():
    """Create necessary directories if they don't exist"""
    directories = ["data/processed", "logs", "models"]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


def save_json(data: Dict[str, Any], filepath: str) -> None:
    """Save data to JSON file"""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(filepath: str) -> Dict[str, Any]:
    """Load data from JSON file"""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def get_date_range(days: int = 7) -> tuple:
    """
    Get date range for news collection
    Args:
        days: Number of days to look back
    Returns:
        Tuple of (start_date, end_date) as strings
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")


def clean_text(text: str) -> str:
    """
    Clean text for sentiment analysis
    Args:
        text: Raw text string
    Returns:
        Cleaned text
    """
    if not text:
        return ""

    # Remove extra whitespace
    text = " ".join(text.split())

    # Remove special characters but keep basic punctuation
    # text = re.sub(r'[^a-zA-Z0-9\s\.\,\!\?]', '', text)

    return text.strip()


def calculate_sentiment_score(positive: float, negative: float, neutral: float) -> float:
    """
    Calculate overall sentiment score from probabilities
    Args:
        positive: Positive sentiment probability
        negative: Negative sentiment probability
        neutral: Neutral sentiment probability
    Returns:
        Sentiment score between -1 (negative) and 1 (positive)
    """
    return positive - negative


def aggregate_sentiments(sentiments: List[float]) -> Dict[str, Any]:
    """
    Aggregate multiple sentiment scores
    Args:
        sentiments: List of sentiment scores
    Returns:
        Dictionary with aggregated statistics
    """
    if not sentiments:
        return {"mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "count": 0}

    df = pd.Series(sentiments)
    return {
        "mean": float(df.mean()),
        "median": float(df.median()),
        "std": float(df.std()),
        "min": float(df.min()),
        "max": float(df.max()),
        "count": len(sentiments),
    }


def format_ticker(ticker: str) -> str:
    """Format ticker symbol to uppercase"""
    return ticker.upper().strip()


def validate_ticker(ticker: str) -> bool:
    """
    Validate ticker symbol format and check if it exists
    Args:
        ticker: Stock ticker symbol
    Returns:
        True if valid format and exists
    """
    ticker = ticker.upper().strip()

    # Basic format validation
    if not ticker or len(ticker) < 1 or len(ticker) > 5:
        return False

    if not ticker.isalpha():
        return False

    # Try to verify ticker exists using yfinance
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        # yfinance returns an almost empty dict for invalid tickers
        if not info or len(info) < 5:
            return False

        # Check for key fields that indicate a valid stock
        if "symbol" not in info and "shortName" not in info and "longName" not in info:
            return False

        return True

    except Exception:
        # If yfinance fails, accept based on format only
        return True


def get_sentiment_label(score: float, threshold: float = 0.3) -> str:
    """
    Convert sentiment score to label
    Args:
        score: Sentiment score between -1 and 1
        threshold: Threshold for positive/negative classification
    Returns:
        "bullish", "neutral", or "bearish"
    """
    if score > threshold:
        return "bullish"
    elif score < -threshold:
        return "bearish"
    else:
        return "neutral"
