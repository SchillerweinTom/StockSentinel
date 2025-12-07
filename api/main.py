from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import logging
from datetime import datetime

from src.data_collector import NewsCollector
from src.sentiment_analyzer import SentimentAnalyzer
from src.stock_scorer import StockScorer
from src.utils import ensure_directories, format_ticker, validate_ticker

# Setup
ensure_directories()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="StockSentinel API", description="Financial News Sentiment Analysis API", version="0.1.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components (singleton pattern)
collector = NewsCollector()
analyzer = SentimentAnalyzer()
scorer = StockScorer()


# Pydantic models
class AnalysisRequest(BaseModel):
    """Request model for sentiment analysis"""

    ticker: str = Field(..., description="Stock ticker symbol (e.g., AAPL)")
    days: int = Field(7, ge=1, le=30, description="Number of days to look back")
    max_articles: int = Field(50, ge=1, le=100, description="Maximum number of articles")


class SentimentResponse(BaseModel):
    """Response model for sentiment analysis"""

    ticker: str
    analysis_date: str
    stock_info: Dict[str, Any]
    sentiment_analysis: Dict[str, Any]
    scoring: Dict[str, Any]
    article_count: int
    top_articles: List[Dict[str, Any]]


# Routes
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "StockSentinel API",
        "version": "0.1.0",
        "endpoints": {
            "analyze": "/api/analyze/{ticker}",
            "info": "/api/stock-info/{ticker}",
            "health": "/health",
        },
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {"news_collector": "ok", "sentiment_analyzer": "ok", "stock_scorer": "ok"},
    }


@app.get("/api/analyze/{ticker}", response_model=SentimentResponse)
async def analyze_ticker(
    ticker: str,
    days: int = Query(7, ge=1, le=30, description="Days to look back"),
    max_articles: int = Query(50, ge=1, le=100, description="Max articles"),
):
    """
    Analyze sentiment for a single stock ticker using FinBERT

    - **ticker**: Stock ticker symbol (e.g., AAPL, GOOGL)
    - **days**: Number of days to look back (1-30)
    - **max_articles**: Maximum number of articles to analyze (1-100)
    """
    ticker = format_ticker(ticker)
    logger.info(f"Analyzing {ticker} with FinBERT, days={days}")

    # Validate ticker
    if not validate_ticker(ticker):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid ticker symbol: {ticker}. Please provide a valid stock ticker.",
        )

    try:
        # Step 1: Collect news
        articles = collector.collect_all(ticker, days=days, max_articles=max_articles)

        if not articles:
            raise HTTPException(
                status_code=404, detail=f"No news articles found for ticker {ticker}"
            )

        # Step 2: Analyze sentiment with FinBERT
        analyzed_articles = analyzer.analyze_articles(articles)

        # Step 3: Calculate scores
        aggregated = analyzer.get_aggregated_sentiment(analyzed_articles)
        scoring = scorer.calculate_score(aggregated, analyzed_articles)
        report = scorer.generate_report(ticker, aggregated, analyzed_articles, scoring)

        return report

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing {ticker}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stock-info/{ticker}")
async def get_stock_info(ticker: str):
    """
    Get basic stock information

    - **ticker**: Stock ticker symbol
    """
    ticker = format_ticker(ticker)

    try:
        info = scorer.get_stock_info(ticker)
        return info
    except Exception as e:
        logger.error(f"Error fetching stock info for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
