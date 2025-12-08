import logging
from typing import Dict, List, Any
from datetime import datetime
import yfinance as yf

from .utils import format_ticker

logger = logging.getLogger(__name__)


class StockScorer:
    """Generate stock scores and recommendations based on sentiment and other factors"""

    def __init__(self):
        """Initialize StockScorer"""
        self.weights = {
            "sentiment_score": 0.70,  # Main Factor: Sentiment
            "sentiment_consistency": 0.15,  # Only relevant with enough articles
            "news_volume": 0.10,  # Neutral, only punishes when there aren't enough articles
            "recency": 0.05,  # Minimal influence because default value of days is 7
        }
        self.min_articles_for_consistency = 10

    def calculate_score(
        self, aggregated_sentiment: Dict[str, Any], articles: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate overall stock score
        Args:
            aggregated_sentiment: Aggregated sentiment statistics
            articles: List of analyzed articles
        Returns:
            Dictionary with score and recommendation
        """
        article_count = len(articles)

        # Component scores
        sentiment_component = self._calculate_sentiment_component(aggregated_sentiment)
        consistency_component = self._calculate_consistency_component(
            aggregated_sentiment, article_count
        )
        volume_component = self._calculate_volume_component(article_count)
        recency_component = self._calculate_recency_component(articles)

        # Weighted total score
        total_score = (
            sentiment_component * self.weights["sentiment_score"]
            + consistency_component * self.weights["sentiment_consistency"]
            + volume_component * self.weights["news_volume"]
            + recency_component * self.weights["recency"]
        )

        # Amplify the score for better differentiation
        # Most real sentiment scores are between -0.3 and +0.3
        # We amplify to use the full 0-100 range more effectively
        amplification_factor = 1.5
        amplified_score = total_score * amplification_factor

        # Normalize to 0-100 with better spread
        normalized_score = (amplified_score + 1) * 50

        # Ensure bounds
        normalized_score = max(0, min(100, normalized_score))

        recommendation = self._generate_recommendation(normalized_score, aggregated_sentiment)

        return {
            "overall_score": round(normalized_score, 2),
            "components": {
                "sentiment": round(sentiment_component, 3),
                "consistency": round(consistency_component, 3),
                "volume": round(volume_component, 3),
                "recency": round(recency_component, 3),
            },
            "weights": self.weights,
            "recommendation": recommendation,
            "confidence": self._calculate_confidence(aggregated_sentiment, article_count),
        }

    def _calculate_sentiment_component(self, aggregated: Dict[str, Any]) -> float:
        """Calculate sentiment score component (-1 to 1)"""
        return aggregated.get("mean_score", 0.0)

    def _calculate_consistency_component(
        self, aggregated: Dict[str, Any], article_count: int
    ) -> float:
        """
        Calculate consistency score based on standard deviation
        Only applies penalty if enough articles are available
        Args:
            aggregated: Aggregated sentiment data
            article_count: Number of articles
        Returns:
            Consistency score (-1 to 1), continuous scale
        """
        # Consistency only matters with sufficient data
        if article_count < self.min_articles_for_consistency:
            return 0.0

        std = aggregated.get("std_score", 1.0)

        consistency = 1.0 - (std * 2.0)

        return max(-1.0, min(1.0, consistency))

    def _calculate_volume_component(self, article_count: int) -> float:
        """
        Calculate news volume score
        Penalty for too few articles, neutral for sufficient coverage
        No bonus for excessive articles
        Args:
            article_count: Number of articles
        Returns:
            Volume score (-1 to 0)
        """
        # Penalize insufficient data, no reward for excessive articles
        if article_count == 0:
            return -1.0
        elif article_count < 5:
            return -0.5
        elif article_count < 10:
            return -0.2
        else:
            return 0.0

    def _calculate_recency_component(self, articles: List[Dict[str, Any]]) -> float:
        """
        Calculate recency score based on article publication dates
        Less important since we typically look at recent timeframes (7 days)
        Args:
            articles: List of articles
        Returns:
            Recency score (-0.5 to 0.5) - limited range due to low importance
        """
        if not articles:
            return 0.0

        now = datetime.now()
        recency_scores = []

        for article in articles:
            try:
                pub_date = datetime.fromisoformat(
                    article.get("published_at", "").replace("Z", "+00:00")
                )
                hours_ago = (now - pub_date).total_seconds() / 3600

                if hours_ago < 24:
                    recency_scores.append(0.5)
                elif hours_ago < 48:
                    recency_scores.append(0.2)
                elif hours_ago < 96:
                    recency_scores.append(0.0)
                else:
                    recency_scores.append(-0.2)
            except:
                recency_scores.append(0.0)

        return sum(recency_scores) / len(recency_scores) if recency_scores else 0.0

    def _generate_recommendation(self, score: float, aggregated: Dict[str, Any]) -> str:
        """
        Generate recommendation based on score
        Args:
            score: Overall score (0-100)
            aggregated: Aggregated sentiment data
        Returns:
            Recommendation string
        """
        if score >= 70:
            return "STRONG BUY"
        elif score >= 60:
            return "BUY"
        elif score >= 55:
            return "WEAK BUY"
        elif score >= 45:
            return "HOLD"
        elif score >= 40:
            return "WEAK SELL"
        elif score >= 30:
            return "SELL"
        else:
            return "STRONG SELL"

    def _calculate_confidence(self, aggregated: Dict[str, Any], article_count: int) -> str:
        """
        Calculate confidence level of the recommendation
        Args:
            aggregated: Aggregated sentiment data
            article_count: Number of articles analyzed
        Returns:
            Confidence level: "high", "medium", or "low"
        """
        # Factors affecting confidence:
        # 1. Number of articles (more = higher confidence)
        # 2. Consistency (lower std = higher confidence)

        std = aggregated.get("std_score", 1.0)

        confidence_score = 0

        # Article count contribution
        if article_count >= 20:
            confidence_score += 2
        elif article_count >= 10:
            confidence_score += 1

        # Consistency contribution
        if std < 0.3:
            confidence_score += 2
        elif std < 0.5:
            confidence_score += 1

        if confidence_score >= 3:
            return "high"
        elif confidence_score >= 2:
            return "medium"
        else:
            return "low"

    def get_stock_info(self, ticker: str) -> Dict[str, Any]:
        """
        Get additional stock information from yfinance
        Args:
            ticker: Stock ticker symbol
        Returns:
            Dictionary with stock information
        """
        ticker = format_ticker(ticker)

        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            return {
                "ticker": ticker,
                "company_name": info.get("longName", ticker),
                "sector": info.get("sector", "Unknown"),
                "industry": info.get("industry", "Unknown"),
                "market_cap": info.get("marketCap"),
                "current_price": info.get("currentPrice"),
                "previous_close": info.get("previousClose"),
                "day_change_percent": self._calculate_change_percent(
                    info.get("currentPrice"), info.get("previousClose")
                ),
            }
        except Exception as e:
            logger.error(f"Error fetching stock info for {ticker}: {e}")
            return {
                "ticker": ticker,
                "company_name": ticker,
                "sector": "Unknown",
                "industry": "Unknown",
            }

    def _calculate_change_percent(self, current: float, previous: float) -> float:
        """Calculate percentage change"""
        if not current or not previous or previous == 0:
            return 0.0
        return ((current - previous) / previous) * 100

    def generate_report(
        self,
        ticker: str,
        aggregated_sentiment: Dict[str, Any],
        articles: List[Dict[str, Any]],
        scoring_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generate comprehensive analysis report
        Args:
            ticker: Stock ticker symbol
            aggregated_sentiment: Aggregated sentiment data
            articles: List of analyzed articles
            scoring_result: Stock scoring result
        Returns:
            Complete analysis report
        """
        stock_info = self.get_stock_info(ticker)

        return {
            "ticker": ticker,
            "analysis_date": datetime.now().isoformat(),
            "stock_info": stock_info,
            "sentiment_analysis": aggregated_sentiment,
            "scoring": scoring_result,
            "article_count": len(articles),
            "top_articles": self._get_top_articles(articles, n=5),
        }

    def _get_top_articles(self, articles: List[Dict[str, Any]], n: int = 5) -> List[Dict[str, Any]]:
        """Get top N articles by sentiment score magnitude"""
        if not articles:
            return []

        # Sort by absolute sentiment score
        sorted_articles = sorted(
            articles, key=lambda x: abs(x.get("sentiment", {}).get("score", 0)), reverse=True
        )

        top_articles = []
        for article in sorted_articles[:n]:
            top_articles.append(
                {
                    "title": article.get("title", ""),
                    "sentiment_score": article.get("sentiment", {}).get("score", 0),
                    "sentiment_label": article.get("sentiment_label", "neutral"),
                    "url": article.get("url", ""),
                    "published_at": article.get("published_at", ""),
                }
            )

        return top_articles
