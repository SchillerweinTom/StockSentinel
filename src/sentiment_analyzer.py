import logging
from typing import Dict, List, Any
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

from .utils import clean_text, calculate_sentiment_score

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """Analyze sentiment of financial news using AI models"""

    def __init__(self, model_name: str = "ProsusAI/finbert"):
        """
        Initialize sentiment analyzer
        Args:
            model_name: Hugging Face model name (default: FinBERT)
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

        try:
            logger.info(f"Loading model: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info("FinBERT model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading FinBERT model: {e}")
            raise RuntimeError(
                "Failed to load FinBERT model. Please check your internet connection."
            )

    def analyze_with_finbert(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment using FinBERT
        Args:
            text: Input text
        Returns:
            Dictionary with sentiment scores
        """
        text = clean_text(text)
        if not text:
            return {"positive": 0.0, "negative": 0.0, "neutral": 1.0, "score": 0.0}

        try:
            # Tokenize and predict
            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512, padding=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

            # FinBERT outputs: [positive, negative, neutral]
            probs = predictions[0].cpu().numpy()

            return {
                "positive": float(probs[0]),
                "negative": float(probs[1]),
                "neutral": float(probs[2]),
                "score": calculate_sentiment_score(
                    float(probs[0]), float(probs[1]), float(probs[2])
                ),
            }

        except Exception as e:
            logger.error(f"Error in FinBERT analysis: {e}")
            return {"positive": 0.0, "negative": 0.0, "neutral": 1.0, "score": 0.0}

    def analyze_article(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze sentiment of a single article using FinBERT
        Args:
            article: Article dictionary with 'title', 'description', 'content'
        Returns:
            Article with added sentiment analysis
        """
        # Combine title and description for analysis
        text = f"{article.get('title', '')} {article.get('description', '')}"

        sentiment = self.analyze_with_finbert(text)

        # Add sentiment to article
        article_with_sentiment = article.copy()
        article_with_sentiment["sentiment"] = sentiment
        article_with_sentiment["sentiment_label"] = self._get_label(sentiment["score"])

        return article_with_sentiment

    def analyze_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analyze sentiment of multiple articles using FinBERT
        Args:
            articles: List of article dictionaries
        Returns:
            Articles with sentiment analysis added
        """
        logger.info(f"Analyzing {len(articles)} articles using FinBERT")

        analyzed_articles = []
        for i, article in enumerate(articles):
            try:
                analyzed = self.analyze_article(article)
                analyzed_articles.append(analyzed)

                if (i + 1) % 10 == 0:
                    logger.info(f"Analyzed {i + 1}/{len(articles)} articles")

            except Exception as e:
                logger.error(f"Error analyzing article {i}: {e}")
                continue

        logger.info(f"Successfully analyzed {len(analyzed_articles)} articles")
        return analyzed_articles

    def get_aggregated_sentiment(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate aggregated sentiment from multiple articles
        Args:
            articles: List of analyzed articles
        Returns:
            Aggregated sentiment statistics
        """
        if not articles:
            return {
                "mean_score": 0.0,
                "median_score": 0.0,
                "positive_ratio": 0.0,
                "negative_ratio": 0.0,
                "neutral_ratio": 0.0,
                "article_count": 0,
                "overall_label": "neutral",
            }

        scores = []
        positive_count = 0
        negative_count = 0
        neutral_count = 0

        for article in articles:
            sentiment = article.get("sentiment", {})
            score = sentiment.get("score", 0.0)
            scores.append(score)

            label = article.get("sentiment_label", "neutral")
            if label == "bullish":
                positive_count += 1
            elif label == "bearish":
                negative_count += 1
            else:
                neutral_count += 1

        mean_score = np.mean(scores)
        total = len(articles)

        return {
            "mean_score": float(mean_score),
            "median_score": float(np.median(scores)),
            "std_score": float(np.std(scores)),
            "min_score": float(np.min(scores)),
            "max_score": float(np.max(scores)),
            "positive_ratio": positive_count / total,
            "negative_ratio": negative_count / total,
            "neutral_ratio": neutral_count / total,
            "article_count": total,
            "overall_label": self._get_label(mean_score),
        }

    def _get_label(self, score: float, threshold: float = 0.3) -> str:
        """
        Convert sentiment score to label
        Args:
            score: Sentiment score
            threshold: Classification threshold
        Returns:
            "bullish", "neutral", or "bearish"
        """
        if score > threshold:
            return "bullish"
        elif score < -threshold:
            return "bearish"
        else:
            return "neutral"
