import pytest
from datetime import datetime
from src.utils import (
    format_ticker,
    validate_ticker,
    get_date_range,
    clean_text,
    calculate_sentiment_score,
    get_sentiment_label,
    aggregate_sentiments,
)


class TestTickerFunctions:
    """Test ticker-related functions"""

    def test_format_ticker(self):
        """Test ticker formatting"""
        assert format_ticker("aapl") == "AAPL"
        assert format_ticker("  googl  ") == "GOOGL"
        assert format_ticker("MSFT") == "MSFT"

    def test_validate_ticker(self):
        """Test ticker validation"""
        assert validate_ticker("AAPL") == True
        assert validate_ticker("GOOGL") == True
        assert validate_ticker("A") == True

        # Invalid tickers
        assert validate_ticker("") == False
        assert validate_ticker("TOOLONG") == False
        assert validate_ticker("123") == False
        assert validate_ticker("AAP@") == False


class TestDateFunctions:
    """Test date-related functions"""

    def test_get_date_range(self):
        """Test date range generation"""
        start, end = get_date_range(days=7)

        # Check format
        assert len(start) == 10  # YYYY-MM-DD
        assert len(end) == 10

        # Parse dates
        start_date = datetime.strptime(start, "%Y-%m-%d")
        end_date = datetime.strptime(end, "%Y-%m-%d")

        # Check difference
        diff = (end_date - start_date).days
        assert diff == 7

    def test_get_date_range_different_days(self):
        """Test with different day values"""
        start1, end1 = get_date_range(days=1)
        start30, end30 = get_date_range(days=30)

        diff1 = (datetime.strptime(end1, "%Y-%m-%d") - datetime.strptime(start1, "%Y-%m-%d")).days
        diff30 = (
            datetime.strptime(end30, "%Y-%m-%d") - datetime.strptime(start30, "%Y-%m-%d")
        ).days

        assert diff1 == 1
        assert diff30 == 30


class TestTextCleaning:
    """Test text processing functions"""

    def test_clean_text(self):
        """Test text cleaning"""
        # Extra whitespace
        assert clean_text("  hello   world  ") == "hello world"

        # Empty string
        assert clean_text("") == ""
        assert clean_text(None) == ""

        # Normal text
        assert clean_text("This is a test.") == "This is a test."

    def test_clean_text_special_chars(self):
        """Test cleaning with special characters"""
        text = "Hello,  world!  How are you?"
        cleaned = clean_text(text)
        assert cleaned == "Hello, world! How are you?"


class TestSentimentFunctions:
    """Test sentiment calculation functions"""

    def test_calculate_sentiment_score(self):
        """Test sentiment score calculation"""
        # Positive sentiment
        score = calculate_sentiment_score(0.8, 0.1, 0.1)
        assert score == pytest.approx(0.7)

        # Negative sentiment
        score = calculate_sentiment_score(0.1, 0.8, 0.1)
        assert score == pytest.approx(-0.7)

        # Neutral
        score = calculate_sentiment_score(0.3, 0.3, 0.4)
        assert score == pytest.approx(0.0)

    def test_get_sentiment_label(self):
        """Test sentiment label generation"""
        assert get_sentiment_label(0.5) == "bullish"
        assert get_sentiment_label(-0.5) == "bearish"
        assert get_sentiment_label(0.0) == "neutral"
        assert get_sentiment_label(0.2) == "neutral"

        # With custom threshold
        assert get_sentiment_label(0.2, threshold=0.1) == "bullish"
        assert get_sentiment_label(-0.2, threshold=0.1) == "bearish"

    def test_aggregate_sentiments(self):
        """Test sentiment aggregation"""
        sentiments = [0.5, 0.3, -0.2, 0.8, -0.1]
        result = aggregate_sentiments(sentiments)

        assert "mean" in result
        assert "median" in result
        assert "std" in result
        assert result["count"] == 5
        assert -1 <= result["mean"] <= 1

    def test_aggregate_sentiments_empty(self):
        """Test aggregation with empty list"""
        result = aggregate_sentiments([])

        assert result["mean"] == 0.0
        assert result["count"] == 0


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_format_ticker_edge_cases(self):
        """Test ticker formatting edge cases"""
        assert format_ticker("a") == "A"
        assert format_ticker("AAAAA") == "AAAAA"

    def test_sentiment_extremes(self):
        """Test with extreme sentiment values"""
        # Maximum positive
        score = calculate_sentiment_score(1.0, 0.0, 0.0)
        assert score == 1.0

        # Maximum negative
        score = calculate_sentiment_score(0.0, 1.0, 0.0)
        assert score == -1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
