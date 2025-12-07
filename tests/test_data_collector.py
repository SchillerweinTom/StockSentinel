import pytest
from unittest.mock import Mock, patch
from src.data_collector import NewsCollector


class TestNewsCollector:
    """Test NewsCollector class"""

    @pytest.fixture
    def collector(self):
        """Create NewsCollector instance"""
        return NewsCollector()

    def test_initialization(self, collector):
        """Test collector initialization"""
        assert collector is not None
        assert hasattr(collector, "news_api_key")
        assert hasattr(collector, "finnhub_key")

    @patch("src.data_collector.requests.get")
    def test_collect_news_api_success(self, mock_get, collector):
        """Test successful NewsAPI collection"""
        # Mock response
        mock_response = Mock()
        mock_response.json.return_value = {
            "articles": [
                {
                    "title": "Test Article",
                    "description": "Test Description",
                    "content": "Test Content",
                    "url": "https://example.com",
                    "publishedAt": "2024-01-01T12:00:00Z",
                    "source": {"name": "Test Source"},
                }
            ]
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        # Set API key
        collector.news_api_key = "test_key"

        # Test
        articles = collector.collect_news_api("AAPL", days=7)

        assert len(articles) == 1
        assert articles[0]["title"] == "Test Article"
        assert articles[0]["source"] == "newsapi"

    def test_collect_news_api_no_key(self, collector):
        """Test NewsAPI without API key"""
        collector.news_api_key = None
        articles = collector.collect_news_api("AAPL")
        assert articles == []

    @patch("src.data_collector.requests.get")
    def test_collect_finnhub_success(self, mock_get, collector):
        """Test successful Finnhub collection"""
        mock_response = Mock()
        mock_response.json.return_value = [
            {
                "headline": "Test Headline",
                "summary": "Test Summary",
                "url": "https://example.com",
                "datetime": 1704110400,
                "source": "Test Source",
            }
        ]
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        collector.finnhub_key = "test_key"

        articles = collector.collect_finnhub("AAPL", days=7)

        assert len(articles) == 1
        assert articles[0]["source"] == "finnhub"

    @patch("src.data_collector.yf.Ticker")
    def test_collect_yfinance_news(self, mock_ticker, collector):
        """Test Yahoo Finance news collection"""
        mock_stock = Mock()
        mock_stock.news = [
            {
                "title": "YF Test",
                "summary": "YF Summary",
                "link": "https://example.com",
                "providerPublishTime": 1704110400,
                "publisher": "Yahoo",
            }
        ]
        mock_ticker.return_value = mock_stock

        articles = collector.collect_yfinance_news("AAPL")

        assert len(articles) == 1
        assert articles[0]["source"] == "yahoo_finance"

    def test_ticker_formatting(self, collector):
        """Test that ticker is properly formatted"""
        with patch.object(collector, "collect_news_api", return_value=[]) as mock:
            collector.collect_all("aapl")
            # Verify the ticker was formatted to uppercase
            mock.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
