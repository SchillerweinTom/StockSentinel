import argparse
import sys
import logging
from typing import Dict
from datetime import datetime

from .data_collector import NewsCollector
from .sentiment_analyzer import SentimentAnalyzer
from .stock_scorer import StockScorer
from .utils import ensure_directories, setup_logging, save_json, format_ticker, validate_ticker


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="StockSentinel - Financial News Sentiment Analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze single stock
  python -m src.cli --ticker AAPL --days 7
  
  # Save results to specific file
  python -m src.cli --ticker NVDA --output results/nvda_analysis.json
        """,
    )

    # Ticker arguments
    parser.add_argument(
        "--ticker", type=str, required=True, help="Single stock ticker symbol (e.g., AAPL)"
    )

    # Analysis parameters
    parser.add_argument(
        "--days", type=int, default=7, help="Number of days to look back for news (default: 7)"
    )

    parser.add_argument(
        "--max-articles",
        type=int,
        default=50,
        help="Maximum number of articles per stock (default: 50)",
    )

    # Output options
    parser.add_argument(
        "--output", type=str, help="Output file path for results (default: data/processed/)"
    )

    args = parser.parse_args()

    # Setup
    ensure_directories()
    log_level = "INFO"
    setup_logging(log_level)
    logger = logging.getLogger(__name__)

    # Format ticker
    ticker = format_ticker(args.ticker)

    # Validate ticker
    logger.info(f"Validating ticker: {ticker}")
    if not validate_ticker(ticker):
        logger.error(f"Invalid ticker symbol: {ticker}")
        logger.error("Please provide a valid stock ticker (e.g., AAPL, GOOGL, MSFT)")
        sys.exit(1)

    logger.info(f"Starting analysis for: {ticker}")
    logger.info(f"Parameters: days={args.days}, max_articles={args.max_articles}")

    # Initialize components
    collector = NewsCollector()
    analyzer = SentimentAnalyzer()
    scorer = StockScorer()

    logger.info(f"\n{'='*60}")
    logger.info(f"Analyzing {ticker}")
    logger.info(f"{'='*60}")

    try:
        # Step 1: Collect news
        logger.info("Step 1/3: Collecting news...")
        articles = collector.collect_all(ticker, days=args.days, max_articles=args.max_articles)

        if not articles:
            logger.error(f"No articles found for {ticker}")
            sys.exit(1)

        logger.info(f"Collected {len(articles)} articles")

        # Step 2: Analyze sentiment
        logger.info("Step 2/3: Analyzing sentiment with FinBERT...")
        analyzed_articles = analyzer.analyze_articles(articles)

        # Step 3: Calculate scores and generate report
        logger.info("Step 3/3: Generating scores and recommendations...")
        aggregated = analyzer.get_aggregated_sentiment(analyzed_articles)
        scoring = scorer.calculate_score(aggregated, analyzed_articles)
        report = scorer.generate_report(ticker, aggregated, analyzed_articles, scoring)

        # Print summary
        print_summary(ticker, report)

        # Save results
        if args.output:
            output_path = args.output
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"data/processed/analysis_{timestamp}.json"

        save_json(report, output_path)
        logger.info(f"\nResults saved to: {output_path}")
        logger.info("\nAnalysis complete!")

    except Exception as e:
        logger.error(f"Error analyzing {ticker}: {e}", exc_info=True)
        sys.exit(1)


def print_summary(ticker: str, report: Dict):
    """Print analysis summary to console"""
    print(f"\n{'='*60}")
    print(f"ANALYSIS SUMMARY: {ticker}")
    print(f"{'='*60}")

    stock_info = report.get("stock_info", {})
    print(f"\nCompany: {stock_info.get('company_name', ticker)}")
    print(f"Sector: {stock_info.get('sector', 'Unknown')}")

    if stock_info.get("current_price"):
        print(f"Current Price: ${stock_info['current_price']:.2f}")
        change = stock_info.get("day_change_percent", 0)
        print(f"Day Change: {change:+.2f}%")

    sentiment = report.get("sentiment_analysis", {})
    print(f"\nSentiment Analysis ({sentiment.get('article_count', 0)} articles):")
    print(f"  Mean Score: {sentiment.get('mean_score', 0):.3f}")
    print(f"  Overall Label: {sentiment.get('overall_label', 'neutral').upper()}")
    print(f"  Positive Ratio: {sentiment.get('positive_ratio', 0):.1%}")
    print(f"  Negative Ratio: {sentiment.get('negative_ratio', 0):.1%}")

    scoring = report.get("scoring", {})
    print(f"\nStock Score: {scoring.get('overall_score', 0):.1f}/100")
    print(f"Recommendation: {scoring.get('recommendation', 'HOLD')}")
    print(f"Confidence: {scoring.get('confidence', 'medium').upper()}")

    components = scoring.get("components", {})
    print(f"\nScore Components:")
    print(f"  Sentiment: {components.get('sentiment', 0):+.3f}")
    print(f"  Consistency: {components.get('consistency', 0):+.3f}")
    print(f"  Volume: {components.get('volume', 0):+.3f}")
    print(f"  Recency: {components.get('recency', 0):+.3f}")

    print(f"\nTop Articles:")
    for i, article in enumerate(report.get("top_articles", [])[:3], 1):
        print(f"\n  {i}. {article.get('title', '')}")
        print(
            f"     Sentiment: {article.get('sentiment_label', 'neutral').upper()} ({article.get('sentiment_score', 0):+.3f})"
        )

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
