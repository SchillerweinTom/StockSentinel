import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_collector import NewsCollector
from src.sentiment_analyzer import SentimentAnalyzer
from src.stock_scorer import StockScorer
from src.utils import ensure_directories, format_ticker, validate_ticker

# Page config
st.set_page_config(
    page_title="StockSentinel", page_icon="📈", layout="wide", initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown(
    """
<style>
    .big-font {
        font-size:20px !important;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""",
    unsafe_allow_html=True,
)


# Initialize session state
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None


def create_sentiment_gauge(score, label):
    """Create a gauge chart for sentiment score"""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=score,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": label, "font": {"size": 24}},
            delta={"reference": 0},
            gauge={
                "axis": {"range": [-1, 1], "tickwidth": 1, "tickcolor": "darkblue"},
                "bar": {"color": "darkblue"},
                "bgcolor": "white",
                "borderwidth": 2,
                "bordercolor": "gray",
                "steps": [
                    {"range": [-1, -0.3], "color": "#ffcccb"},
                    {"range": [-0.3, 0.3], "color": "#ffffcc"},
                    {"range": [0.3, 1], "color": "#ccffcc"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": score,
                },
            },
        )
    )

    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))

    return fig


def create_sentiment_distribution(articles):
    """Create distribution chart of sentiment scores"""
    sentiments = [a["sentiment"]["score"] for a in articles if "sentiment" in a]

    fig = px.histogram(
        x=sentiments,
        nbins=30,
        title="Sentiment Score Distribution",
        labels={"x": "Sentiment Score", "y": "Count"},
    )

    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    fig.update_layout(height=300)

    return fig


def create_timeline_chart(articles):
    """Create timeline of articles with sentiment"""
    data = []
    for article in articles:
        if "sentiment" in article:
            data.append(
                {
                    "date": article.get("published_at", ""),
                    "sentiment": article["sentiment"]["score"],
                    "title": article.get("title", "")[:50] + "...",
                }
            )

    df = pd.DataFrame(data)
    if df.empty:
        return None

    df["date"] = pd.to_datetime(df["date"], format="ISO8601", utc=True)
    df = df.sort_values("date")

    fig = px.scatter(
        df,
        x="date",
        y="sentiment",
        hover_data=["title"],
        title="Sentiment Over Time",
        color="sentiment",
        color_continuous_scale=["red", "yellow", "green"],
    )

    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(height=400)

    return fig


def display_recommendation(scoring):
    """Display recommendation with color coding"""
    recommendation = scoring["recommendation"]
    score = scoring["overall_score"]
    confidence = scoring["confidence"]

    # Color coding
    if "BUY" in recommendation:
        color = "green"
    elif "SELL" in recommendation:
        color = "red"
    else:
        color = "orange"

    st.markdown(
        f"""
    <div style='background-color: {color}; padding: 20px; border-radius: 10px; text-align: center;'>
        <h1 style='color: white; margin: 0;'>{recommendation}</h1>
        <h3 style='color: white; margin: 10px 0 0 0;'>Score: {score:.1f}/100</h3>
        <p style='color: white; margin: 5px 0 0 0;'>Confidence: {confidence.upper()}</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


def main():
    """Main Streamlit app"""
    ensure_directories()

    # Header
    st.title("📈 StockSentinel")
    st.markdown("### Financial News Sentiment Analysis for Stock Market Insights")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Input
        ticker_input = st.text_input(
            "Stock Ticker Symbol",
            value="AAPL",
            help="Enter a stock ticker (e.g., AAPL, GOOGL, TSLA)",
        ).upper()

        days = st.slider(
            "Days to Analyze",
            min_value=1,
            max_value=30,
            value=7,
            help="Number of days to look back for news",
        )

        max_articles = st.slider("Max Articles", min_value=10, max_value=100, value=50, step=10)

        analyze_button = st.button("🔍 Analyze", type="primary", use_container_width=True)

        st.markdown("---")
        st.markdown("### About")
        st.info(
            "This tool analyzes financial news sentiment to help identify "
            "potential investment opportunities. **Not financial advice!**"
        )

    # Main content
    if analyze_button:
        # Validate ticker first
        ticker_formatted = format_ticker(ticker_input)

        with st.spinner(f"Validating ticker {ticker_formatted}..."):
            if not validate_ticker(ticker_formatted):
                st.error(f"❌ Invalid ticker symbol: **{ticker_formatted}**")
                st.warning(
                    "Please enter a valid stock ticker symbol (e.g., AAPL, GOOGL, MSFT, TSLA)"
                )
                st.info("💡 Tip: Make sure the ticker is listed on a major stock exchange.")
                return

        with st.spinner(f"Analyzing {ticker_formatted}... This may take a moment."):
            try:
                # Initialize components
                collector = NewsCollector()
                analyzer = SentimentAnalyzer()
                scorer = StockScorer()

                # Collect news
                st.info(f"📰 Collecting news for {ticker_formatted}...")
                articles = collector.collect_all(
                    ticker_formatted, days=days, max_articles=max_articles
                )

                if not articles:
                    st.error(f"No news articles found for {ticker_formatted}")
                    st.info("💡 Try:")
                    st.markdown("- A different ticker symbol")
                    st.markdown("- Increasing the number of days")
                    st.markdown("- Checking if the company has recent news coverage")
                    return

                st.success(f"Collected {len(articles)} articles")

                # Analyze sentiment
                st.info("🧠 Analyzing sentiment with FinBERT...")
                analyzed_articles = analyzer.analyze_articles(articles)

                # Calculate scores
                st.info("📊 Calculating scores...")
                aggregated = analyzer.get_aggregated_sentiment(analyzed_articles)
                scoring = scorer.calculate_score(aggregated, analyzed_articles)
                report = scorer.generate_report(
                    ticker_formatted, aggregated, analyzed_articles, scoring
                )

                st.session_state.analysis_results = report

            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
                return

    # Display results
    if st.session_state.analysis_results:
        report = st.session_state.analysis_results

        # Stock info
        stock_info = report["stock_info"]
        st.markdown(f"## {stock_info.get('company_name', report['ticker'])}")
        st.markdown(
            f"**Sector:** {stock_info.get('sector', 'Unknown')} | "
            f"**Industry:** {stock_info.get('industry', 'Unknown')}"
        )

        # Key metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if stock_info.get("current_price"):
                st.metric(
                    "Current Price",
                    f"${stock_info['current_price']:.2f}",
                    f"{stock_info.get('day_change_percent', 0):+.2f}%",
                )

        with col2:
            sentiment = report["sentiment_analysis"]
            st.metric(
                "Sentiment Score",
                f"{sentiment['mean_score']:.3f}",
                sentiment["overall_label"].upper(),
            )

        with col3:
            st.metric("Articles Analyzed", sentiment["article_count"])

        with col4:
            st.metric("Overall Score", f"{report['scoring']['overall_score']:.1f}/100")

        st.markdown("---")

        # Recommendation
        st.markdown("### 🎯 Recommendation")
        display_recommendation(report["scoring"])

        st.markdown("---")

        # Visualizations
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Sentiment Gauge")
            gauge_fig = create_sentiment_gauge(
                sentiment["mean_score"], f"{report['ticker']} Sentiment"
            )
            st.plotly_chart(gauge_fig, use_container_width=True)

        with col2:
            st.markdown("### Sentiment Distribution")
            dist_fig = create_sentiment_distribution(analyzed_articles)
            st.plotly_chart(dist_fig, use_container_width=True)

        # Timeline
        st.markdown("### 📅 Sentiment Timeline")
        timeline_fig = create_timeline_chart(analyzed_articles)
        if timeline_fig:
            st.plotly_chart(timeline_fig, use_container_width=True)

        st.markdown("---")

        # Top articles
        st.markdown("### 📰 Top Articles")

        for i, article in enumerate(report["top_articles"][:5], 1):
            with st.expander(f"{i}. {article['title']}"):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**Published:** {article['published_at']}")
                    st.markdown(f"[Read full article]({article['url']})")
                with col2:
                    label = article["sentiment_label"]
                    score = article["sentiment_score"]
                    if label == "bullish":
                        st.success(f"🟢 {label.upper()}\n\n{score:+.3f}")
                    elif label == "bearish":
                        st.error(f"🔴 {label.upper()}\n\n{score:+.3f}")
                    else:
                        st.warning(f"🟡 {label.upper()}\n\n{score:+.3f}")


if __name__ == "__main__":
    main()
