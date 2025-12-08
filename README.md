# StockSentinel - Financial News Sentiment Analyzer

Automated sentiment analysis of financial news to identify stock investment opportunities.

## Project Overview

StockSentinel is a tool that:

- Collects financial news from multiple sources (NewsAPI, Finnhub, Yahoo Finance)
- Analyzes sentiment using NLP model FinBERT
- Scores stocks based on sentiment, consistency, volume, and recency
- Recommends investment decisions (BUY / HOLD / SELL) with confidence levels

---

### ⚠️ Disclaimer
This tool is for **educational and research purposes only**.  
**Not financial advice!**

## Setup Instructions

### Prerequisites

- Python 3.10+
- `pyenv` and `pyenv-virtualenv` (recommended)
- API Keys (free):
  - [NewsAPI](https://newsapi.org/) - 100 requests/day
  - [Finnhub](https://finnhub.io/) - 60 calls/minute

### Installation

```bash
# 1. Clone repository
git clone https://github.com/SchillerweinTom/StockSentinel
cd stocksentinel

# 2. Create virtual environment
pyenv install 3.10.6
pyenv virtualenv 3.10.6 stocksentinel
pyenv activate stocksentinel

# 3. Install dependencies
pip install -r requirements.txt
pip install -e .

# 4. Setup environment variables
touch .env
# Edit .env and add your API keys
```

### Environment Configuration

Add your API keys to `.env`:

```bash
NEWS_API_KEY=your_newsapi_key_here
FINNHUB_API_KEY=your_finnhub_key_here
```

---

## How to Run

### Option 1: Command-Line Interface (CLI)

```bash
# Analyze a single stock
python -m src.cli --ticker AAPL --days 7

# With custom settings
python -m src.cli --ticker TSLA --days 5 --max-articles 30

# Save results to file
python -m src.cli --ticker NVDA --output results/nvda_analysis.json
```

**CLI Arguments:**
- `--ticker`: Stock ticker symbol (required)
- `--days`: Number of days to look back (default: 7)
- `--max-articles`: Maximum articles to analyze (default: 50)
- `--output`: Output file path (optional)

### Option 2: FastAPI Backend

```bash
# Start API server
uvicorn api.main:app --reload --port 8000

# Or using Makefile
make run-api
```

**Access API Documentation:** http://localhost:8000/docs

**Example API Calls:**

```bash
# Analyze stock
curl "http://localhost:8000/api/analyze/AAPL?days=7"

# Get stock info
curl "http://localhost:8000/api/stock-info/AAPL"

# API health check
curl "http://localhost:8000/health"
```

### Option 3: Streamlit Frontend

```bash
# Start web interface
streamlit run frontend/app.py

# Or using Makefile
make run-frontend
```

**Access Dashboard:** http://localhost:8501

---

## Expected Output

### CLI Output Example

```
============================================================
ANALYSIS SUMMARY: AAPL
============================================================

Company: Apple Inc.
Sector: Technology

Sentiment Analysis (47 articles):
  Mean Score: +0.342
  Overall Label: BULLISH
  Positive Ratio: 68.1%
  Negative Ratio: 14.9%

Stock Score: 73.2/100
Recommendation: BUY
Confidence: HIGH

Score Components:
  Sentiment: +0.342
  Consistency: +0.567
  Volume: 0.000
  Recency: +0.234

Top Articles:
  1. Apple announces new AI features
     Sentiment: BULLISH (+0.789)
```

### API Response Example

```json
{
  "ticker": "AAPL",
  "analysis_date": "2024-12-08T10:30:00",
  "stock_info": {
    "company_name": "Apple Inc.",
    "sector": "Technology",
    "current_price": 195.50
  },
  "sentiment_analysis": {
    "mean_score": 0.342,
    "overall_label": "bullish",
    "positive_ratio": 0.681,
    "article_count": 47
  },
  "scoring": {
    "overall_score": 73.2,
    "recommendation": "BUY",
    "confidence": "high"
  }
}
```

## Docker Deployment

### Build

```bash
# Build Docker image
docker build -t stocksentinel:latest .

# Or use Make commands
make docker-build
```

### Docker Compose (Full Stack)

```bash
# Start API + Frontend
docker-compose up --build

# Or using Makefile
make docker-compose
```

**Services:**
- API: http://localhost:8000
- Frontend: http://localhost:8501

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Or using Makefile
make test
```

**CI Pipeline (Format + Lint + Test):**

```bash
make ci
```

---

## Project Structure

```
stocksentinel/
├── src/                      # Core application code
│   ├── data_collector.py     # News collection (APIs)
│   ├── sentiment_analyzer.py # FinBERT sentiment analysis
│   ├── stock_scorer.py       # Scoring & recommendations
│   ├── utils.py              # Helper functions
│   └── cli.py                # Command-line interface
├── api/
│   └── main.py               # FastAPI backend
├── frontend/
│   └── app.py                # Streamlit interface
├── tests/                    # Unit tests
├── data/                     # Data storage
├── requirements.txt          # Python dependencies
├── setup.py                  # Package configuration
├── Dockerfile                # Docker configuration
├── docker-compose.yml        # Multi-container setup
├── Makefile                  # Automation commands
└── README.md
```

---

## Dependencies

### Core Libraries
- `requests` - HTTP requests for API calls
- `pandas`, `numpy` - Data processing
- `python-dotenv` - Environment variable management

### Financial Data
- `newsapi-python` - NewsAPI integration
- `yfinance` - Yahoo Finance stock data

### NLP & AI
- `transformers` - Hugging Face library for FinBERT
- `torch` - PyTorch backend
- `sentencepiece` - Tokenization

### Backend
- `fastapi` - REST API framework
- `uvicorn` - ASGI server
- `pydantic` - Data validation

### Frontend
- `streamlit` - Web interface
- `plotly` - Interactive charts

### Development
- `pytest` - Testing framework
- `black` - Code formatter
- `flake8` - Linter

---

## How It Works

### 1. Data Collection
Fetches financial news from multiple sources and deduplicates articles.

### 2. Sentiment Analysis
Uses **FinBERT** (financial domain-specific BERT model) to classify sentiment as:
- Positive (bullish)
- Negative (bearish)
- Neutral

### 3. Stock Scoring Algorithm

**Weighted Components:**
- **Sentiment (70%)**: Mean sentiment across all articles
- **Consistency (15%)**: Agreement between sources (only if ≥10 articles)
- **Volume (10%)**: Penalty for insufficient data, neutral for adequate coverage
- **Recency (5%)**: Minor bonus for recent news

**Score Calculation:**
```
total_score = weighted_sum × amplification_factor (1.5x)
final_score = (total_score + 1) × 50  # Normalized to 0-100
```

### 4. Recommendation Thresholds

| Score Range | Recommendation |
|-------------|----------------|
| ≥ 70        | STRONG BUY     |
| 60-69       | BUY            |
| 55-59       | WEAK BUY       |
| 45-54       | HOLD           |
| 40-44       | WEAK SELL      |
| 30-39       | SELL           |
| < 30        | STRONG SELL    |

---

## Make Commands

```bash
make install        # Install dependencies
make test           # Run tests
make ci             # Full CI pipeline (format + lint + test)
make run-api        # Start FastAPI server
make run-frontend   # Start Streamlit app
make docker-build   # Build Docker image
make docker-compose # Run Docker container with compose
make help           # Show all commands
```

---

## Limitations

- **API Rate Limits**: Free tier APIs have request limitations
- **Language**: English news only
- **Market Coverage**: Major exchanges only (yfinance validation)
- **Real-time Data**: News updates with API refresh delays

---

## Future Improvements

- Social media sentiment (Twitter/Reddit)
- Historical backtesting against actual stock performance
- Multi-language support
- Email alerts for significant sentiment changes
- Portfolio tracking

---

## External Resources & Credits

### Models & Libraries
- **FinBERT**: [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert) - Financial sentiment analysis model
- **Transformers**: Hugging Face library
- **Streamlit**: Web framework
- **FastAPI**: API framework

### Data Sources
- **NewsAPI**: [newsapi.org](https://newsapi.org/)
- **Finnhub**: [finnhub.io](https://finnhub.io/)
- **Yahoo Finance**: Via `yfinance` library

### Development
- **Black**: Code formatting
- **Flake8**: Code linting
- **Pytest**: Testing framework

---

## Author

**Tom Schillerwein**  
Course Project - Introduction to AI  
HSLU - Hochschule Luzern  
December 2025

---

## License

This project is created for educational purposes as part of a university course.
