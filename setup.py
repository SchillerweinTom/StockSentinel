from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="StockSentinel",
    version="0.1.0",
    author="Tom Schillerwein",
    description="Financial News Sentiment Analyzer for Stock Market Analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SchillerweinTom/StockSentinel",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "requests>=2.31.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "python-dotenv>=1.0.0",
        "newsapi-python>=0.2.7",
        "yfinance>=0.2.28",
        "transformers>=4.30.0",
        "torch>=2.0.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0",
        "streamlit>=1.25.0",
        "plotly>=5.15.0",
    ],
    entry_points={
        "console_scripts": [
            "stocksentinel=src.cli:main",
        ],
    },
)