# 🤖 Dynamic AI Sentiment Analysis Dashboard

A powerful, real-time sentiment analysis system that collects social media posts without APIs, analyzes sentiment using state-of-the-art transformer models, and provides interactive visualizations through a modern web dashboard.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)
![Transformers](https://img.shields.io/badge/transformers-4.35+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 🌟 Features

### Core Functionality
- **Real-time Data Collection**: Scrapes ~5000 posts from multiple sources without requiring API keys
- **AI-Powered Analysis**: Uses DistilBERT transformer model for accurate sentiment classification
- **Interactive Dashboard**: Built with Streamlit for real-time analysis and visualization
- **Multi-Source Support**: Collects data from Reddit, news sites, and generates synthetic real-time data
- **No API Required**: Pure web scraping approach for maximum accessibility

### Visualization Components
- 📊 **Sentiment Distribution Pie Chart**: Visual breakdown of positive, negative, and neutral sentiments
- 📈 **Trend Analysis**: Sentiment changes over time with interactive line charts
- ☁️ **Word Clouds**: Frequency visualization for each sentiment category
- 📍 **Source Distribution**: Analysis of data sources and their contributions
- 📝 **Sample Posts**: Display of representative posts for each sentiment

### Technical Highlights
- **Batch Processing**: Efficient processing of thousands of posts
- **Concurrent Scraping**: Multi-threaded data collection for speed
- **Smart Preprocessing**: Advanced text cleaning and normalization
- **Confidence Scoring**: Each prediction includes confidence metrics
- **Export Functionality**: Download results as CSV or JSON

## 🚀 Quick Start

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/sentiment-analysis-dashboard.git
cd sentiment-analysis-dashboard
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Running the Application

```bash
streamlit run app.py
```

The dashboard will open automatically in your browser at `http://localhost:8501`

## 📖 Usage Guide

### Basic Usage

1. **Enter a Keyword**: Type any topic, brand, or keyword you want to analyze
2. **Configure Settings**: 
   - Select number of posts (100-5000)
   - Choose data sources (Reddit, News, Synthetic)
3. **Click Analyze**: The system will collect and analyze posts in real-time
4. **Explore Results**: Interactive charts, word clouds, and sample posts
5. **Export Data**: Download results as CSV or JSON for further analysis

### Advanced Features

#### Multi-Source Analysis
- **Reddit**: Searches across multiple subreddits for comprehensive coverage
- **News**: Aggregates from news sites and discussion forums
- **Synthetic Real-time**: Generates realistic data for testing and demonstration

#### Sentiment Categories
- **Positive**: Posts with optimistic, supportive, or enthusiastic sentiment
- **Negative**: Posts with critical, disappointed, or frustrated sentiment
- **Neutral**: Factual, balanced, or unclear sentiment posts

##
