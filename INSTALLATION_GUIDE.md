# 🚀 Dynamic AI Sentiment Analysis Dashboard - Installation Guide

## Prerequisites
- Python 3.8 or higher
- pip package manager
- Git (for cloning repositories)

## Quick Start

### 1. Clone or Download the Project
```bash
git clone <your-repository-url>
cd sentiment-analysis-dashboard
```

### 2. Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv sentiment_env

# Activate virtual environment
# On Windows:
sentiment_env\Scripts\activate
# On macOS/Linux:
source sentiment_env/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download NLTK Data
```python
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

### 5. Set Up API Credentials (For Live Scraping)

#### Twitter/X Setup (using snscrape - no API key needed):
- snscrape works without API credentials
- For rate limiting, consider using rotating proxies

#### Reddit Setup (using PRAW):
1. Go to https://www.reddit.com/prefs/apps
2. Click "Create App" or "Create Another App"
3. Fill in the form:
   - Name: Your app name
   - App type: Select "script"
   - Redirect URI: http://localhost:8000
4. Note your client ID and client secret

Create a `.env` file in the project root:
```
REDDIT_CLIENT_ID=your_client_id_here
REDDIT_CLIENT_SECRET=your_client_secret_here
REDDIT_USER_AGENT=sentiment_analyzer_v1.0
```

### 6. Run the Dashboard
```bash
streamlit run streamlit_sentiment_dashboard.py
```

The dashboard will open in your default browser at `http://localhost:8501`

## Project Structure
```
sentiment-analysis-dashboard/
├── streamlit_sentiment_dashboard.py    # Main dashboard app
├── requirements.txt                    # Python dependencies
├── data_collection/
│   ├── twitter_scraper.py             # Twitter data collection
│   └── reddit_scraper.py              # Reddit data collection
├── preprocessing/
│   └── text_preprocessor.py           # Text cleaning and preprocessing
├── sentiment_analysis/
│   └── distilbert_analyzer.py         # DistilBERT sentiment analysis
├── sample_data/
│   └── sample_sentiment_data.csv      # Sample dataset for testing
└── .env                               # Environment variables (create this)
```

## Usage Guide

### 1. Using Sample Data
- Select "Use Sample Data" in the sidebar
- Explore the dashboard features with pre-loaded data

### 2. Scraping New Data
- Select "Scrape New Data" in the sidebar
- Enter a keyword/topic to analyze
- Choose platform (Twitter, Reddit, or Both)
- Set number of posts to scrape
- Click "Scrape & Analyze"

### 3. Dashboard Features
- **Sentiment Distribution**: Pie chart showing positive/negative/neutral breakdown
- **Sentiment Over Time**: Line chart showing sentiment trends
- **Platform Comparison**: Compare sentiment across Twitter and Reddit
- **Word Clouds**: Visual representation of frequent words by sentiment
- **Sample Posts**: View actual posts categorized by sentiment
- **Data Export**: Download results as CSV files

## Customization

### Adding New Data Sources
1. Create a new scraper module in `data_collection/`
2. Implement data collection function
3. Update the dashboard to include new source option

### Using Different Models
1. Modify `sentiment_analysis/distilbert_analyzer.py`
2. Replace DistilBERT with your preferred model:
   - BERT
   - RoBERTa  
   - VADER
   - Custom fine-tuned models

### Styling and Layout
- Modify the CSS in `streamlit_sentiment_dashboard.py`
- Adjust colors, fonts, and layout in the Streamlit configuration
- Add custom components using Streamlit components

## Deployment Options

### 1. Streamlit Cloud (Free)
1. Push code to GitHub
2. Go to https://share.streamlit.io/
3. Connect your GitHub repository
4. Deploy with one click

### 2. Heroku
1. Create `Procfile`: `web: streamlit run streamlit_sentiment_dashboard.py --server.port=$PORT --server.address=0.0.0.0`
2. Set up Heroku app and deploy

### 3. HuggingFace Spaces
1. Create new Space on HuggingFace
2. Upload files and requirements
3. Set Space type to "Streamlit"

## Troubleshooting

### Common Issues:

1. **Import Errors**: Ensure all dependencies are installed
2. **NLTK Data Missing**: Run the NLTK download commands
3. **Memory Issues**: Reduce batch size or number of posts
4. **API Rate Limits**: Implement delays between requests
5. **Model Loading**: Ensure sufficient disk space for transformer models

### Performance Tips:
- Use smaller datasets for testing
- Implement caching with `@st.cache_data`
- Consider using GPU for faster inference
- Batch process large datasets

## Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License
This project is open source under the MIT License.

## Support
- Create issues on GitHub for bug reports
- Check documentation for detailed guides
- Join the community discussions
