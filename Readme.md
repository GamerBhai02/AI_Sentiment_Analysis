# Dynamic Sentiment Analysis Dashboard

A real-time sentiment analysis application that scrapes social media posts and analyzes their sentiment using a Transformer-based AI model.

## Features

- **Real-time Data Collection**: Scrapes Twitter posts without API authentication
- **Text Preprocessing**: Cleans and normalizes text data
- **Sentiment Analysis**: Uses DistilBERT model to classify posts as Positive, Neutral, or Negative
- **Interactive Visualizations**: 
  - Sentiment distribution pie chart
  - Sentiment trend over time
  - Word clouds for each sentiment category
  - Sample posts with sentiment highlighting

## Installation

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the application: `streamlit run app.py`

## Usage

1. Enter a keyword or hashtag in the sidebar
2. Adjust the number of posts to analyze (100-2000)
3. Click "Analyze Sentiment"
4. Explore the visualizations and insights

## Technical Details

- **Data Collection**: Uses snscrape to fetch tweets without API keys
- **Preprocessing**: Removes URLs, mentions, hashtags, and special characters
- **Model**: DistilBERT-base-uncased-finetuned-sst-2-english from Hugging Face Transformers
- **Dashboard**: Built with Streamlit for interactive visualization

## How to Run the Application

1. Create a new directory and save all the files as described above
2. Install the required packages: `pip install -r requirements.txt`
3. Run the Streamlit application: `streamlit run app.py`
4. Open your browser to the provided local URL (usually http://localhost:8501)
