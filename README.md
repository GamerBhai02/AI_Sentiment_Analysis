# 🎭 Dynamic AI Sentiment Analysis Dashboard

A real-time sentiment analysis system that scrapes social media posts and analyzes sentiment using transformer models (DistilBERT).

## 🌟 Features

- **Real-time Data Collection**: Scrape ~5000 posts from Twitter/X and Reddit
- **Advanced Sentiment Analysis**: DistilBERT transformer model for high accuracy
- **Interactive Visualizations**: 
  - Sentiment distribution pie charts
  - Trend analysis over time
  - Platform comparison
  - Word clouds by sentiment
- **Comprehensive Dashboard**: Built with Streamlit for easy interaction
- **Data Export**: Download results and analysis reports
- **Responsive Design**: Works on desktop and mobile devices

## 🚀 Quick Start

### Installation
```bash
pip install -r requirements.txt
streamlit run streamlit_sentiment_dashboard.py
```

### Usage
1. Enter a keyword/topic to analyze
2. Select data source (Twitter, Reddit, or both)  
3. Set number of posts to scrape
4. View real-time sentiment analysis results
5. Export data and insights

## 📊 Screenshots

### Dashboard Overview
- Live sentiment metrics
- Interactive charts and visualizations
- Real-time data processing

### Sentiment Analysis
- Positive/Negative/Neutral classification
- Confidence scores for each prediction
- Trend analysis over time

### Word Clouds
- Visual representation of frequent terms
- Separate clouds for each sentiment category
- Color-coded by sentiment type

## 🛠️ Technical Stack

- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly, Matplotlib, Seaborn, WordCloud
- **ML Model**: DistilBERT (Hugging Face Transformers)
- **Data Collection**: snscrape, PRAW
- **Text Processing**: NLTK

## 📈 Model Performance

- **Accuracy**: 90%+ on sentiment classification
- **Speed**: Real-time analysis of 1000+ posts
- **Languages**: Optimized for English text
- **Platforms**: Twitter/X and Reddit supported

## 🔧 Configuration

### Environment Variables
```
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USER_AGENT=your_user_agent
```

### Model Settings
- Batch size: Configurable for performance tuning
- Confidence threshold: Adjustable sentiment classification
- Data limits: Set maximum posts per scraping session

## 📁 Project Structure

```
sentiment-analysis-dashboard/
├── streamlit_sentiment_dashboard.py    # Main application
├── data_collection/                    # Data scraping modules
├── sentiment_analysis/                 # ML models and analysis
├── preprocessing/                      # Text cleaning utilities
├── requirements.txt                    # Dependencies
├── sample_data/                       # Sample datasets
└── docs/                             # Documentation
```

## 🚢 Deployment

### Streamlit Cloud
1. Push to GitHub
2. Connect to Streamlit Cloud
3. One-click deployment

### HuggingFace Spaces  
1. Upload to HF Spaces
2. Set runtime to Streamlit
3. Automatic deployment

### Docker
```dockerfile
FROM python:3.9
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_sentiment_dashboard.py"]
```

## 📚 Documentation

- [Installation Guide](INSTALLATION_GUIDE.md)
- [API Documentation](docs/api.md)
- [Model Details](docs/models.md)
- [Contributing Guidelines](CONTRIBUTING.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes  
4. Submit a pull request

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Hugging Face for transformer models
- Streamlit for the amazing dashboard framework
- The open source community for tools and libraries

## 📞 Contact

- GitHub: [Your GitHub Profile]
- LinkedIn: [Your LinkedIn Profile]  
- Email: [Your Email]

---

⭐ **Star this repository if you found it helpful!**
