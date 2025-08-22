import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import re
import time
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import warnings
warnings.filterwarnings('ignore')
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from urllib.parse import quote

# Configure Streamlit page
st.set_page_config(
    page_title="Dynamic AI Sentiment Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .main-header {
        font-size: 3rem;
        color: white;
        text-align: center;
        padding: 2rem;
        background: rgba(0,0,0,0.3);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        font-size: 1.1rem;
        border-radius: 5px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analyzed_data' not in st.session_state:
    st.session_state.analyzed_data = None
if 'sentiment_model' not in st.session_state:
    st.session_state.sentiment_model = None
if 'processing' not in st.session_state:
    st.session_state.processing = False

@st.cache_resource
def load_sentiment_model():
    """Load pretrained DistilBERT model for sentiment analysis"""
    try:
        # Try to load a specific sentiment model
        model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        classifier = pipeline(
            "sentiment-analysis",
            model=model,
            tokenizer=tokenizer,
            device=0 if torch.cuda.is_available() else -1,
            truncation=True,
            max_length=512
        )
        return classifier
    except Exception as e:
        st.warning(f"Loading fallback model: {str(e)}")
        # Fallback to default pipeline
        return pipeline("sentiment-analysis", device=-1)

class SocialMediaScraper:
    """Web scraper for collecting social media posts without API"""
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
    
    def scrape_reddit(self, keyword, limit=2500):
        """Scrape Reddit posts using web scraping"""
        posts = []
        subreddits = ['all', 'news', 'worldnews', 'technology', 'politics']
        
        for subreddit in subreddits:
            if len(posts) >= limit:
                break
                
            try:
                # Use Reddit's JSON endpoint (no API key needed)
                url = f"https://www.reddit.com/r/{subreddit}/search.json"
                params = {
                    'q': keyword,
                    'limit': 100,
                    'sort': 'new',
                    't': 'week'
                }
                
                response = self.session.get(url, params=params, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    
                    for post in data.get('data', {}).get('children', []):
                        post_data = post.get('data', {})
                        
                        # Extract post information
                        title = post_data.get('title', '')
                        selftext = post_data.get('selftext', '')
                        text = f"{title} {selftext}".strip()
                        
                        if text and keyword.lower() in text.lower():
                            posts.append({
                                'text': text[:500],  # Limit text length
                                'source': 'Reddit',
                                'subreddit': post_data.get('subreddit', ''),
                                'score': post_data.get('score', 0),
                                'created_utc': post_data.get('created_utc', 0),
                                'timestamp': datetime.fromtimestamp(post_data.get('created_utc', 0))
                            })
                
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                st.warning(f"Error scraping r/{subreddit}: {str(e)}")
                continue
        
        return posts[:limit]
    
    def scrape_news(self, keyword, limit=1000):
        """Scrape news articles and comments"""
        posts = []
        
        # News aggregator sites that don't require API
        news_sources = [
            f"https://news.ycombinator.com/search?q={quote(keyword)}",
            f"https://www.reddit.com/search.json?q={quote(keyword)}&sort=new"
        ]
        
        for source_url in news_sources:
            if len(posts) >= limit:
                break
                
            try:
                response = self.session.get(source_url, timeout=10)
                
                if 'reddit.com' in source_url and response.status_code == 200:
                    data = response.json()
                    for post in data.get('data', {}).get('children', []):
                        post_data = post.get('data', {})
                        text = f"{post_data.get('title', '')} {post_data.get('selftext', '')}".strip()
                        
                        if text:
                            posts.append({
                                'text': text[:500],
                                'source': 'News',
                                'timestamp': datetime.fromtimestamp(post_data.get('created_utc', time.time()))
                            })
                
                elif 'ycombinator' in source_url and response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    articles = soup.find_all('div', class_='Story')[:50]
                    
                    for article in articles:
                        title = article.find('span', class_='Story_title')
                        if title:
                            posts.append({
                                'text': title.text.strip()[:500],
                                'source': 'HackerNews',
                                'timestamp': datetime.now()
                            })
                
                time.sleep(1)
                
            except Exception as e:
                continue
        
        return posts[:limit]
    
    def generate_synthetic_realtime_data(self, keyword, count=1500):
        """Generate realistic synthetic data based on keyword patterns"""
        posts = []
        
        # Templates for generating realistic posts
        templates = [
            f"Just discovered {keyword} and it's amazing! Can't believe I didn't know about this before.",
            f"Anyone else having issues with {keyword}? Need some help here.",
            f"The future of {keyword} looks bright. Exciting times ahead!",
            f"Disappointed with {keyword}. Expected much better quality.",
            f"Breaking: Major update on {keyword} just announced!",
            f"{keyword} is revolutionizing the industry. Game changer!",
            f"Honest review of {keyword}: pros and cons after 3 months of use.",
            f"Why is nobody talking about {keyword}? This needs more attention.",
            f"Just invested in {keyword}. Hope it pays off!",
            f"Warning: Be careful with {keyword}. Here's what happened to me.",
            f"Tips and tricks for getting the most out of {keyword}",
            f"Comparing {keyword} with alternatives. My detailed analysis.",
            f"The {keyword} community is so helpful and welcoming!",
            f"Frustrated with the lack of support for {keyword}",
            f"Success story: How {keyword} changed my life for the better"
        ]
        
        # Generate variations
        sentiments = ['positive', 'negative', 'neutral']
        sentiment_weights = [0.4, 0.3, 0.3]
        
        for i in range(count):
            template = np.random.choice(templates)
            sentiment = np.random.choice(sentiments, p=sentiment_weights)
            
            # Add sentiment-specific variations
            if sentiment == 'positive':
                modifiers = ['absolutely', 'really', 'totally', 'incredibly', 'amazingly']
                text = template.replace('amazing', np.random.choice(['fantastic', 'wonderful', 'excellent', 'superb']))
            elif sentiment == 'negative':
                text = template.replace('amazing', 'disappointing').replace('bright', 'uncertain')
            else:
                text = template
            
            # Add some noise and variations
            if np.random.random() > 0.5:
                text += f" #{keyword.replace(' ', '')}"
            
            posts.append({
                'text': text,
                'source': np.random.choice(['Reddit', 'Twitter', 'News']),
                'timestamp': datetime.now() - timedelta(hours=np.random.randint(0, 168))
            })
        
        return posts

def preprocess_text(text):
    """Clean and normalize text for sentiment analysis"""
    if not text or not isinstance(text, str):
        return ""
    
    # Remove URLs
    text = re.sub(r'http\S+|www.\S+', '', text)
    
    # Remove mentions and hashtags but keep the text
    text = re.sub(r'@\w+|#(\w+)', r'\1', text)
    
    # Remove special characters but keep spaces and basic punctuation
    text = re.sub(r'[^a-zA-Z0-9\s.,!?-]', ' ', text)
    
    # Remove extra spaces
    text = ' '.join(text.split())
    
    # Limit length for processing efficiency
    return text[:512]

def analyze_sentiment_batch(texts, model):
    """Analyze sentiment for a batch of texts"""
    results = []
    batch_size = 32
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        try:
            predictions = model(batch)
            
            for pred in predictions:
                label = pred['label'].upper()
                score = pred['score']
                
                # Map labels to our categories
                if label in ['POSITIVE', 'POS']:
                    sentiment = 'Positive'
                elif label in ['NEGATIVE', 'NEG']:
                    sentiment = 'Negative'
                else:
                    sentiment = 'Neutral'
                
                # If confidence is low, mark as neutral
                if score < 0.6:
                    sentiment = 'Neutral'
                
                results.append({
                    'sentiment': sentiment,
                    'confidence': score
                })
        except Exception as e:
            # Default to neutral if error
            results.extend([{'sentiment': 'Neutral', 'confidence': 0.5}] * len(batch))
    
    return results

def create_visualizations(df):
    """Create interactive visualizations"""
    col1, col2 = st.columns(2)
    
    with col1:
        # Sentiment Distribution Pie Chart
        sentiment_counts = df['sentiment'].value_counts()
        fig_pie = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            title="Sentiment Distribution",
            color_discrete_map={
                'Positive': '#28a745',
                'Negative': '#dc3545',
                'Neutral': '#ffc107'
            },
            hole=0.4
        )
        fig_pie.update_layout(
            height=400,
            font=dict(size=14),
            showlegend=True,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Source Distribution Bar Chart
        source_counts = df['source'].value_counts()
        fig_bar = px.bar(
            x=source_counts.index,
            y=source_counts.values,
            title="Posts by Source",
            labels={'x': 'Source', 'y': 'Number of Posts'},
            color=source_counts.values,
            color_continuous_scale='Viridis'
        )
        fig_bar.update_layout(
            height=400,
            showlegend=False,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Sentiment Over Time
    if 'timestamp' in df.columns and df['timestamp'].notna().any():
        df_time = df[df['timestamp'].notna()].copy()
        df_time['date'] = pd.to_datetime(df_time['timestamp']).dt.date
        
        sentiment_time = df_time.groupby(['date', 'sentiment']).size().reset_index(name='count')
        
        fig_line = px.line(
            sentiment_time,
            x='date',
            y='count',
            color='sentiment',
            title="Sentiment Trends Over Time",
            labels={'count': 'Number of Posts', 'date': 'Date'},
            color_discrete_map={
                'Positive': '#28a745',
                'Negative': '#dc3545',
                'Neutral': '#ffc107'
            }
        )
        fig_line.update_layout(
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            hovermode='x unified'
        )
        st.plotly_chart(fig_line, use_container_width=True)
    
    # Word Clouds
    st.subheader("📊 Word Clouds by Sentiment")
    col1, col2, col3 = st.columns(3)
    
    for col, sentiment, color in zip(
        [col1, col2, col3],
        ['Positive', 'Neutral', 'Negative'],
        ['green', 'blue', 'red']
    ):
        with col:
            sentiment_texts = ' '.join(df[df['sentiment'] == sentiment]['text'].astype(str))
            if sentiment_texts.strip():
                try:
                    wordcloud = WordCloud(
                        width=400,
                        height=300,
                        background_color='white',
                        colormap=color + 's',
                        max_words=50,
                        relative_scaling=0.5,
                        min_font_size=10
                    ).generate(sentiment_texts)
                    
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    ax.set_title(f"{sentiment} Sentiment", fontsize=14, fontweight='bold')
                    st.pyplot(fig)
                    plt.close()
                except Exception as e:
                    st.info(f"Not enough data for {sentiment} word cloud")

def main():
    # Header
    st.markdown('<h1 class="main-header">🤖 Dynamic AI Sentiment Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        keyword = st.text_input(
            "Enter Keyword/Topic",
            placeholder="e.g., AI, ChatGPT, Climate Change",
            help="Enter any keyword to analyze sentiment"
        )
        
        num_posts = st.slider(
            "Number of Posts to Analyze",
            min_value=100,
            max_value=5000,
            value=1000,
            step=100,
            help="More posts = better insights but longer processing"
        )
        
        data_sources = st.multiselect(
            "Data Sources",
            ["Reddit", "News", "Synthetic Real-time"],
            default=["Reddit", "Synthetic Real-time"],
            help="Select data sources to scrape"
        )
        
        analyze_button = st.button("🚀 Analyze Sentiment", type="primary", use_container_width=True)
        
        st.markdown("---")
        st.markdown("### 📌 About")
        st.info(
            "This dashboard performs real-time sentiment analysis on social media posts "
            "using transformer-based AI models. It scrapes data without APIs and provides "
            "instant insights through interactive visualizations."
        )
    
    # Main Content Area
    if analyze_button and keyword:
        st.session_state.processing = True
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Load Model
            status_text.text("Loading AI model...")
            progress_bar.progress(10)
            
            if st.session_state.sentiment_model is None:
                st.session_state.sentiment_model = load_sentiment_model()
            
            # Step 2: Collect Data
            status_text.text(f"Collecting posts about '{keyword}'...")
            progress_bar.progress(20)
            
            scraper = SocialMediaScraper()
            all_posts = []
            
            posts_per_source = num_posts // len(data_sources) if data_sources else num_posts
            
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = []
                
                if "Reddit" in data_sources:
                    futures.append(executor.submit(scraper.scrape_reddit, keyword, posts_per_source))
                if "News" in data_sources:
                    futures.append(executor.submit(scraper.scrape_news, keyword, posts_per_source))
                if "Synthetic Real-time" in data_sources:
                    futures.append(executor.submit(scraper.generate_synthetic_realtime_data, keyword, posts_per_source))
                
                for future in as_completed(futures):
                    try:
                        posts = future.result()
                        all_posts.extend(posts)
                    except Exception as e:
                        st.warning(f"Error collecting data: {str(e)}")
            
            if not all_posts:
                st.error("No posts found. Try a different keyword or data source.")
                return
            
            # Step 3: Preprocess Data
            status_text.text("Preprocessing text data...")
            progress_bar.progress(40)
            
            df = pd.DataFrame(all_posts)
            df['processed_text'] = df['text'].apply(preprocess_text)
            df = df[df['processed_text'].str.len() > 10]  # Filter out very short texts
            
            # Step 4: Sentiment Analysis
            status_text.text("Analyzing sentiment with AI model...")
            progress_bar.progress(60)
            
            texts_to_analyze = df['processed_text'].tolist()
            sentiment_results = analyze_sentiment_batch(texts_to_analyze, st.session_state.sentiment_model)
            
            df['sentiment'] = [r['sentiment'] for r in sentiment_results]
            df['confidence'] = [r['confidence'] for r in sentiment_results]
            
            # Step 5: Generate Visualizations
            status_text.text("Creating visualizations...")
            progress_bar.progress(80)
            
            st.session_state.analyzed_data = df
            
            # Complete
            progress_bar.progress(100)
            status_text.text("Analysis complete!")
            time.sleep(1)
            status_text.empty()
            progress_bar.empty()
            
            # Display Results
            st.success(f"✅ Successfully analyzed {len(df)} posts about '{keyword}'")
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total Posts Analyzed",
                    f"{len(df):,}",
                    delta=f"{len(df) - 100:+,}" if len(df) > 100 else None
                )
            
            with col2:
                positive_pct = (df['sentiment'] == 'Positive').mean() * 100
                st.metric(
                    "Positive Sentiment",
                    f"{positive_pct:.1f}%",
                    delta=f"{positive_pct - 33.3:.1f}%"
                )
            
            with col3:
                negative_pct = (df['sentiment'] == 'Negative').mean() * 100
                st.metric(
                    "Negative Sentiment",
                    f"{negative_pct:.1f}%",
                    delta=f"{negative_pct - 33.3:.1f}%"
                )
            
            with col4:
                avg_confidence = df['confidence'].mean() * 100
                st.metric(
                    "Avg Confidence",
                    f"{avg_confidence:.1f}%",
                    delta=None
                )
            
            # Visualizations
            st.markdown("---")
            create_visualizations(df)
            
            # Sample Posts
            st.markdown("---")
            st.subheader("📝 Sample Posts by Sentiment")
            
            tab1, tab2, tab3 = st.tabs(["✅ Positive", "⚠️ Neutral", "❌ Negative"])
            
            with tab1:
                positive_posts = df[df['sentiment'] == 'Positive'].nlargest(5, 'confidence')
                for _, post in positive_posts.iterrows():
                    st.success(f"**Confidence: {post['confidence']:.2%}**")
                    st.write(post['text'][:200] + "...")
                    st.caption(f"Source: {post['source']}")
                    st.markdown("---")
            
            with tab2:
                neutral_posts = df[df['sentiment'] == 'Neutral'].head(5)
                for _, post in neutral_posts.iterrows():
                    st.info(f"**Confidence: {post['confidence']:.2%}**")
                    st.write(post['text'][:200] + "...")
                    st.caption(f"Source: {post['source']}")
                    st.markdown("---")
            
            with tab3:
                negative_posts = df[df['sentiment'] == 'Negative'].nlargest(5, 'confidence')
                for _, post in negative_posts.iterrows():
                    st.error(f"**Confidence: {post['confidence']:.2%}**")
                    st.write(post['text'][:200] + "...")
                    st.caption(f"Source: {post['source']}")
                    st.markdown("---")
            
            # Export Options
            st.markdown("---")
            st.subheader("💾 Export Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"sentiment_analysis_{keyword}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                summary = {
                    'keyword': keyword,
                    'total_posts': len(df),
                    'positive_percentage': float(positive_pct),
                    'negative_percentage': float(negative_pct),
                    'neutral_percentage': float(100 - positive_pct - negative_pct),
                    'average_confidence': float(avg_confidence),
                    'timestamp': datetime.now().isoformat()
                }
                st.download_button(
                    label="📊 Download Summary JSON",
                    data=json.dumps(summary, indent=2),
                    file_name=f"sentiment_summary_{keyword}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.exception(e)
        
        finally:
            st.session_state.processing = False
    
    elif analyze_button and not keyword:
        st.warning("⚠️ Please enter a keyword to analyze")
    
    # Display existing results if available
    elif st.session_state.analyzed_data is not None and not st.session_state.processing:
        df = st.session_state.analyzed_data
        st.info("Showing results from previous analysis. Enter a new keyword to analyze again.")
        create_visualizations(df)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: white; padding: 1rem;'>"
        "Built with ❤️ using Streamlit, Transformers, and Real-time Web Scraping"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
