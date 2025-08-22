
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import re
import time
from collections import Counter

# Import custom modules (these would be separate files in actual implementation)
# from data_collection.twitter_scraper import scrape_tweets
# from data_collection.reddit_scraper import RedditScraper
# from preprocessing.text_preprocessor import TextPreprocessor
# from sentiment_analysis.distilbert_analyzer import DistilBERTSentimentAnalyzer

# Page configuration
st.set_page_config(
    page_title="AI Sentiment Analysis Dashboard",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .sentiment-positive {
        color: #28a745;
        font-weight: bold;
    }
    .sentiment-negative {
        color: #dc3545;
        font-weight: bold;
    }
    .sentiment-neutral {
        color: #6c757d;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.title("🎭 Sentiment Analysis Dashboard")
st.sidebar.markdown("---")

# Mock functions for demonstration (replace with actual implementations)
@st.cache_data
def load_sample_data():
    """Load sample data for demonstration"""
    np.random.seed(42)

    sample_texts = [
        "I love this new AI technology! It's revolutionary!",
        "This artificial intelligence system is terrible and buggy.",
        "The machine learning algorithm works as expected.",
        "Amazing breakthrough in deep learning research!",
        "Not impressed with this AI chatbot at all.",
        "Neutral opinion about the new neural network.",
        "Fantastic results from the computer vision model!",
        "The natural language processing is disappointing.",
        "Standard performance from this recommendation system.",
        "Incredible advances in robotics and automation!",
        "AI is transforming healthcare in amazing ways",
        "Worried about AI replacing human jobs",
        "Machine learning models need better interpretability",
        "Excited about the future of artificial intelligence",
        "AI chatbots still have many limitations"
    ]

    # Generate more sample data
    extended_texts = sample_texts * 20

    data = {
        'text': extended_texts,
        'timestamp': pd.date_range(start='2025-01-01', periods=len(extended_texts), freq='2h'),
        'source': np.random.choice(['Twitter', 'Reddit'], size=len(extended_texts)),
        'engagement': np.random.randint(1, 1000, size=len(extended_texts))
    }

    df = pd.DataFrame(data)

    # Add mock sentiment analysis results
    sentiments = []
    scores = []

    for text in df['text']:
        if any(word in text.lower() for word in ['love', 'amazing', 'fantastic', 'incredible', 'excited']):
            sentiment = 'Positive'
            score = np.random.uniform(0.8, 0.99)
        elif any(word in text.lower() for word in ['terrible', 'worst', 'disappointing', 'worried', 'limitations']):
            sentiment = 'Negative' 
            score = np.random.uniform(0.8, 0.99)
        else:
            sentiment = 'Neutral'
            score = np.random.uniform(0.5, 0.8)

        sentiments.append(sentiment)
        scores.append(score)

    df['sentiment'] = sentiments
    df['sentiment_score'] = scores

    return df

def mock_scrape_data(keyword, source, num_posts):
    """Mock data scraping function"""
    time.sleep(2)  # Simulate API delay

    # Generate mock data based on keyword
    positive_words = ['great', 'amazing', 'love', 'excellent', 'fantastic']
    negative_words = ['terrible', 'hate', 'awful', 'disappointing', 'worst']
    neutral_words = ['okay', 'standard', 'normal', 'regular', 'typical']

    texts = []
    for i in range(num_posts):
        sentiment_type = np.random.choice(['positive', 'negative', 'neutral'], p=[0.4, 0.3, 0.3])

        if sentiment_type == 'positive':
            word = np.random.choice(positive_words)
            text = f"This {keyword} is {word}! Really impressed with the results."
        elif sentiment_type == 'negative':
            word = np.random.choice(negative_words)
            text = f"The {keyword} experience was {word}. Not satisfied at all."
        else:
            word = np.random.choice(neutral_words)
            text = f"The {keyword} performance seems {word}. Nothing special to report."

        texts.append(text)

    data = {
        'text': texts,
        'timestamp': pd.date_range(start=datetime.now() - timedelta(days=7), periods=num_posts, freq='1h'),
        'source': [source] * num_posts,
        'engagement': np.random.randint(1, 500, size=num_posts)
    }

    df = pd.DataFrame(data)

    # Add sentiment analysis
    sentiments = []
    scores = []

    for text in df['text']:
        if any(word in text.lower() for word in positive_words):
            sentiment = 'Positive'
            score = np.random.uniform(0.8, 0.99)
        elif any(word in text.lower() for word in negative_words):
            sentiment = 'Negative' 
            score = np.random.uniform(0.8, 0.99)
        else:
            sentiment = 'Neutral'
            score = np.random.uniform(0.5, 0.8)

        sentiments.append(sentiment)
        scores.append(score)

    df['sentiment'] = sentiments
    df['sentiment_score'] = scores

    return df

def create_wordcloud(texts, sentiment_type):
    """Create word cloud for specific sentiment"""
    text = ' '.join(texts)

    # Color schemes for different sentiments
    if sentiment_type == 'Positive':
        colormap = 'Greens'
    elif sentiment_type == 'Negative':
        colormap = 'Reds'
    else:
        colormap = 'Blues'

    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='white',
        colormap=colormap,
        max_words=100,
        relative_scaling=0.5,
        random_state=42
    ).generate(text)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(f'{sentiment_type} Sentiment Word Cloud', fontsize=16, fontweight='bold')

    return fig

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">🎭 AI Sentiment Analysis Dashboard</h1>', unsafe_allow_html=True)

    # Sidebar controls
    st.sidebar.header("📊 Analysis Configuration")

    # Data source selection
    data_source = st.sidebar.radio(
        "Select Data Source:",
        ["Use Sample Data", "Scrape New Data"]
    )

    if data_source == "Scrape New Data":
        # Keyword input
        keyword = st.sidebar.text_input(
            "Enter keyword/hashtag:",
            value="artificial intelligence",
            help="Enter the topic you want to analyze"
        )

        # Platform selection
        platform = st.sidebar.selectbox(
            "Select Platform:",
            ["Twitter", "Reddit", "Both"]
        )

        # Number of posts
        num_posts = st.sidebar.slider(
            "Number of posts to analyze:",
            min_value=100,
            max_value=5000,
            value=1000,
            step=100
        )

        # Scrape button
        if st.sidebar.button("🔍 Scrape & Analyze", type="primary"):
            with st.spinner("Scraping data and analyzing sentiment..."):
                if platform == "Both":
                    twitter_data = mock_scrape_data(keyword, "Twitter", num_posts//2)
                    reddit_data = mock_scrape_data(keyword, "Reddit", num_posts//2)
                    df = pd.concat([twitter_data, reddit_data], ignore_index=True)
                else:
                    df = mock_scrape_data(keyword, platform, num_posts)

                st.session_state['data'] = df
                st.session_state['keyword'] = keyword
                st.success(f"Successfully scraped and analyzed {len(df)} posts!")

    # Load data
    if 'data' not in st.session_state:
        df = load_sample_data()
        st.session_state['data'] = df
        st.session_state['keyword'] = "AI Technology"
    else:
        df = st.session_state['data']

    # Display current analysis info
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Current Analysis:**")
    st.sidebar.markdown(f"Topic: {st.session_state.get('keyword', 'AI Technology')}")
    st.sidebar.markdown(f"Total Posts: {len(df)}")
    st.sidebar.markdown(f"Date Range: {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")

    # Date range filter
    st.sidebar.markdown("---")
    st.sidebar.header("📅 Date Filter")

    date_range = st.sidebar.date_input(
        "Select date range:",
        value=(df['timestamp'].min().date(), df['timestamp'].max().date()),
        min_value=df['timestamp'].min().date(),
        max_value=df['timestamp'].max().date()
    )

    if len(date_range) == 2:
        start_date, end_date = date_range
        df = df[
            (df['timestamp'].dt.date >= start_date) & 
            (df['timestamp'].dt.date <= end_date)
        ]

    # Main dashboard
    if len(df) > 0:
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)

        total_posts = len(df)
        positive_pct = (df['sentiment'] == 'Positive').mean() * 100
        negative_pct = (df['sentiment'] == 'Negative').mean() * 100
        avg_engagement = df['engagement'].mean()

        with col1:
            st.metric(
                label="📊 Total Posts",
                value=f"{total_posts:,}",
                delta=None
            )

        with col2:
            st.metric(
                label="😊 Positive Sentiment",
                value=f"{positive_pct:.1f}%",
                delta=f"{positive_pct-33.3:.1f}%" if positive_pct > 33.3 else f"{positive_pct-33.3:.1f}%"
            )

        with col3:
            st.metric(
                label="😞 Negative Sentiment", 
                value=f"{negative_pct:.1f}%",
                delta=f"{negative_pct-33.3:.1f}%" if negative_pct < 33.3 else f"{negative_pct-33.3:.1f}%"
            )

        with col4:
            st.metric(
                label="📈 Avg Engagement",
                value=f"{avg_engagement:.0f}",
                delta=None
            )

        # Sentiment distribution
        st.markdown("---")
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("📊 Sentiment Distribution")

            sentiment_counts = df['sentiment'].value_counts()

            fig_pie = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                color=sentiment_counts.index,
                color_discrete_map={
                    'Positive': '#28a745',
                    'Negative': '#dc3545', 
                    'Neutral': '#6c757d'
                },
                title="Overall Sentiment Distribution"
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            st.subheader("📈 Sentiment Over Time")

            # Resample by day and calculate sentiment percentages
            df['date'] = df['timestamp'].dt.date
            time_sentiment = df.groupby(['date', 'sentiment']).size().unstack(fill_value=0)
            time_sentiment_pct = time_sentiment.div(time_sentiment.sum(axis=1), axis=0) * 100

            fig_line = go.Figure()

            colors = {'Positive': '#28a745', 'Negative': '#dc3545', 'Neutral': '#6c757d'}

            for sentiment in ['Positive', 'Negative', 'Neutral']:
                if sentiment in time_sentiment_pct.columns:
                    fig_line.add_trace(go.Scatter(
                        x=time_sentiment_pct.index,
                        y=time_sentiment_pct[sentiment],
                        mode='lines+markers',
                        name=sentiment,
                        line=dict(color=colors[sentiment], width=3),
                        marker=dict(size=6)
                    ))

            fig_line.update_layout(
                title="Sentiment Trends Over Time",
                xaxis_title="Date",
                yaxis_title="Percentage (%)",
                hovermode='x unified',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )

            st.plotly_chart(fig_line, use_container_width=True)

        # Platform comparison
        if df['source'].nunique() > 1:
            st.markdown("---")
            st.subheader("🔀 Platform Comparison")

            platform_sentiment = pd.crosstab(df['source'], df['sentiment'], normalize='index') * 100

            fig_bar = px.bar(
                platform_sentiment,
                x=platform_sentiment.index,
                y=['Positive', 'Negative', 'Neutral'],
                title="Sentiment Distribution by Platform",
                color_discrete_map={
                    'Positive': '#28a745',
                    'Negative': '#dc3545',
                    'Neutral': '#6c757d'
                }
            )
            fig_bar.update_layout(
                xaxis_title="Platform",
                yaxis_title="Percentage (%)",
                legend_title="Sentiment"
            )

            st.plotly_chart(fig_bar, use_container_width=True)

        # Word clouds
        st.markdown("---")
        st.subheader("☁️ Word Clouds by Sentiment")

        col1, col2, col3 = st.columns(3)

        sentiments = ['Positive', 'Negative', 'Neutral']
        columns = [col1, col2, col3]

        for sentiment, col in zip(sentiments, columns):
            with col:
                sentiment_texts = df[df['sentiment'] == sentiment]['text'].tolist()
                if sentiment_texts:
                    with st.spinner(f"Generating {sentiment} word cloud..."):
                        fig = create_wordcloud(sentiment_texts, sentiment)
                        st.pyplot(fig)
                        plt.close()
                else:
                    st.info(f"No {sentiment.lower()} posts found")

        # Sample posts
        st.markdown("---")
        st.subheader("📝 Sample Posts by Sentiment")

        tab1, tab2, tab3 = st.tabs(["😊 Positive", "😞 Negative", "😐 Neutral"])

        with tab1:
            positive_posts = df[df['sentiment'] == 'Positive'].nlargest(5, 'sentiment_score')
            for idx, row in positive_posts.iterrows():
                st.markdown(f"""
                <div style="background-color: #d4edda; padding: 10px; border-radius: 5px; margin: 5px 0;">
                    <strong>📅 {row['timestamp'].strftime('%Y-%m-%d %H:%M')}</strong> | 
                    <strong>📱 {row['source']}</strong> | 
                    <strong>💯 {row['sentiment_score']:.3f}</strong><br>
                    {row['text']}
                </div>
                """, unsafe_allow_html=True)

        with tab2:
            negative_posts = df[df['sentiment'] == 'Negative'].nlargest(5, 'sentiment_score')
            for idx, row in negative_posts.iterrows():
                st.markdown(f"""
                <div style="background-color: #f8d7da; padding: 10px; border-radius: 5px; margin: 5px 0;">
                    <strong>📅 {row['timestamp'].strftime('%Y-%m-%d %H:%M')}</strong> | 
                    <strong>📱 {row['source']}</strong> | 
                    <strong>💯 {row['sentiment_score']:.3f}</strong><br>
                    {row['text']}
                </div>
                """, unsafe_allow_html=True)

        with tab3:
            neutral_posts = df[df['sentiment'] == 'Neutral'].nlargest(5, 'sentiment_score')
            for idx, row in neutral_posts.iterrows():
                st.markdown(f"""
                <div style="background-color: #e2e3e5; padding: 10px; border-radius: 5px; margin: 5px 0;">
                    <strong>📅 {row['timestamp'].strftime('%Y-%m-%d %H:%M')}</strong> | 
                    <strong>📱 {row['source']}</strong> | 
                    <strong>💯 {row['sentiment_score']:.3f}</strong><br>
                    {row['text']}
                </div>
                """, unsafe_allow_html=True)

        # Export data
        st.markdown("---")
        st.subheader("💾 Export Analysis Results")

        col1, col2 = st.columns(2)

        with col1:
            # Download full data
            csv = df.to_csv(index=False)
            st.download_button(
                label="📊 Download Full Dataset (CSV)",
                data=csv,
                file_name=f"sentiment_analysis_{st.session_state.get('keyword', 'data').replace(' ', '_')}.csv",
                mime='text/csv'
            )

        with col2:
            # Download summary report
            summary_data = {
                'Metric': ['Total Posts', 'Positive %', 'Negative %', 'Neutral %', 'Avg Confidence'],
                'Value': [
                    total_posts,
                    f"{positive_pct:.1f}%",
                    f"{negative_pct:.1f}%", 
                    f"{(df['sentiment'] == 'Neutral').mean() * 100:.1f}%",
                    f"{df['sentiment_score'].mean():.3f}"
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_csv = summary_df.to_csv(index=False)

            st.download_button(
                label="📋 Download Summary Report (CSV)",
                data=summary_csv,
                file_name=f"sentiment_summary_{st.session_state.get('keyword', 'data').replace(' ', '_')}.csv",
                mime='text/csv'
            )

    else:
        st.warning("No data available for the selected date range. Please adjust your filters.")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        🎭 <strong>AI Sentiment Analysis Dashboard</strong><br>
        Built with Streamlit, DistilBERT, and Plotly<br>
        <em>Transforming social media conversations into actionable insights</em>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
