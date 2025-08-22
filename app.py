import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time

# Import utility functions
from utils.scraper import scrape_tweets, get_past_date
from utils.preprocess import preprocess_data, clean_text
from utils.sentiment_analysis import get_sentiment_model, analyze_sentiment_batch

# Page configuration
st.set_page_config(
    page_title="Dynamic Sentiment Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {font-size: 3rem; color: #1DA1F2;}
    .sub-header {font-size: 1.5rem; color: #14171A;}
    .positive {color: #4CAF50;}
    .negative {color: #F44336;}
    .neutral {color: #9E9E9E;}
    .stProgress > div > div > div > div {background-color: #1DA1F2;}
</style>
""", unsafe_allow_html=True)

# App title and description
st.markdown('<h1 class="main-header">Dynamic AI Sentiment Analysis Dashboard</h1>', unsafe_allow_html=True)
st.markdown("""
This application analyzes sentiment of social media posts in real-time. 
Enter a keyword or hashtag to see sentiment distribution and insights.
""")

# Sidebar for input
with st.sidebar:
    st.header("Search Parameters")
    keyword = st.text_input("Enter keyword or hashtag", "artificial intelligence")
    max_tweets = st.slider("Number of tweets to analyze", 100, 2000, 500)
    analyze_btn = st.button("Analyze Sentiment", type="primary")

# Initialize session state
if 'tweets_df' not in st.session_state:
    st.session_state.tweets_df = None
if 'results_df' not in st.session_state:
    st.session_state.results_df = None

# Main content
if analyze_btn or st.session_state.results_df is not None:
    if analyze_btn:
        # Scrape and process data
        with st.spinner("Scraping tweets..."):
            progress_bar = st.progress(0)
            tweets_df = scrape_tweets(keyword, max_tweets)
            progress_bar.progress(30)
            
            if tweets_df.empty:
                st.error("No tweets found or error occurred during scraping. Try a different keyword.")
                st.stop()
            
            # Preprocess data
            with st.spinner("Preprocessing data..."):
                tweets_df = preprocess_data(tweets_df)
                progress_bar.progress(60)
            
            # Analyze sentiment
            with st.spinner("Analyzing sentiment..."):
                model = get_sentiment_model()
                sentiments, scores = analyze_sentiment_batch(
                    tweets_df['cleaned_text'].tolist(), 
                    model
                )
                
                tweets_df['sentiment'] = sentiments
                tweets_df['confidence'] = scores
                progress_bar.progress(90)
            
            # Store in session state
            st.session_state.tweets_df = tweets_df
            progress_bar.progress(100)
            time.sleep(0.5)
            progress_bar.empty()
    
    # Display results
    if st.session_state.tweets_df is not None:
        tweets_df = st.session_state.tweets_df
        
        # Summary stats
        st.subheader("Summary Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Posts", len(tweets_df))
        with col2:
            positive_count = len(tweets_df[tweets_df['sentiment'] == 'Positive'])
            st.metric("Positive Posts", positive_count)
        with col3:
            negative_count = len(tweets_df[tweets_df['sentiment'] == 'Negative'])
            st.metric("Negative Posts", negative_count)
        with col4:
            neutral_count = len(tweets_df[tweets_df['sentiment'] == 'Neutral'])
            st.metric("Neutral Posts", neutral_count)
        
        # Sentiment distribution chart
        st.subheader("Sentiment Distribution")
        fig = px.pie(
            tweets_df, 
            names='sentiment', 
            color='sentiment',
            color_discrete_map={'Positive':'green', 'Negative':'red', 'Neutral':'gray'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Sentiment over time
        st.subheader("Sentiment Over Time")
        if 'datetime' in tweets_df.columns:
            tweets_df['date'] = tweets_df['datetime'].dt.date
            daily_sentiment = tweets_df.groupby(['date', 'sentiment']).size().unstack(fill_value=0)
            
            fig = px.line(
                daily_sentiment, 
                x=daily_sentiment.index, 
                y=daily_sentiment.columns,
                title="Sentiment Trend Over Time",
                labels={'value': 'Number of Posts', 'date': 'Date', 'variable': 'Sentiment'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Word clouds
        st.subheader("Word Clouds by Sentiment")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<p class="positive">Positive Words</p>', unsafe_allow_html=True)
            positive_text = " ".join(tweets_df[tweets_df['sentiment'] == 'Positive']['cleaned_text'])
            if positive_text:
                wordcloud = WordCloud(width=300, height=200, background_color='white').generate(positive_text)
                plt.figure(figsize=(5, 4))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis("off")
                st.pyplot(plt)
            else:
                st.info("No positive posts to display")
        
        with col2:
            st.markdown('<p class="neutral">Neutral Words</p>', unsafe_allow_html=True)
            neutral_text = " ".join(tweets_df[tweets_df['sentiment'] == 'Neutral']['cleaned_text'])
            if neutral_text:
                wordcloud = WordCloud(width=300, height=200, background_color='white').generate(neutral_text)
                plt.figure(figsize=(5, 4))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis("off")
                st.pyplot(plt)
            else:
                st.info("No neutral posts to display")
        
        with col3:
            st.markdown('<p class="negative">Negative Words</p>', unsafe_allow_html=True)
            negative_text = " ".join(tweets_df[tweets_df['sentiment'] == 'Negative']['cleaned_text'])
            if negative_text:
                wordcloud = WordCloud(width=300, height=200, background_color='white').generate(negative_text)
                plt.figure(figsize=(5, 4))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis("off")
                st.pyplot(plt)
            else:
                st.info("No negative posts to display")
        
        # Sample posts
        st.subheader("Sample Posts")
        
        sentiment_filter = st.selectbox("Filter by sentiment", ["All", "Positive", "Neutral", "Negative"])
        
        if sentiment_filter != "All":
            sample_df = tweets_df[tweets_df['sentiment'] == sentiment_filter]
        else:
            sample_df = tweets_df
        
        sample_df = sample_df.head(10)
        
        for _, row in sample_df.iterrows():
            sentiment_color = "positive" if row['sentiment'] == "Positive" else \
                             "negative" if row['sentiment'] == "Negative" else "neutral"
            
            st.markdown(f"""
            <div style="padding: 10px; border-radius: 5px; border-left: 5px solid var(--{sentiment_color}); 
                        background-color: #f9f9f9; margin-bottom: 10px;">
                <p style="margin: 0;"><strong>@{row.get('username', 'N/A')}</strong> · 
                <span class="{sentiment_color}">{row['sentiment']}</span> · 
                {row.get('datetime', '').strftime('%Y-%m-%d %H:%M') if pd.notna(row.get('datetime')) else 'N/A'}</p>
                <p style="margin: 5px 0 0 0;">{row['text']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Download button
        csv = tweets_df[['datetime', 'username', 'text', 'sentiment', 'confidence']].to_csv(index=False)
        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name=f"sentiment_analysis_{keyword}.csv",
            mime="text/csv"
        )

else:
    # Show instructions before analysis
    st.info("""
    👆 Enter a keyword or hashtag in the sidebar and click 'Analyze Sentiment' to begin.
    
    This tool will:
    1. Scrape recent tweets containing your keyword
    2. Clean and preprocess the text
    3. Analyze sentiment using a Transformer-based AI model
    4. Visualize the results with charts and word clouds
    
    Try keywords like: 
    - "artificial intelligence"
    - "climate change"
    - "cryptocurrency"
    - "[your favorite product]"
    """)
    
    # Placeholder for demo purposes
    st.image("https://via.placeholder.com/800x400?text=Dynamic+Sentiment+Analysis+Dashboard", use_column_width=True)

# Footer
st.markdown("---")
st.markdown("### About")
st.markdown("""
This sentiment analysis dashboard uses:
- **snscrape** for collecting Twitter data without API
- **DistilBERT** transformer model for sentiment classification
- **Streamlit** for the interactive web interface

The analysis is performed in real-time on recently scraped data.
""")
