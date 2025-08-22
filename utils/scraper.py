import snscrape.modules.twitter as sntwitter
import pandas as pd
from datetime import datetime, timedelta
import time

def scrape_tweets(keyword, max_tweets=1000):
    """
    Scrape tweets using snscrape without API authentication
    """
    tweets_list = []
    
    # Using TwitterSearchScraper to scrape data
    try:
        for i, tweet in enumerate(sntwitter.TwitterSearchScraper(f'{keyword} since:{get_past_date(7)}').get_items()):
            if i >= max_tweets:
                break
            tweets_list.append([
                tweet.date, 
                tweet.id, 
                tweet.content, 
                tweet.user.username, 
                tweet.likeCount, 
                tweet.retweetCount,
                tweet.replyCount
            ])
    except Exception as e:
        print(f"Error scraping tweets: {e}")
        return pd.DataFrame()
    
    # Create a dataframe
    tweets_df = pd.DataFrame(tweets_list, columns=[
        'datetime', 'tweet_id', 'text', 'username', 'like_count', 
        'retweet_count', 'reply_count'
    ])
    
    return tweets_df

def get_past_date(days):
    """Get date string for days ago"""
    return (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
