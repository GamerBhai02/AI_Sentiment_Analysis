import re
import emoji
from cleantext import clean
import pandas as pd

def clean_text(text):
    """
    Clean and preprocess text data
    """
    if not isinstance(text, str):
        return ""
    
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove mentions
    text = re.sub(r'@\w+', '', text)
    # Remove hashtags
    text = re.sub(r'#\w+', '', text)
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert emojis to text
    text = emoji.demojize(text)
    # Convert to lowercase
    text = text.lower()
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def preprocess_data(df):
    """
    Preprocess the entire dataframe
    """
    df = df.copy()
    df['cleaned_text'] = df['text'].apply(clean_text)
    df = df[df['cleaned_text'].str.len() > 10]  # Remove very short texts
    return df
