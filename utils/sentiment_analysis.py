from transformers import pipeline
import pandas as pd
import numpy as np

# Cache the sentiment analysis model
_sentiment_pipeline = None

def get_sentiment_model():
    """
    Load and return the sentiment analysis model
    """
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        _sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            return_all_scores=True
        )
    return _sentiment_pipeline

def analyze_sentiment(text, model):
    """
    Analyze sentiment of a single text
    """
    if not text or len(text.strip()) < 5:
        return "Neutral", 0.5
    
    try:
        result = model(text[:512])[0]  # Limit to first 512 characters
        # Find the sentiment with highest score
        sentiment_scores = {item['label']: item['score'] for item in result}
        
        # Convert to Positive/Neutral/Negative format
        if 'POSITIVE' in sentiment_scores and 'NEGATIVE' in sentiment_scores:
            pos_score = sentiment_scores['POSITIVE']
            neg_score = sentiment_scores['NEGATIVE']
            
            if pos_score > neg_score and pos_score > 0.6:
                return "Positive", pos_score
            elif neg_score > pos_score and neg_score > 0.6:
                return "Negative", neg_score
            else:
                return "Neutral", max(pos_score, neg_score)
        else:
            # Fallback if labels are different
            best = max(result, key=lambda x: x['score'])
            return best['label'], best['score']
            
    except Exception as e:
        print(f"Error in sentiment analysis: {e}")
        return "Neutral", 0.5

def analyze_sentiment_batch(texts, model):
    """
    Analyze sentiment for a batch of texts
    """
    sentiments = []
    confidence_scores = []
    
    for text in texts:
        sentiment, score = analyze_sentiment(text, model)
        sentiments.append(sentiment)
        confidence_scores.append(score)
    
    return sentiments, confidence_scores
