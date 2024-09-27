import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download necessary NLTK resources
nltk.download('vader_lexicon') #VADER utilizes a sentiment lexicon that contains over 7,500 lexical features, each assigned a valence score ranging from -4 (extremely negative) to +4 (extremely positive).
 

def sentiment_analysis(text):
    # Create a SentimentIntensityAnalyzer object
    sia = SentimentIntensityAnalyzer()

    # Perform sentiment analysis on the given text
    sentiment_scores = sia.polarity_scores(text)

    return sentiment_scores


# Example usage
text = "."

sentiment_scores = sentiment_analysis(text)

print("Sentiment Analysis Results:")
print(f"Positive score: {sentiment_scores['pos']}")
print(f"Negative score: {sentiment_scores['neg']}")
print(f"Neutral score: {sentiment_scores['neu']}")
print(f"Compound score: {sentiment_scores['compound']}")
