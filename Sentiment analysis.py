import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download necessary NLTK resources
nltk.download('vader_lexicon')


def sentiment_analysis(text):
    # Create a SentimentIntensityAnalyzer object
    sia = SentimentIntensityAnalyzer()

    # Perform sentiment analysis on the given text
    sentiment_scores = sia.polarity_scores(text)

    return sentiment_scores


# Example usage
text = "The NSE Nifty 50 index was down 0.4% at 24,199.9, and the S&P BSE Sensex had shed 0.38% to 79,161.1. Both indexes were down about 0.3% ahead of the decision. They extended losses to about 0.6% immediately after the policy announcement, before trimming some losses.The six-member MPC also continued with the monetary policy stance of withdrawal of accommodation. Four out of six MPC members voted in favour of the rate decision. Announcing the decision, RBI Governor Shaktikanda Das said that inflation broadly has been on a declining trajectory.The RBI had last cut the repo rate by 40 basis points to 4 per cent in May 2020 when the Covid pandemic raged across the country affecting the entire economy, leading to slowdown in demand, production cuts and job losses. Since then, the RBI has hiked the repo rate by 250 points to 6.50 per cent in order to tackle high inflation level after the epidemic subsided.Among the 30 Sensex firms, Power Grid, Infosys, Larsen & Toubro, JSW Steel, UltraTech Cement and Asian Paints were the biggest laggards. Tata Motors, HDFC Bank, Tech Mahindra and ITC were among the gainers during the initial trade."

sentiment_scores = sentiment_analysis(text)

print("Sentiment Analysis Results:")
print(f"Positive score: {sentiment_scores['pos']}")
print(f"Negative score: {sentiment_scores['neg']}")
print(f"Neutral score: {sentiment_scores['neu']}")
print(f"Compound score: {sentiment_scores['compound']}")
