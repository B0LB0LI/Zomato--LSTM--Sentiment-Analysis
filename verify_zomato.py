import pandas as pd
import sys
import os

file_path = r"e:\Techwing work\TCW DL\Zomato LSTM Sentiment analysis\Ratings.csv"

def define_sentiment(rating):
    try:
        r = float(rating)
        if r >= 4.0:
            return 'Positive'
        elif r >= 3.0:
            return 'Neutral'
        else:
            return 'Negative'
    except:
        return 'Negative'

print("Loading dataset head...")
df = pd.read_csv(file_path, nrows=100)

print("\n--- Dataset Head ---")
print(df.head())

print("\n--- Dataset Info ---")
df.info()

print("\n--- Sentiment Mapping Test ---")
test_ratings = [5.0, 4.0, 3.5, 3.0, 2.5, 1.0]
for r in test_ratings:
    print(f"Rating {r} -> {define_sentiment(r)}")

df['Sentiment'] = df['rating'].apply(define_sentiment)
print("\nSample Sentiment counts (from first 100 rows):")
print(df['Sentiment'].value_counts())
