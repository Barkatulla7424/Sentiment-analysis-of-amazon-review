import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.pipeline import make_pipeline

# Download NLTK resources
import nltk
nltk.download('punkt')
nltk.download('stopwords')

# Load your dataset (replace 'your_dataset.csv' with the actual file path)
df = pd.read_csv('amazon_vfl_reviews.csv')

# Drop rows with missing reviews
df.dropna(subset=['review'], inplace=True)

# Convert ratings to sentiments
df['label'] = df['rating'].apply(lambda x: 'positive' if x > 3 else 'neutral' if x == 3 else 'negative')

# Select relevant columns
df = df[['name', 'review', 'label']]

# Create a pipeline with TF-IDF vectorizer and Multinomial Naive Bayes classifier
model = make_pipeline(
    TfidfVectorizer(stop_words=stopwords.words('english'), tokenizer=word_tokenize),
    MultinomialNB()
)

# Train the model
model.fit(df['review'], df['label'])

# Example usage for predicting sentiment of a new review
new_product_name = input("Enter the product name: ")
new_review = input("Enter the review: ")

predicted_sentiment = model.predict([new_review])[0]
print(f'Predicted Sentiment for the review of {new_product_name}: {predicted_sentiment}')