import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

reviews = pd.read_csv("../data/reviews_0-250.csv", low_memory=False)

print("\n=== DATASET PREVIEW ===\n")
print(reviews.head())

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

reviews["clean_review"] = reviews["review_text"].apply(clean_text)

print("\n=== CLEANED DATA ===\n")
print(reviews[["review_text", "clean_review"]].head())

def label_sentiment(rating):
    if rating >= 4:
        return "positive"
    elif rating == 3:
        return "neutral"
    else:
        return "negative"

reviews["sentiment"] = reviews["rating"].apply(label_sentiment)

print("\n=== SENTIMENT LABELS ===\n")
print(reviews[["rating", "sentiment"]].head())

X = reviews["clean_review"]
y = reviews["sentiment"]

vectorizer = TfidfVectorizer(max_features=5000)
X_vec = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2)

model = LogisticRegression(max_iter=300)
model.fit(X_train, y_train)

print("\n=== MODEL TRAINED ===\n")

y_pred = model.predict(X_test)

print("\n=== MODEL EVALUATION ===\n")
print(classification_report(y_test, y_pred))