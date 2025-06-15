from transformers import pipeline
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score

# Load data
df = pd.read_csv("data/sentiment_test_dataset_200.csv")

# Load improved sentiment model
sentiment_model = pipeline("sentiment-analysis", model="siebert/sentiment-roberta-large-english")

# This model returns only "POSITIVE" or "NEGATIVE"
def map_labels(label):
    if label == "POSITIVE":
        return "positive"
    elif label == "NEGATIVE":
        return "negative"
    else:
        return "neutral"

# Make predictions
results = sentiment_model(df["text"].tolist(), batch_size=16)
predicted_labels = [map_labels(res["label"]) for res in results]

# Evaluate
true_labels = df["label"].str.lower()
print("Accuracy:", accuracy_score(true_labels, predicted_labels))
print("\nClassification Report:\n")
print(classification_report(true_labels, predicted_labels, digits=4))
