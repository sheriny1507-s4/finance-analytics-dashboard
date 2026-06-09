import pandas as pd
import pickle

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load dataset
df = pd.read_csv("large_dataset.csv")

# Features
X = df["Description"]

# Labels
y = df["Category"]

# Create model
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", MultinomialNB())
])

# Train
model.fit(X, y)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ model.pkl created successfully")
