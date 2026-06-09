import pandas as pd
import pickle

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load dataset
import pandas as pd

df = pd.read_csv(
    r"C:\Users\Sherin Y\OneDrive\Desktop\finance_analytics\data\rule_based_labeled_dataset.csv"
)

# Features and labels
X = df["Description"]
y = df["Category"]

# Create model pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", MultinomialNB())
])

# Train model
model.fit(X, y)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ model.pkl created successfully")
