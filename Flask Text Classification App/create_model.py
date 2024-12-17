from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib
import os

# Create a directory for saving the model if it doesn't exist
model_dir = 'model'
os.makedirs(model_dir, exist_ok=True)

# Load the 20 Newsgroups dataset
newsgroups = fetch_20newsgroups(subset='all', categories=None, shuffle=True, random_state=42)

# Extract the data and labels
X = newsgroups.data  # Text data
y = newsgroups.target  # Labels (categories)

# Convert text data into TF-IDF features
vectorizer = TfidfVectorizer(stop_words='english', max_features=15000)  # Convert text to numerical features
X_tfidf = vectorizer.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Build a Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate model accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the trained vectorizer and model
joblib.dump(vectorizer, os.path.join(model_dir, 'vectorizer.pkl'))  # Save vectorizer
joblib.dump(model, os.path.join(model_dir, 'classifier.pkl'))  # Save the classifier
