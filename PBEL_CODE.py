import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("music_sentiment_dataset.csv")

print("Dataset Preview:\n")
print(df.head())

print("\nMissing Values:\n")
print(df.isnull().sum())

df = df.dropna(subset=["User_Text", "Sentiment_Label"])

emotion_count = df["Sentiment_Label"].value_counts()

show_graph = input("Do you want to see emotion distribution graph? (yes/no): ")

if show_graph.lower() == "yes":
    plt.figure()
    emotion_count.plot(kind="bar")
    plt.title("Emotion Distribution")
    plt.xlabel("Emotion")
    plt.ylabel("Number of Texts")
    plt.show()

X = df["User_Text"]
y = df["Sentiment_Label"]

vectorizer = TfidfVectorizer()

X_vector = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vector, y, test_size=0.2, random_state=42
)

model = LogisticRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\n----- Emotion Detection -----")

user_text = input("Enter a sentence: ")

user_vector = vectorizer.transform([user_text])

prediction = model.predict(user_vector)

print("Predicted Emotion:", prediction[0])