# PBEL
NAME: HEENA JAIN
BATCH: 8
PROJECT: AI BASED EMOTION RECOGNIZE SYSTEM FROM TEXT
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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

show_matrix = input("Do you want to see confusion matrix? (yes/no): ")

if show_matrix.lower() == "yes":
    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.colorbar()
    plt.show()

print("\n----- Emotion Detection -----")

user_text = input("Enter a sentence: ")

user_vector = vectorizer.transform([user_text])

prediction = model.predict(user_vector)

print("Predicted Emotion:", prediction[0])
```
<img width="1082" height="657" alt="image" src="https://github.com/user-attachments/assets/578adb80-a995-4fad-b0a1-bbf3220eea59" />
<img width="679" height="487" alt="image" src="https://github.com/user-attachments/assets/f48d2029-730e-43c9-9592-926b960eff1a" />
<img width="809" height="691" alt="image" src="https://github.com/user-attachments/assets/d8ff8b14-9602-45df-91ae-be1521f2fc0b" />
<img width="805" height="687" alt="image" src="https://github.com/user-attachments/assets/05d69878-f35c-4292-bb22-b3e2d0a17846" />



