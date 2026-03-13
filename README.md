# PBEL
NAME: HEENA JAIN
BATCH: 8
PROJECT: AI BASED EMOTION RECOGNIZE SYSTEM FROM TEXT
# PRESENTATION
[DOWNLOAD PPT] (https://drive.google.com/open?id=1wGR83c128HoNH6dSOwHAB84dPFDTvccM&usp=drive_copy)
## PRESENTATION RECORDING
[WATCH VIDEO] (https://drive.google.com/open?id=12wFuCFE2kWaawjngs33y9vPR_pBWNP_c&usp=drive_copy)
# CODE
```
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
```
<img width="925" height="906" alt="image" src="https://github.com/user-attachments/assets/e218aad6-74d2-43eb-bb2d-33ed599646f1" />
<img width="561" height="191" alt="image" src="https://github.com/user-attachments/assets/aa683bcb-ae83-4cc9-8f50-d63fd3deac80" />
<img width="798" height="684" alt="image" src="https://github.com/user-attachments/assets/f5a1d2ab-8e52-463d-8334-42a6c5e1236b" />

