import pandas as pd
import re
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


fake = pd.read_csv("dataset/Fake.csv")
true = pd.read_csv("dataset/True.csv")


fake["label"] = 0
true["label"] = 1


data = pd.concat([fake, true], ignore_index=True)


data = data[["text", "label"]]


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


data["text"] = data["text"].apply(clean_text)

X = data["text"]
y = data["label"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)


model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("=" * 40)
print("Model Accuracy :", accuracy)
print("=" * 40)

print(fake.shape)
print(true.shape)
print(fake["label"].value_counts())
print(true["label"].value_counts())


pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model and Vectorizer saved successfully!")


while True:
    news = input("\nEnter News (type 'exit' to quit): ")

    if news.lower() == "exit":
        print("Program Closed.")
        break

    cleaned_news = clean_text(news)
    news_vector = vectorizer.transform([cleaned_news])

    prediction = model.predict(news_vector)

    if prediction[0] == 0:
        print("Prediction: Fake News")
    else:
        print("Prediction: Real News")