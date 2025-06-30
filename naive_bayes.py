import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import torch
from dotenv import load_dotenv
import os


load_dotenv()
path = os.getenv("PATH_DATA")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Usando dispositivo:", device)

df = pd.read_csv(path)
df = df[["preprocessed_news", "label"]]
df['label'] = df['label'].map({'fake': 0, 'true': 1})
df = df.dropna()

train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['preprocessed_news'], df['label'], test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(max_features=5000)

X_train = vectorizer.fit_transform(train_texts)
X_test = vectorizer.transform(test_texts)

nb = MultinomialNB()

nb.fit(X_train, train_labels)

y_pred_nb = nb.predict(X_test)

print("Matriz de Confusão:")
print(confusion_matrix(test_labels, y_pred_nb))

print("\nRelatório de Classificação:")
print(classification_report(test_labels, y_pred_nb))

print("\nAcurácia:", accuracy_score(test_labels, y_pred_nb))

"""
Relatório de Classificação:
              precision    recall  f1-score   support

           0       0.88      0.82      0.85       718
           1       0.83      0.89      0.86       722

    accuracy                           0.86      1440
   macro avg       0.86      0.86      0.86      1440
weighted avg       0.86      0.86      0.86      1440


Acurácia: 0.8555555555555555
"""
