import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
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

X_train_tensor = torch.tensor(X_train.toarray(), dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(train_labels.values, dtype=torch.float32).unsqueeze(1).to(device)

X_test_tensor = torch.tensor(X_test.toarray(), dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(test_labels.values, dtype=torch.float32).unsqueeze(1).to(device)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

class MLPClassifier(nn.Module):
    def __init__(self, input_dim):
        super(MLPClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

input_dim = X_train.shape[1]
model = MLPClassifier(input_dim).to(device)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor)
    predictions_binary = (predictions > 0.5).float()

y_true = y_test_tensor.cpu().numpy()
y_pred = predictions_binary.cpu().numpy()

print("\nMatriz de Confusão:")
print(confusion_matrix(y_true, y_pred))

print("\nRelatório de Classificação:")
print(classification_report(y_true, y_pred))

print("Acurácia:", accuracy_score(y_true, y_pred))


"""
Relatório de Classificação:
              precision    recall  f1-score   support

         0.0       0.96      0.95      0.96       718
         1.0       0.95      0.97      0.96       722

    accuracy                           0.96      1440
   macro avg       0.96      0.96      0.96      1440
weighted avg       0.96      0.96      0.96      1440

Acurácia: 0.9576388888888889
"""
