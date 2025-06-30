import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm
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

tokenizer = BertTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")

class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.encodings = tokenizer(list(texts), truncation=True, padding=True, max_length=max_len)
        self.labels = list(labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = NewsDataset(train_texts, train_labels, tokenizer)
test_dataset = NewsDataset(test_texts, test_labels, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8)

model = BertForSequenceClassification.from_pretrained(
    "neuralmind/bert-base-portuguese-cased",use_safetensors=True, num_labels=2
)
model.to(device)

optimizer = AdamW(model.parameters(), lr=1e-5)

EPOCHS = 3
model.train()

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    loop = tqdm(train_loader, leave=True)
    for batch in loop:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loop.set_description(f"Epoch {epoch+1}")
        loop.set_postfix(loss=loss.item())

model.eval()
preds, targets = [], []

with torch.no_grad():
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        preds.extend(predictions.cpu().numpy())
        targets.extend(batch["labels"].cpu().numpy())

print("\nRelatório de Classificação:\n")
print(classification_report(targets, preds, target_names=["Fake", "True"]))

"""
Resultados que eu irei usar

Relatório de Classificação:

              precision    recall  f1-score   support

        Fake       0.99      0.94      0.97       718
        True       0.95      0.99      0.97       722

    accuracy                           0.97      1440
   macro avg       0.97      0.97      0.97      1440
weighted avg       0.97      0.97      0.97      1440

"""