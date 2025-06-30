# Classificar Fake News

Este projeto usa um modelo BERT,KNN, MLP e Naive Bayes para classificar notícias como **falsas** ou **verdadeiras**, com base no corpus [Fake.br-Corpus](https://github.com/roneysco/Fake.br-Corpus).

---

## 🚀 Funcionalidades

- Classificação binária de fake news (`true` ou `fake`)
- Pré-processamento de dados com `pandas`
- Treinamento com `BERT` (modelo `neuralmind/bert-base-portuguese-cased`)
- Avaliação com métricas do `scikit-learn`
- Visualização do progresso com `tqdm`
- Carregamento seguro de variáveis com `.env`

---

## 📂 Estrutura do Projeto

```
classificar_fakenews/
│
├── bert.py                # Script principal para treino/teste
├── .env                   # Caminho para o dataset
├── requirements.txt       # Dependências
├── README.md              # Este arquivo
└── ...
```

---

## 📦 Instalação

### 1. Clone o repositório

```bash
git clone https://github.com/victgab20/classificar_fakenews.git
cd classificar_fakenews
```

### 2. Crie e ative um ambiente virtual

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

> ⚠️ Se você tiver uma GPU e quiser usar CUDA (ex: CUDA 12.1), veja instruções específicas em: [https://pytorch.org/get-started/locally](https://pytorch.org/get-started/locally)

---

## ⚙️ Configuração

Crie um arquivo `.env` com o caminho completo do seu dataset CSV:

```
PATH_DATA=C:/caminho/para/seu/dataset/pre-processed.csv
```

---

## Como executar

Após configurar o caminho no `.env`, basta executar o script:

```bash
python bert.py
```

---

## 🧠 Modelo usado

Este projeto usa o modelo:

- [`neuralmind/bert-base-portuguese-cased`](https://huggingface.co/neuralmind/bert-base-portuguese-cased)

---
