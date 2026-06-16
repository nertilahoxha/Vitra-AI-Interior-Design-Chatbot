# 🪑 Chatbot AI Multimodale per Vitra

Assistente virtuale intelligente per il catalogo prodotti Vitra, basato su **Retrieval-Augmented Generation (RAG)** e **Computer Vision**.

Il sistema consente agli utenti di:

- 📖 Interrogare le schede tecniche dei prodotti tramite linguaggio naturale
- 📷 Caricare una fotografia di una sedia Vitra per identificarne automaticamente il modello
- 🤖 Ricevere risposte tecniche accurate generate da un LLM utilizzando esclusivamente la documentazione ufficiale

## 🎯 Obiettivo del Progetto

Vitra possiede un catalogo ampio e altamente tecnico, caratterizzato da numerose varianti di:

- Materiali
- Colori
- Tessuti
- Dimensioni
- Configurazioni

L'obiettivo del progetto è semplificare l'accesso a queste informazioni per clienti, architetti, interior designer e personale showroom attraverso un chatbot multimodale capace di comprendere sia testo che immagini.

## 🏗️ Architettura del Sistema

<p align="center">
  <img src="https://github.com/user-attachments/assets/5c613cb6-ad63-4214-b61e-e408eeb3701f" width="100%">
</p>

## 🔍 Modulo RAG

### Dataset Documentale

Per ciascun prodotto selezionato sono state utilizzate:

- Schede tecniche ufficiali Vitra
- Documentazione in lingua italiana
- Documentazione in lingua inglese

I PDF vengono elaborati tramite parsing automatico e successivamente suddivisi in chunk semantici.

Ogni chunk viene arricchito con i seguenti metadati:

- **Product**
- **Language**
- **Section**

per migliorare la precisione del retrieval e la qualità delle risposte.

### Embedding

Modello utilizzato:

```python
sentence-transformers/all-MiniLM-L6-v2
```

Per l'indicizzazione vettoriale è stato utilizzato **ChromaDB**.

Funzionalità principali:

- Persistenza locale
- Nearest-neighbor search
- Filtraggio tramite metadati
- API Python native

### Retrieval-Augmented Generation

Pipeline:

```text
Query Utente
      ↓
Generazione Embedding
      ↓
Ricerca Semantica
      ↓
Top-K Retrieval
      ↓
Prompt Building
      ↓
GPT-4.1-mini
      ↓
Risposta
```

Il sistema applica inoltre filtri su:

- Lingua
- Prodotto identificato

prima dell'invio del contesto al modello linguistico.

### LLM

Modello utilizzato:

```text
GPT-4.1-mini
```

Il prompt è progettato per:

- Utilizzare esclusivamente il contesto recuperato
- Ridurre le allucinazioni
- Fornire risposte tecniche coerenti e verificabili

## 👁️ Computer Vision

### Dataset Immagini

| Caratteristica | Valore |
|--------------|---------|
| Modelli Vitra | 10 |
| Immagini Totali | 1627 |

Le immagini sono state raccolte tramite scraping web e successivamente annotate con **Roboflow**.

### YOLO11n

YOLO11n è il modello principale utilizzato per il riconoscimento delle sedie Vitra.

Configurazione del training:

| Parametro | Valore |
|------------|---------|
| Epochs | 100 |
| Image Size | 640 |
| Batch Size | 8 |
| Augmentation | Flip |

Performance ottenute:

```text
mAP@50 = 0.923
```

Il modello ha dimostrato ottime prestazioni nel riconoscimento dei prodotti Vitra sia su immagini da catalogo che in contesti reali.

### Faster R-CNN

Come termine di confronto è stato addestrato anche un modello Two-Stage basato su:

```text
ResNet-50 + FPN + Faster R-CNN
```

L'obiettivo era confrontare accuratezza, robustezza e capacità di generalizzazione rispetto all'approccio YOLO.

## 📊 Metriche

Per la valutazione del sistema sono state utilizzate le seguenti metriche:

- Recall@K
- Reciprocal Rank (RR)
- Cosine Similarity
- Faithfulness Score

Queste metriche consentono di valutare sia la qualità del retrieval sia la coerenza delle risposte generate.

## 🛠️ Stack Tecnologico

- Python
- Streamlit
- LangChain
- ChromaDB
- Sentence Transformers
- SBERT
- OpenAI GPT-4.1-mini
- YOLO11n
- PyTorch
- Faster R-CNN
- Roboflow
- DuckDuckGo Search
- PyPDFLoader

## 👥 Autori

- Gloria Albisini
- Edoardo Bruno
- Nertila Hoxha

## 📜 Licenza

Questo progetto è stato sviluppato per finalità accademiche e dimostrative.

I marchi, i prodotti e la documentazione Vitra appartengono ai rispettivi proprietari.
