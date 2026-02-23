# ArXiv Research Topic Analyzer — Milestone 1

> **Traditional NLP-based Research Analysis System**

---

## 📁 Project Structure

```
AgenticResearchAssistant/
├── data/
│   └── arxiv-metadata-oai-snapshot.csv   # ArXiv JSONL dataset (~5GB)
├── src/
│   ├── app.py                            # Streamlit UI (entry point)
│   ├── data_loader.py                    # Dataset loading & sampling
│   ├── preprocessor.py                   # Text cleaning, tokenization, lemmatization
│   ├── topic_modeler.py                  # TF-IDF + LDA / K-Means
│   └── summarizer.py                     # Extractive summarization
├── tests/
│   ├── test_preprocessor.py
│   └── test_summarizer.py
├── reports/
│   └── milestone1_report.md              # Limitations report
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
cd AgenticResearchAssistant
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Run the Streamlit app

```bash
streamlit run src/app.py
```

### 3. Run tests

```bash
pytest tests/ -v
```

---

## ⚙️ Features (Milestone 1)

| Feature | Implementation |
|---------|---------------|
| Text Preprocessing | Tokenization, stop-word removal, lemmatization (spaCy / NLTK) |
| Feature Extraction | TF-IDF, Bag-of-Words (CountVectorizer) |
| Topic Modeling | LDA or K-Means Clustering |
| Extractive Summary | Word-frequency sentence scoring |
| Visualization | Word clouds, bar charts, pie charts (Plotly) |
| UI | Streamlit dark-mode dashboard |

---

## 📊 Dataset

- **Source**: [ArXiv Cornell University Kaggle](https://www.kaggle.com/datasets/Cornell-University/arxiv)
- **Format**: JSONL — each line is a JSON paper record
- **Size**: ~2.9M papers, ~5GB
- **Key fields**: `id`, `title`, `abstract`, `categories`, `authors`, `update_date`

The application uses **efficient streaming + reservoir sampling** to handle the large dataset without loading it all into memory.

---

## 🔬 NLP Pipeline

```
Raw ArXiv Abstract
       ↓
  [Data Loader]      — stream & sample, keyword filter
       ↓
  [Preprocessor]     — clean LaTeX, tokenize, remove stop words, lemmatize
       ↓
  [Feature Extractor]— TF-IDF matrix (scikit-learn)
       ↓
  [Topic Modeler]    — LDA or K-Means → topic assignments
       ↓
  [Summarizer]       — word-frequency extractive summary per topic
       ↓
  [Streamlit UI]     — word cloud, charts, topic cards, download
```

---

## 📝 Report

See [`reports/milestone1_report.md`](reports/milestone1_report.md) for a detailed discussion of the system architecture and **limitations of traditional NLP approaches**.
