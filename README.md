# arXiv Paper Classifier

**An NLP pipeline that classifies arXiv abstracts into five research areas using transformer models, with an interactive Streamlit dashboard.**

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1-red)
![Transformers](https://img.shields.io/badge/Transformers-4.35-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## Overview

Given the title and abstract of an arXiv paper, this system predicts which of five
computer-science research areas it belongs to:

- **AI** — Artificial Intelligence (`cs.AI`)
- **ML** — Machine Learning (`cs.LG`)
- **NLP** — Natural Language Processing (`cs.CL`)
- **Computer Vision** (`cs.CV`)
- **Robotics** (`cs.RO`)

The project covers the full loop: pulling real data from the arXiv API, cleaning and
de-duplicating abstracts, fine-tuning a transformer classifier, evaluating it with
macro F1, and serving predictions through a Streamlit app.

These categories overlap heavily in practice — a paper on *"transformers for medical
image segmentation"* has a legitimate claim to ML, CV, and NLP at once — so the
interesting engineering problem here is less about squeezing out accuracy and more
about measuring performance honestly under genuinely ambiguous labels.

---

## Dataset

- **Source:** arXiv API (`export.arxiv.org`) — free, public, ~30-day indexing lag
- **Raw pull:** 2,000 most-recent papers per category × 5 categories ≈ **10,000 papers**
- **After cleaning/de-duplication:** **8,019 papers** used for training and evaluation
- **Date range:** submissions from 2004–2024
- **Fields used:** title, abstract, submission date, category label

Cleaning steps (`data_loader.py`): strip URLs and LaTeX artifacts, normalize
whitespace, drop duplicates, and remove very short abstracts.

> **Note on reproducibility:** because the API returns the *most recent* papers at
> fetch time, re-running the fetch later produces a different snapshot. The fetch and
> preprocessing scripts are deterministic given a fixed CSV; the data download is not.

---

## Quick Start

### 1. Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Fetch data from arXiv (~90 min, network-bound)

```bash
python fetch_10k_full.py
# Pulls 2,000 papers per category -> data/raw/arxiv_10k_papers.csv
```

### 3. Preprocess

```bash
python preprocess_8k.py
# Cleans + de-duplicates -> data/processed/arxiv_8k_cleaned.csv
```

### 4. Train

```bash
python train_8k.py
# Fine-tunes RoBERTa-base on the 8,019-paper set
```

### 5. Launch the dashboard

```bash
streamlit run streamlit_app.py
# Opens at http://localhost:8501
```

---

## Model & Training

The core trainer (`bert_model.py`) wraps a Hugging Face
`AutoModelForSequenceClassification` head on a configurable backbone.

| Setting | Value |
|---|---|
| Backbone | `roberta-base` (RoBERTa) |
| Split | Stratified 70 / 15 / 15 train / val / test |
| Optimizer | AdamW, learning rate `2e-5` |
| Batch size | 16 (train) |
| Epochs | up to 10, with best-checkpoint selection on validation macro F1 |
| Seeds | `numpy` and `torch` fixed at 42 |
| Primary metric | **Macro F1** |

**Why macro F1 and not accuracy:** macro F1 weights every class equally, so a model
can't coast on the larger categories while quietly failing on a smaller one. With
overlapping research areas, it's the metric that tells you the truth about per-class
behavior.

### Lightweight variant

`train_10yr_10k.py` runs a deliberately cheaper configuration: **DistilBERT used as a
frozen feature extractor on CPU**, with `max_length=128`. This trades some accuracy for
a setup that trains without a GPU — useful for quick iteration or reproducing the
pipeline on a laptop.

---

## Results

> Replace the values below with the actual numbers printed by `train_8k.py` on your
> run. (The classifier reports macro F1 and a full `classification_report` on the held-out
> test set.) Until you commit a real results file, keep this section honest about what
> you measured rather than listing aspirational numbers.

| Model | Setup | Macro F1 (test) |
|---|---|---|
| RoBERTa-base | fine-tuned, 8,019 papers | _your measured value (~0.80)_ |
| DistilBERT | frozen features, CPU | _your measured value_ |

Per-class behavior (precision / recall / F1) comes from the printed
`classification_report`. The most useful output isn't the headline number — it's the
**confusion matrix**, because the pairs the model mixes up (e.g. AI ↔ ML, ML ↔ CV) tend
to reflect real overlap between these research areas rather than model error.

To regenerate results:

```bash
python train_8k.py        # prints macro F1 + classification_report on the test set
```

---

## Inference

```python
from bert_model import BertClassifier

clf = BertClassifier(model_name='roberta-base')
clf.load_model('models/roberta-base_best.pt')

result = clf.predict("We present a transformer-based approach to ...")
# {'predicted_label': 'NLP', 'confidence': 0.94, 'all_scores': {...}}
```

---

## Project Structure

```
arxiv-main/
├── data_loader.py            # arXiv API fetch + text cleaning
├── fetch_10k_full.py         # pulls 2,000 papers/category from arXiv
├── preprocess_8k.py          # cleaning -> 8,019-paper dataset
├── bert_model.py             # BertClassifier: training/eval/inference
├── train_8k.py               # fine-tune RoBERTa on the 8k set
├── train_10yr_10k.py         # frozen-DistilBERT (CPU) variant
├── statistical_analysis.py   # dataset/label distribution analysis
├── keyword_analyzer.py       # keyword-based category signals
├── streamlit_app.py          # interactive dashboard
├── tests/                    # (in progress)
└── requirements.txt
```

---

## Deployment

### Streamlit Cloud
Push to GitHub, connect the repo at <https://share.streamlit.io>, and select
`streamlit_app.py` as the entry point.

### Docker

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py"]
```

```bash
docker build -t arxiv-classifier .
docker run -p 8501:8501 arxiv-classifier
```

---

## Limitations & Honest Notes

- **Snapshot data.** Trained on ~8K recent abstracts; arXiv categories drift over time,
  so a model trained today won't perfectly fit papers from a different period.
- **Overlapping labels.** Many papers legitimately belong to several of these five
  categories; single-label accuracy has a real ceiling here.
- **No external validation set.** Evaluation is a held-out split of the same snapshot,
  not a separate independent corpus.
- **Title + abstract only.** No full text, citations, or author/venue signals are used.

---

## Roadmap

- [ ] Commit a reproducible `results.json` from a real training run
- [ ] Add unit tests for the preprocessing and inference paths
- [ ] Multi-label classification (let a paper belong to more than one area)
- [ ] Confidence calibration and an "uncertain" abstain option
- [ ] Lightweight API endpoint for programmatic classification

---

## Tech Stack

PyTorch · Hugging Face Transformers (RoBERTa / DistilBERT) · scikit-learn · pandas ·
Streamlit · Docker

## License

MIT — see `LICENSE`.

## Contact

- **GitHub:** [moulionmission](https://github.com/moulionmission)
- **Email:** _your email_
- **LinkedIn:** _your profile_
