# Project 9: arXiv Scientific Paper Classification
## Research-Level NLP Pipeline for UF Portfolio

---

## 1. PROJECT ARCHITECTURE

```
arxiv-paper-classification/
├── data/
│   ├── raw/                    # Unprocessed arXiv dump
│   ├── processed/              # Cleaned, tokenized datasets
│   └── metadata/               # Topic categories, citations
├── models/
│   ├── bert_classifier.py      # BERT fine-tuning
│   ├── roberta_classifier.py   # RoBERTa variant
│   └── distilbert_classifier.py # Lightweight variant
├── analysis/
│   ├── topic_evolution.py      # Trend analysis over time
│   ├── citation_analysis.py    # Citation patterns
│   └── statistical_tests.py    # Significance testing
├── utils/
│   ├── data_loader.py          # arXiv API integration
│   ├── preprocessor.py         # Text cleaning
│   └── visualizer.py           # Matplotlib/Plotly charts
├── notebooks/
│   ├── 01_eda.ipynb           # Exploratory analysis
│   ├── 02_model_training.ipynb
│   └── 03_results_analysis.ipynb
├── streamlit_app.py            # Interactive dashboard
├── requirements.txt
└── README.md
```

---

## 2. PHASE 1: DATA COLLECTION & PREPROCESSING

### 2.1 arXiv Data Pipeline
**Source**: arXiv API (free, 30-day lag)  
**Approach**: Bulk download of recent papers with metadata

```python
# Key fields to extract:
- Paper ID
- Title
- Abstract
- Authors
- Submission date
- Primary category (cs.AI, cs.LG, cs.NE, cs.CV, cs.RO)
- Citation count (from arXiv API or Semantic Scholar API)
```

**Target Categories** (5-way classification):
- AI (cs.AI)
- ML (cs.LG)
- NLP (cs.CL)
- Computer Vision (cs.CV)
- Robotics (cs.RO)

### 2.2 Dataset Size
- **Initial target**: 50,000–100,000 papers (2023–2025)
- **Training split**: 70% train, 15% val, 15% test
- **Class balance check**: Ensure <3:1 imbalance ratio

### 2.3 Text Preprocessing
1. **Tokenization**: Use `transformers.AutoTokenizer` (model-specific)
2. **Length handling**: Truncate abstracts to 512 tokens (BERT limit)
3. **Normalization**: Remove URLs, extra whitespace; keep domain terminology
4. **No aggressive stemming** (preserve semantic meaning)

---

## 3. PHASE 2: MODEL DEVELOPMENT

### 3.1 Three-Model Comparison

| Model | Params | Inference Time | Accuracy | Best For |
|-------|--------|-----------------|----------|----------|
| **BERT-base** | 110M | ~200ms | ~92% | Baseline (best accuracy) |
| **RoBERTa-base** | 125M | ~210ms | ~93% | Better pretraining (improved) |
| **DistilBERT** | 66M | ~80ms | ~89% | Deployment (speed trade-off) |

### 3.2 Fine-Tuning Strategy
- **Learning rate**: 2e-5 (standard for classification)
- **Batch size**: 16–32 (depends on GPU RAM)
- **Epochs**: 3–5 (avoid overfitting)
- **Warmup steps**: 10% of total steps
- **Loss**: CrossEntropyLoss (5-class)

### 3.3 Evaluation Metrics
- **Primary**: Macro F1 (handle class imbalance)
- **Secondary**: Per-class precision/recall, confusion matrix
- **Interpretability**: SHAP values on abstract embeddings

---

## 4. PHASE 3: STATISTICAL ANALYSIS

### 4.1 Topic Evolution Analysis
**Question**: How are research trends changing over time?

```
Approach:
1. Bin papers by month/quarter (2023–2025)
2. Calculate % of papers in each category over time
3. Fit polynomial trend lines
4. Run Mann-Kendall test for trend significance
```

**Visualization**: Stacked area chart, category-specific trend lines

### 4.2 Citation Trend Analysis
**Question**: Which research areas have highest citation velocity?

```
Approach:
1. Extract citation counts from metadata
2. Group by category + submission date
3. Calculate:
   - Mean citations per category
   - Citation growth rate (linear regression)
   - Time-to-citation curve
4. Perform ANOVA to test significance of differences
```

**Visualization**: Box plots, citation velocity scatter plots

### 4.3 Research Growth Metrics
- **Growth rate**: % year-over-year increase per category
- **Diversity**: Topic entropy within each category
- **Correlation**: Citation count vs. abstract length, author count, etc.

---

## 5. PHASE 4: DEPLOYMENT & INSIGHTS

### 5.1 Streamlit Dashboard
**Interactive Features**:
- Upload & classify a new paper (real-time prediction)
- Confidence scores for each category
- Topic trend visualization (interactive filters)
- Citation analytics dashboard
- SHAP-based model interpretability

### 5.2 GitHub Repository
- Clear README with methodology
- Reproducible environment (`requirements.txt`)
- Pre-trained model weights (HuggingFace Hub)
- Jupyter notebooks for full pipeline
- MIT License

### 5.3 Research Artifacts
- **Technical Report**: Model comparison, ablations, error analysis
- **Interactive Dashboard**: Deployed on Streamlit Cloud
- **Insights Brief**: Key findings on topic evolution (1-2 pages)

---

## 6. TIMELINE & MILESTONES

| Week | Milestone | Deliverable |
|------|-----------|------------|
| 1 | Data collection & EDA | Cleaned dataset (100K papers), distribution plots |
| 2 | Preprocessing & tokenization | Ready-to-train data splits |
| 3 | Model training (3 variants) | Trained models, validation results |
| 4 | Analysis & statistical tests | Trend analysis, p-values, visualizations |
| 5 | Streamlit app + deployment | Live dashboard on Streamlit Cloud |
| 6 | Repo + documentation | GitHub with full reproducibility |

---

## 7. WHY THIS IMPRESSES UF LABS

✅ **Methodological rigor**: Proper train/val/test splits, cross-validation, statistical testing  
✅ **Reproducibility**: Public dataset, open models, clear code  
✅ **Scalability**: Handles 100K+ papers; benchmarked on 3 SOTA models  
✅ **Interpretability**: SHAP analysis, error analysis, trend quantification  
✅ **Deployment-ready**: Streamlit app + GitHub + HuggingFace integration  
✅ **Research insight**: Demonstrates ability to extract actionable insights from data  

---

## 8. NEXT STEPS

1. **Confirm tooling**:
   - Python 3.10+
   - transformers (HuggingFace)
   - torch/tensorflow
   - scikit-learn, scipy (stats)
   - streamlit

2. **Choose GPU resource**:
   - Local GPU? (training speed)
   - Google Colab? (free, 15GB VRAM)
   - AWS SageMaker? (mid-tier)

3. **Start Phase 1**:
   - Set up arXiv API data pipeline
   - Pull initial 10K papers for EDA

4. **Create GitHub repo** (skeleton structure ready)

---

## ESTIMATED EFFORT

- **Coding**: 40–50 hours
- **Model training**: 8–12 hours (GPU time)
- **Analysis & writeup**: 10–15 hours
- **Total**: ~70 hours over 6 weeks (10–12 hrs/week)
