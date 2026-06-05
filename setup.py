#!/usr/bin/env python3
"""
arXiv Paper Classification: Quick Start Setup
Run this script to initialize the project structure
"""

import os
import sys
from pathlib import Path

def create_directories():
    """Create required directory structure"""
    
    dirs = [
        'data/raw',
        'data/processed',
        'models',
        'results',
        'notebooks',
        'tests'
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {dir_path}")

def create_gitignore():
    """Create .gitignore file"""
    
    gitignore_content = """
# Data files
data/raw/*.csv
data/processed/*.csv

# Model checkpoints
models/*.pt
models/*.bin

# Environment
venv/
.venv/
env/

# IDE
.vscode/
.idea/
*.pyc
__pycache__/

# Jupyter
.ipynb_checkpoints/
*.ipynb

# OS
.DS_Store
Thumbs.db

# Logs
*.log
*.txt~

# Results
results/
output/
"""
    
    with open('.gitignore', 'w') as f:
        f.write(gitignore_content)
    
    print("✓ Created .gitignore")

def create_init_files():
    """Create __init__.py files for Python packages"""
    
    init_files = [
        'tests/__init__.py',
    ]
    
    for file_path in init_files:
        Path(file_path).touch()
        print(f"✓ Created {file_path}")

def create_example_notebook():
    """Create example Jupyter notebook"""
    
    notebook_content = """{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# arXiv Paper Classification: Full Pipeline\\n",
    "\\n",
    "This notebook demonstrates the complete workflow for training and evaluating the paper classifier."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Install dependencies\\n",
    "!pip install -r requirements.txt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "import pandas as pd\\n",
    "from data_loader import ArxivDataLoader, TextPreprocessor\\n",
    "\\n",
    "# Step 1: Load data from arXiv\\n",
    "loader = ArxivDataLoader()\\n",
    "df = loader.fetch_all_categories(papers_per_category=10000)\\n",
    "print(f'Loaded {len(df)} papers')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Step 2: Preprocess\\n",
    "preprocessor = TextPreprocessor()\\n",
    "df = preprocessor.prepare_dataset(df, 'data/processed/arxiv_papers_cleaned.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Step 3: Train BERT model\\n",
    "from bert_model import BertClassifier\\n",
    "\\n",
    "classifier = BertClassifier(model_name='roberta-base')\\n",
    "train_loader, val_loader, test_loader = classifier.prepare_data(df)\\n",
    "history = classifier.train(train_loader, val_loader, epochs=3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Step 4: Evaluate\\n",
    "results = classifier.test(test_loader)\\n",
    "print('✓ Model evaluation complete')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Step 5: Statistical analysis\\n",
    "from statistical_analysis import TopicEvolutionAnalysis\\n",
    "\\n",
    "analysis = TopicEvolutionAnalysis(df)\\n",
    "analysis.plot_topic_trends()\\n",
    "analysis.print_summary()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}"""
    
    with open('notebooks/01_full_pipeline.ipynb', 'w') as f:
        f.write(notebook_content)
    
    print("✓ Created example notebook: notebooks/01_full_pipeline.ipynb")

def print_next_steps():
    """Print next steps for user"""
    
    print("""
    
╔══════════════════════════════════════════════════════════════════════╗
║              arXiv Paper Classification - SETUP COMPLETE             ║
╚══════════════════════════════════════════════════════════════════════╝

📁 Project structure created successfully!

🚀 NEXT STEPS:

1. CREATE VIRTUAL ENVIRONMENT
   $ python -m venv venv
   $ source venv/bin/activate  # Linux/Mac
   $ venv\\Scripts\\activate     # Windows

2. INSTALL DEPENDENCIES
   $ pip install -r requirements.txt

3. FETCH & PREPARE DATA
   Option A (Recommended for first run - uses sample):
   $ python -c "
     from data_loader import ArxivDataLoader
     loader = ArxivDataLoader()
     df = loader.fetch_all_categories(papers_per_category=5000)
     loader.save_to_csv('data/raw/arxiv_papers.csv')
     "

   Option B (Full dataset - takes 1-2 hours):
   $ python -c "
     from data_loader import ArxivDataLoader
     loader = ArxivDataLoader()
     df = loader.fetch_all_categories(papers_per_category=20000)
     loader.save_to_csv('data/raw/arxiv_papers.csv')
     "

4. PREPROCESS DATA
   $ python -c "
     import pandas as pd
     from data_loader import TextPreprocessor
     df = pd.read_csv('data/raw/arxiv_papers.csv')
     preprocessor = TextPreprocessor()
     df = preprocessor.prepare_dataset(df, 'data/processed/arxiv_papers_cleaned.csv')
     "

5. TRAIN MODELS (GPU recommended)
   $ python bert_model.py
   Training time: ~10 hours for all 3 models

6. RUN ANALYSIS
   $ python statistical_analysis.py

7. LAUNCH DASHBOARD
   $ streamlit run streamlit_app.py
   Open: http://localhost:8501

📊 FILE STRUCTURE:

arxiv-paper-classification/
├── data/
│   ├── raw/              ← Downloaded arXiv papers
│   └── processed/        ← Cleaned data
├── models/               ← Trained BERT variants
├── results/              ← Analysis + visualizations
├── notebooks/            ← Jupyter notebooks
├── bert_model.py         ← Model training
├── data_loader.py        ← Data collection
├── statistical_analysis.py ← Trend analysis
├── streamlit_app.py      ← Interactive dashboard
└── requirements.txt

⏱️ ESTIMATED TIME:
├── Data collection:    2 hours
├── Preprocessing:      30 minutes
├── Model training:     10 hours (GPU)
├── Analysis:           1 hour
└── Total:              ~14 hours

💾 STORAGE REQUIREMENTS:
├── Raw data:           ~5 GB
├── Processed data:     ~2 GB
├── Models:             ~1 GB
└── Total:              ~8 GB

🔑 KEY FILES TO MODIFY:
1. data_loader.py       - Adjust arXiv categories/date ranges
2. bert_model.py        - Modify training hyperparameters
3. streamlit_app.py     - Customize dashboard layout
4. statistical_analysis.py - Add more trend tests

🌟 PRO TIPS:
✓ Start with smaller dataset (5K papers) to test pipeline
✓ Use GPU for model training (10x faster)
✓ Save model checkpoints frequently
✓ Monitor validation F1 to avoid overfitting
✓ Deploy to Streamlit Cloud for free hosting

📚 DOCUMENTATION:
- Full guide: README.md
- Model details: bert_model.py (docstrings)
- Data pipeline: data_loader.py (docstrings)
- Analysis methods: statistical_analysis.py (docstrings)

🔗 USEFUL LINKS:
- arXiv API Docs: https://arxiv.org/help/api
- HuggingFace Transformers: https://huggingface.co/docs
- Streamlit Docs: https://docs.streamlit.io/
- PyTorch Tutorial: https://pytorch.org/tutorials

💡 TROUBLESHOOTING:
- Low GPU memory? → Use DistilBERT or reduce batch size
- Slow data loading? → Increase number of API requests
- Training instability? → Lower learning rate or use warmup
- Poor results? → Check data quality and class balance

✅ Good luck! This is a publication-worthy project.

═══════════════════════════════════════════════════════════════════════
    """)

def main():
    """Main setup function"""
    
    print("╔════════════════════════════════════════════════════╗")
    print("║  arXiv Paper Classification - Project Setup        ║")
    print("╚════════════════════════════════════════════════════╝\n")
    
    print("Creating project structure...")
    create_directories()
    print()
    
    print("Creating configuration files...")
    create_gitignore()
    create_init_files()
    print()
    
    print("Creating example notebook...")
    create_example_notebook()
    print()
    
    print_next_steps()

if __name__ == "__main__":
    main()
