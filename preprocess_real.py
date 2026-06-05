import pandas as pd
from data_loader import TextPreprocessor

df = pd.read_csv('data/raw/real_arxiv_papers.csv')
print(f'Loaded {len(df)} REAL papers from arXiv\n')

preprocessor = TextPreprocessor()
df_clean = preprocessor.prepare_dataset(df, 'data/processed/real_arxiv_cleaned.csv')

print(f'\n✓ Real data ready for training!')
