import pandas as pd
from data_loader import TextPreprocessor

df = pd.read_csv('data/raw/sample_papers.csv')
print(f'Loaded {len(df)} papers')

preprocessor = TextPreprocessor()
df_clean = preprocessor.prepare_dataset(df, 'data/processed/arxiv_cleaned.csv')

print(f'✓ Preprocessing complete!')