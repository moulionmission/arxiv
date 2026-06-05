import pandas as pd
from data_loader import TextPreprocessor
import os

print("\n" + "="*70)
print("PREPROCESSING 8,019 PAPERS")
print("="*70)

df = pd.read_csv('data/raw/arxiv_master.csv')
print(f"Loaded {len(df)} papers\n")

os.makedirs('data/processed', exist_ok=True)

preprocessor = TextPreprocessor()
df_clean = preprocessor.prepare_dataset(df, 'data/processed/arxiv_8k_cleaned.csv')

print("\n✓ Preprocessing complete!")
print("Next: python train_8k.py")
print("="*70 + "\n")
