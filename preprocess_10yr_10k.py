import pandas as pd
from data_loader import TextPreprocessor
import os

print("\n" + "="*70)
print("PREPROCESSING 10-YEAR 10,000 PAPERS UNBIASED DATASET")
print("="*70 + "\n")

raw_path = 'data/raw/arxiv_10yr_10k_unbiased.csv'
processed_path = 'data/processed/arxiv_10yr_10k_unbiased_cleaned.csv'

if not os.path.exists(raw_path):
    print(f"❌ Error: Raw file {raw_path} not found.")
    exit(1)

df = pd.read_csv(raw_path)
print(f"Loaded {len(df)} raw papers.")

os.makedirs('data/processed', exist_ok=True)

preprocessor = TextPreprocessor()
df_clean = preprocessor.prepare_dataset(df, processed_path)

print(f"\nCleaned dataset details:")
print(f"Total papers: {len(df_clean)}")
print(f"Class distribution:")
print(df_clean['label'].value_counts())
print("\n✓ Preprocessing complete!")
print(f"Output saved to: {processed_path}")
print("="*70 + "\n")
