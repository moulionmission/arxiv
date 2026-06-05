import pandas as pd
from data_loader import TextPreprocessor
from bert_model import BertClassifier

# Use NEW larger dataset
df = pd.read_csv('data/raw/real_arxiv_papers_1k.csv')
print(f'Loaded {len(df)} papers')

preprocessor = TextPreprocessor()
df_clean = preprocessor.prepare_dataset(df, 'data/processed/real_arxiv_1k_cleaned.csv')

# Train with more data
df_clean = pd.read_csv('data/processed/real_arxiv_1k_cleaned.csv')
classifier = BertClassifier(model_name='roberta-base')
train_loader, val_loader, test_loader = classifier.prepare_data(df_clean)

# More epochs because more data
history = classifier.train(train_loader, val_loader, epochs=8)

print("\n" + "="*60)
results = classifier.test(test_loader)
print("✓ Complete!")
