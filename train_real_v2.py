import pandas as pd
from bert_model import BertClassifier

df = pd.read_csv('data/processed/real_arxiv_cleaned.csv')
print(f"Training on {len(df)} REAL papers\n")

# Use RoBERTa (stronger than DistilBERT)
classifier = BertClassifier(model_name='roberta-base')
train_loader, val_loader, test_loader = classifier.prepare_data(df)

# 5 epochs = better learning
history = classifier.train(train_loader, val_loader, epochs=5)

print("\n" + "="*60)
results = classifier.test(test_loader)
print("✓ Complete!")
