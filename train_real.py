import pandas as pd
from bert_model import BertClassifier

print("Loading REAL cleaned data...")
df = pd.read_csv('data/processed/real_arxiv_cleaned.csv')

print(f"Training on {len(df)} REAL arXiv papers...\n")

classifier = BertClassifier(model_name='distilbert-base-uncased')
train_loader, val_loader, test_loader = classifier.prepare_data(df)

print("\nTraining for 2 epochs...\n")
history = classifier.train(train_loader, val_loader, epochs=2)

print("\n" + "="*60)
print("EVALUATING ON REAL TEST SET")
print("="*60)
results = classifier.test(test_loader)

print("\n✓ Training on REAL data complete!")
