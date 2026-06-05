import pandas as pd
from bert_model import BertClassifier

print("\n" + "="*70)
print("TRAINING ON 8,019 PAPERS")
print("="*70 + "\n")

df = pd.read_csv('data/processed/arxiv_8k_cleaned.csv')
print(f"Training on {len(df)} papers\n")

classifier = BertClassifier(model_name='roberta-base')
train_loader, val_loader, test_loader = classifier.prepare_data(df)

print("\nTraining for 10 epochs...\n")
history = classifier.train(train_loader, val_loader, epochs=10)

print("\n" + "="*70)
print("EVALUATING ON TEST SET")
print("="*70)
results = classifier.test(test_loader)

print("\n✓ TRAINING COMPLETE ON 8,019 PAPERS!")
print("="*70 + "\n")
