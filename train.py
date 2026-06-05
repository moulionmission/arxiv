import pandas as pd
from bert_model import BertClassifier

print("Loading cleaned data...")
df = pd.read_csv('data/processed/arxiv_cleaned.csv')

print(f"Training BERT classifier on {len(df)} papers...\n")

# Initialize classifier
classifier = BertClassifier(model_name='bert-base-uncased')

# Prepare data splits
print("Preparing train/val/test splits...")
train_loader, val_loader, test_loader = classifier.prepare_data(df)

# Train for 1 epoch (testing)
print("\nTraining for 1 epoch (this will take 5-10 minutes)...\n")
history = classifier.train(train_loader, val_loader, epochs=1)

# Evaluate
print("\n" + "="*60)
print("EVALUATING ON TEST SET")
print("="*60)
results = classifier.test(test_loader)

print("\n✓ Training complete!")
print("✓ Model saved to: models/bert-base-uncased_best.pt")
print("✓ Results saved")