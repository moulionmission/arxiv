"""
Train BERT-base classifier on unbiased 10-year dataset using feature extraction.
Same fast approach as DistilBERT training — freeze backbone, extract embeddings, train only classifier head.
"""

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from bert_model import BertClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import seaborn as sns

print("\n" + "="*70)
print("TRAINING BERT-BASE ON 10-YEAR UNBIASED DATASET (FEATURE EXTRACTION)")
print("="*70 + "\n")

processed_path = 'data/processed/arxiv_10yr_10k_unbiased_cleaned.csv'

if not os.path.exists(processed_path):
    print(f"❌ Error: {processed_path} not found. Run preprocess_10yr_10k.py first.")
    exit(1)

np.random.seed(42)
torch.manual_seed(42)

df = pd.read_csv(processed_path)
print(f"Loaded {len(df)} cleaned papers.\n")

model_name = 'bert-base-uncased'
classifier = BertClassifier(model_name=model_name, device='cpu')
device = 'cpu'
print(f"Using device: {device} (forced for stability)\n")

df['label_id'] = df['label'].map(classifier.label2id)

labels = df['label_id'].values
indices = np.arange(len(df))

train_idx, temp_idx, train_labels, temp_labels = train_test_split(
    indices, labels, test_size=0.30, random_state=42, stratify=labels
)
val_idx, test_idx, val_labels, test_labels = train_test_split(
    temp_idx, temp_labels, test_size=0.50, random_state=42, stratify=temp_labels
)

print(f"Train size: {len(train_idx)} | Val size: {len(val_idx)} | Test size: {len(test_idx)}")

MAX_LENGTH = 128
print(f"\nTokenizing all abstracts (max_length={MAX_LENGTH})...")
texts = df['abstract_clean'].values.tolist()
encodings = classifier.tokenizer(
    texts,
    max_length=MAX_LENGTH,
    padding='max_length',
    truncation=True,
    return_tensors='pt'
)
print("Tokenization complete.")

def extract_embeddings(idx_list, name="dataset"):
    dataset = TensorDataset(
        encodings['input_ids'][idx_list],
        encodings['attention_mask'][idx_list]
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    embeddings = []
    classifier.model.eval()
    
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Extracting features ({name})"):
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            
            # Forward pass through BERT base model
            outputs = classifier.model.bert(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            # Use pooler output (CLS token through pooling layer) — standard for BERT classification
            pooled = outputs.pooler_output
            embeddings.append(pooled.cpu())
            
    return torch.cat(embeddings, dim=0)

print("\nRunning feature extraction through frozen BERT backbone...")
train_embeddings = extract_embeddings(train_idx, "train")
val_embeddings = extract_embeddings(val_idx, "val")
test_embeddings = extract_embeddings(test_idx, "test")
print(f"Feature extraction complete. Embedding dim: {train_embeddings.shape[1]}")

class FeatureDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        return self.embeddings[idx], torch.tensor(self.labels[idx], dtype=torch.long)

train_feat_dataset = FeatureDataset(train_embeddings, train_labels)
val_feat_dataset = FeatureDataset(val_embeddings, val_labels)

train_feat_loader = DataLoader(train_feat_dataset, batch_size=64, shuffle=True)
val_feat_loader = DataLoader(val_feat_dataset, batch_size=128, shuffle=False)

# BERT-base uses a single linear classifier layer (no pre_classifier like DistilBERT)
classifier_layer = classifier.model.classifier.to(device)

# Also add a dropout for regularization
dropout = nn.Dropout(0.1).to(device)

optimizer = torch.optim.AdamW(classifier_layer.parameters(), lr=1e-3, weight_decay=1e-2)
criterion = nn.CrossEntropyLoss()

print("\nTraining classification head for 10 epochs...")
best_val_f1 = 0
best_head_state = None
history = []

for epoch in range(10):
    classifier_layer.train()
    total_loss = 0
    
    for features, targets in train_feat_loader:
        optimizer.zero_grad()
        features, targets = features.to(device), targets.to(device)
        
        # BERT classifier: dropout -> linear
        x = dropout(features)
        logits = classifier_layer(x)
        
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    # Evaluate on validation
    classifier_layer.eval()
    val_preds = []
    val_targets = []
    
    with torch.no_grad():
        for features, targets in val_feat_loader:
            features = features.to(device)
            logits = classifier_layer(features)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            val_preds.extend(preds)
            val_targets.extend(targets.numpy())
            
    val_f1 = f1_score(val_targets, val_preds, average='macro')
    train_loss = total_loss / len(train_feat_loader)
    
    print(f"Epoch {epoch+1}/10 | Train Loss: {train_loss:.4f} | Val F1: {val_f1:.4f}")
    
    history.append({
        'epoch': epoch + 1,
        'train_loss': train_loss,
        'val_f1': val_f1
    })
    
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_head_state = {k: v.cpu().clone() for k, v in classifier_layer.state_dict().items()}

# Restore best head state
classifier_layer.load_state_dict({k: v.to(device) for k, v in best_head_state.items()})
print(f"\n✓ Best validation F1: {best_val_f1:.4f}")

# Save the full model
print("\nSaving complete model...")
os.makedirs('models', exist_ok=True)
final_model_path = f'models/{model_name.split("/")[-1]}_best.pt'
classifier.save_model(final_model_path)
print(f"✓ Saved full model to {final_model_path}")

# Evaluate on test set
print("\n" + "="*60)
print("EVALUATING ON TEST SET")
print("="*60)

classifier_layer.eval()
test_preds = []
test_targets_list = []

with torch.no_grad():
    for features, targets in DataLoader(FeatureDataset(test_embeddings, test_labels), batch_size=128, shuffle=False):
        features = features.to(device)
        logits = classifier_layer(features)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        test_preds.extend(preds)
        test_targets_list.extend(targets.numpy())

report = classification_report(
    test_targets_list,
    test_preds,
    target_names=[classifier.id2label[i] for i in range(classifier.num_classes)],
    digits=4
)
print(report)

# Save confusion matrix
cm = confusion_matrix(test_targets_list, test_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=[classifier.id2label[i] for i in range(classifier.num_classes)],
    yticklabels=[classifier.id2label[i] for i in range(classifier.num_classes)]
)
plt.title(f'Confusion Matrix - {model_name} (10yr Unbiased)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
os.makedirs('results', exist_ok=True)
cm_path = f'results/confusion_matrix_{model_name.split("/")[-1]}_unbiased.png'
plt.savefig(cm_path, dpi=300)
print(f"✓ Confusion matrix saved to {cm_path}")
plt.close()

# Quick sanity check
print("\n--- Sanity Check ---")
text = 'Stance detection remains challenging in low-resource languages. We present a cross-lingual framework using multilingual transformer representations aligned through contrastive learning.'
result = classifier.predict(text)
print(f"NLP test → {result['predicted_label']} ({result['confidence']*100:.1f}%)")

print(f"\n✓ COMPLETE!")
print("="*70 + "\n")
