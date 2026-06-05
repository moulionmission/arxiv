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
print("TRAINING ON 10-YEAR UNBIASED DATASET (FEATURE EXTRACTION MODE)")
print("="*70 + "\n")

processed_path = 'data/processed/arxiv_10yr_10k_unbiased_cleaned.csv'

if not os.path.exists(processed_path):
    print(f"❌ Error: Processed data file {processed_path} not found. Run preprocess_10yr_10k.py first.")
    exit(1)

np.random.seed(42)
torch.manual_seed(42)

df = pd.read_csv(processed_path)
print(f"Loaded {len(df)} cleaned papers.\n")

# Use distilbert-base-uncased (faster and lighter, perfect for CPU/MPS feature extraction)
model_name = 'distilbert-base-uncased'
classifier = BertClassifier(model_name=model_name)
device = 'cpu'  # Force CPU for absolute stability and to avoid MPS compilation deadlocks
classifier.model.to(device)
print(f"Using device: {device} (forced for stability)")

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

# Reduce max_length to 128 (arxiv abstracts average ~150 words, so 128 is perfect and speeds up extraction by 16x)
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
    # CPU batch size 128 works very well
    loader = DataLoader(dataset, batch_size=128, shuffle=False)
    
    embeddings = []
    classifier.model.eval()
    
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Extracting features ({name})"):
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            
            # Forward pass through base distilbert model
            outputs = classifier.model.distilbert(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            # Take CLS token representation (first token)
            sequence_output = outputs[0]
            cls_repr = sequence_output[:, 0, :]
            
            embeddings.append(cls_repr.cpu())
            
    return torch.cat(embeddings, dim=0)

print("\nRunning feature extraction through frozen backbone...")
train_embeddings = extract_embeddings(train_idx, "train")
val_embeddings = extract_embeddings(val_idx, "val")
test_embeddings = extract_embeddings(test_idx, "test")
print("Feature extraction complete.")

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

# Get classifier layers for distilbert
pre_classifier = classifier.model.pre_classifier.to(device)
classifier_layer = classifier.model.classifier.to(device)

# We optimize both layers of the classifier head
params = list(pre_classifier.parameters()) + list(classifier_layer.parameters())
optimizer = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-2)
criterion = nn.CrossEntropyLoss()

print("\nTraining classification head for 10 epochs...")
best_val_f1 = 0
best_head_state = None
history = []

for epoch in range(10):
    pre_classifier.train()
    classifier_layer.train()
    total_loss = 0
    
    for features, targets in train_feat_loader:
        optimizer.zero_grad()
        features, targets = features.to(device), targets.to(device)
        
        # DistilBERT classifier head forward pass
        x = pre_classifier(features)
        x = torch.nn.functional.relu(x)
        logits = classifier_layer(x)
        
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    # Evaluate validation split
    pre_classifier.eval()
    classifier_layer.eval()
    val_preds = []
    val_targets = []
    
    with torch.no_grad():
        for features, targets in val_feat_loader:
            features = features.to(device)
            x = pre_classifier(features)
            x = torch.nn.functional.relu(x)
            logits = classifier_layer(x)
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
        best_head_state = {
            'pre_classifier': {k: v.cpu().clone() for k, v in pre_classifier.state_dict().items()},
            'classifier': {k: v.cpu().clone() for k, v in classifier_layer.state_dict().items()}
        }

# Restore best head state
pre_classifier.load_state_dict({k: v.to(device) for k, v in best_head_state['pre_classifier'].items()})
classifier_layer.load_state_dict({k: v.to(device) for k, v in best_head_state['classifier'].items()})
print(f"\n✓ Best validation F1: {best_val_f1:.4f}")

# Save the final full model
print("\nSaving complete Hugging Face classifier model...")
os.makedirs('models', exist_ok=True)
final_model_path = f'models/{model_name.split("/")[-1]}_10yr_10k_unbiased_best.pt'
classifier.save_model(final_model_path)
print(f"✓ Saved full model to {final_model_path}")

# Evaluate on test set
print("\n" + "="*60)
print("EVALUATING ON TEST SET")
print("="*60)

pre_classifier.eval()
classifier_layer.eval()
test_preds = []
test_targets = []

with torch.no_grad():
    for features, targets in DataLoader(FeatureDataset(test_embeddings, test_labels), batch_size=128, shuffle=False):
        features = features.to(device)
        x = pre_classifier(features)
        x = torch.nn.functional.relu(x)
        logits = classifier_layer(x)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        test_preds.extend(preds)
        test_targets.extend(targets.numpy())

report = classification_report(
    test_targets,
    test_preds,
    target_names=[classifier.id2label[i] for i in range(classifier.num_classes)],
    digits=4
)
print(report)

# Save confusion matrix
cm = confusion_matrix(test_targets, test_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=[classifier.id2label[i] for i in range(classifier.num_classes)],
    yticklabels=[classifier.id2label[i] for i in range(classifier.num_classes)]
)
plt.title(f'Confusion Matrix - {model_name} (10yr 10k Unbiased)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
os.makedirs('results', exist_ok=True)
cm_path = f'results/confusion_matrix_{model_name.split("/")[-1]}_10yr_10k_unbiased.png'
plt.savefig(cm_path, dpi=300)
print(f"✓ Confusion matrix saved to {cm_path}")
plt.close()

print(f"\n✓ COMPLETE!")
print("="*70 + "\n")
