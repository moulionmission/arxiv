"""
arXiv Paper Classification: BERT Fine-tuning Model
Train BERT, RoBERTa, and DistilBERT on abstract classification task
"""

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
from datetime import datetime

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)


class ArxivDataset(Dataset):
    """PyTorch Dataset for arXiv papers"""
    
    def __init__(self, 
                 texts: list,
                 labels: list,
                 tokenizer,
                 max_length: int = 512):
        """
        Args:
            texts: List of abstract strings
            labels: List of label indices
            tokenizer: HuggingFace tokenizer
            max_length: Max token length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class BertClassifier:
    """BERT-based paper classifier"""
    
    def __init__(self, 
                 model_name: str = 'bert-base-uncased',
                 num_classes: int = 5,
                 device: str = None):
        """
        Args:
            model_name: HuggingFace model identifier
            num_classes: Number of output classes
            device: 'cuda' or 'cpu'
        """
        
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = 'cuda'
        elif torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'
            
        print(f"Using device: {self.device}")
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_classes
        ).to(self.device)
        
        # Label mapping
        self.label2id = {
            'AI': 0,
            'ML': 1,
            'NLP': 2,
            'Computer Vision': 3,
            'Robotics': 4
        }
        self.id2label = {v: k for k, v in self.label2id.items()}
        
        self.training_history = []
        
    def freeze_backbone(self, freeze: bool = True, num_trainable_layers: int = 0):
        """
        Freeze base model parameters to speed up training.
        If num_trainable_layers > 0, the last N layers of the transformer remain trainable.
        """
        if not freeze:
            for param in self.model.parameters():
                param.requires_grad = True
            print("Backbone unfrozen. All parameters are trainable.")
            return
            
        # First, freeze everything in the base model
        for param in self.model.base_model.parameters():
            param.requires_grad = False
            
        # Unfreeze classification head(s) and any other layer not part of the base model backbone
        for name, param in self.model.named_parameters():
            if not name.startswith("base_model") and not name.startswith(self.model.base_model_prefix):
                param.requires_grad = True
                
        # Specifically ensure classification heads are unfrozen
        if hasattr(self.model, "classifier"):
            for param in self.model.classifier.parameters():
                param.requires_grad = True
        if hasattr(self.model, "pre_classifier"):
            for param in self.model.pre_classifier.parameters():
                param.requires_grad = True
                
        # Optionally unfreeze top N encoder blocks
        if num_trainable_layers > 0:
            layers = []
            if hasattr(self.model.base_model, "encoder") and hasattr(self.model.base_model.encoder, "layer"):
                layers = list(self.model.base_model.encoder.layer)
            elif hasattr(self.model.base_model, "transformer") and hasattr(self.model.base_model.transformer, "layer"):
                layers = list(self.model.base_model.transformer.layer)
                
            if layers:
                num_layers = len(layers)
                trainable_start_idx = max(0, num_layers - num_trainable_layers)
                print(f"Unfreezing the top {num_trainable_layers} layers of the transformer (layers {trainable_start_idx} to {num_layers-1})")
                for layer in layers[trainable_start_idx:]:
                    for param in layer.parameters():
                        param.requires_grad = True
            else:
                print("⚠️ Warning: Could not locate transformer layers list. Backbone is fully frozen.")
                
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.model.parameters() if not p.requires_grad)
        print(f"Model parameters: Trainable = {trainable:,} | Frozen = {frozen:,}")
    
    def prepare_data(self, df: pd.DataFrame, test_size: float = 0.15):
        """
        Prepare train/val/test splits
        """
        
        # Convert labels to indices
        df['label_id'] = df['label'].map(self.label2id)
        
        # Convert to numpy arrays to avoid pandas ArrowExtensionArray issues
        texts = np.array(df['abstract_clean'].values, dtype=object)
        labels = np.array(df['label_id'].values, dtype=int)
        
        # Train + Val split (85%)
        train_texts, temp_texts, train_labels, temp_labels = train_test_split(
            texts,
            labels,
            test_size=test_size * 2,
            random_state=42,
            stratify=labels
        )
        
        # Val + Test split (50/50 of remaining)
        val_texts, test_texts, val_labels, test_labels = train_test_split(
            temp_texts,
            temp_labels,
            test_size=0.5,
            random_state=42,
            stratify=temp_labels
        )
        
        print(f"Train: {len(train_texts)} | Val: {len(val_texts)} | Test: {len(test_texts)}")
        
        # Create datasets
        train_dataset = ArxivDataset(train_texts, train_labels, self.tokenizer)
        val_dataset = ArxivDataset(val_texts, val_labels, self.tokenizer)
        test_dataset = ArxivDataset(test_texts, test_labels, self.tokenizer)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        self.test_labels = test_labels
        
        return train_loader, val_loader, test_loader
    
    def train_epoch(self, train_loader, optimizer, scheduler):
        """Train for one epoch"""
        
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    
    def evaluate(self, val_loader):
        """Evaluate on validation set"""
        
        self.model.eval()
        total_loss = 0
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                predictions.extend(preds)
                true_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        f1 = f1_score(true_labels, predictions, average='macro')
        
        return avg_loss, f1, predictions, true_labels
    
    def train(self, 
              train_loader,
              val_loader,
              epochs: int = 3,
              learning_rate: float = 2e-5):
        """
        Full training loop
        
        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            epochs: Number of epochs
            learning_rate: Learning rate
        """
        
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = AdamW(trainable_params, lr=learning_rate)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )
        
        best_f1 = 0
        
        for epoch in range(epochs):
            print(f"\n--- Epoch {epoch + 1}/{epochs} ---")
            
            train_loss = self.train_epoch(train_loader, optimizer, scheduler)
            val_loss, val_f1, _, _ = self.evaluate(val_loader)
            
            self.training_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_f1': val_f1
            })
            
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")
            
            if val_f1 > best_f1:
                best_f1 = val_f1
                self.save_model(f'models/{self.model_name.split("/")[-1]}_best.pt')
                print(f"✓ Model saved (F1: {best_f1:.4f})")
        
        return self.training_history
    
    def test(self, test_loader):
        """Evaluate on test set with detailed metrics"""
        
        _, _, predictions, true_labels = self.evaluate(test_loader)
        
        print("\n" + "="*60)
        print("TEST SET RESULTS")
        print("="*60)
        
        # Classification report
        report = classification_report(
            true_labels,
            predictions,
            target_names=[self.id2label[i] for i in range(self.num_classes)],
            digits=4
        )
        print(report)
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        self._plot_confusion_matrix(cm)
        
        return {
            'predictions': predictions,
            'true_labels': true_labels,
            'confusion_matrix': cm.tolist()
        }
    
    def _plot_confusion_matrix(self, cm):
        """Plot confusion matrix"""
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=[self.id2label[i] for i in range(self.num_classes)],
            yticklabels=[self.id2label[i] for i in range(self.num_classes)]
        )
        plt.title(f'Confusion Matrix - {self.model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'results/confusion_matrix_{self.model_name.split("/")[-1]}.png', dpi=300)
        print(f"✓ Confusion matrix saved")
        plt.close()
    
    def save_model(self, filepath: str):
        """Save model checkpoint"""
        torch.save(self.model.state_dict(), filepath)
    
    def load_model(self, filepath: str):
        """Load model checkpoint"""
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
    
    def predict(self, text: str) -> dict:
        """Predict on new text"""
        
        self.model.eval()
        
        encoding = self.tokenizer(
            text,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            pred_id = np.argmax(probs)
        
        return {
            'predicted_label': self.id2label[pred_id],
            'confidence': float(probs[pred_id]),
            'all_scores': {
                self.id2label[i]: float(probs[i])
                for i in range(self.num_classes)
            }
        }


# Training script
if __name__ == "__main__":
    
    print("Loading data...")
    df = pd.read_csv('data/processed/arxiv_papers_cleaned.csv')
    
    # Initialize models
    models_to_train = [
        'bert-base-uncased',
        'roberta-base',
        'distilbert-base-uncased'
    ]
    
    results = {}
    
    for model_name in models_to_train:
        print(f"\n{'='*60}")
        print(f"Training {model_name}")
        print(f"{'='*60}")
        
        classifier = BertClassifier(model_name=model_name)
        train_loader, val_loader, test_loader = classifier.prepare_data(df)
        
        # Train
        history = classifier.train(train_loader, val_loader, epochs=3)
        
        # Test
        test_results = classifier.test(test_loader)
        results[model_name] = test_results
        
        # Save results
        with open(f'results/{model_name}_results.json', 'w') as f:
            json.dump(test_results, f, indent=2)
    
    print("\nTraining complete!")
