import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, get_scheduler
from transformers import logging
import torch.nn.functional as F

logging.set_verbosity_error()

class MentalHealthDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def preprocess_text(text):
    # Basic preprocessing: lowercasing and removing punctuation
    import re
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

def train_epoch(model, data_loader, optimizer, device, scheduler):
    model = model.train()
    losses = []
    correct_predictions = 0
    total_batches = len(data_loader)

    for batch_idx, batch in enumerate(data_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss
        logits = outputs.logits

        _, preds = torch.max(logits, dim=1)
        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        if batch_idx % 10 == 0:
            print(f'Batch {batch_idx+1}/{total_batches} - Loss: {loss.item():.4f}')

    return correct_predictions.double() / len(data_loader.dataset), np.mean(losses)

def eval_model(model, data_loader, device):
    model = model.eval()
    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            logits = outputs.logits

            _, preds = torch.max(logits, dim=1)
            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())

    return correct_predictions.double() / len(data_loader.dataset), np.mean(losses)

def main(args):
    # Load dataset
    df = pd.read_csv(args.data_path)

    # Preprocess text
    df['text'] = df['text'].apply(preprocess_text)

    # Map labels to integers
    label_list = [
        'depression',
        'anxiety',
        'stress',
        'bipolar disorder',
        'eating disorder',
        'obsessive-compulsive disorder',
        'post-traumatic stress disorder'
    ]
    label_map = {label: idx for idx, label in enumerate(label_list)}
    df['label'] = df['label'].map(label_map)

    # Drop rows with missing labels or texts
    df = df.dropna(subset=['text', 'label'])

    # Split dataset
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['text'].values,
        df['label'].values,
        test_size=0.2,
        random_state=42,
        stratify=df['label'].values
    )

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    max_len = args.max_len

    train_dataset = MentalHealthDataset(train_texts, train_labels, tokenizer, max_len)
    val_dataset = MentalHealthDataset(val_texts, val_labels, tokenizer, max_len)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=len(label_list)
    )
    model = model.to(device)

    # Compute class weights to handle imbalance
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    # Modify loss function to use class weights
    def weighted_loss(outputs, labels):
        loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)
        return loss_fct(outputs, labels)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    best_accuracy = 0

    for epoch in range(args.epochs):
        print(f'Epoch {epoch + 1}/{args.epochs}')
        train_acc, train_loss = train_epoch(model, train_loader, optimizer, device, scheduler)
        print(f'Train loss {train_loss:.4f} accuracy {train_acc:.4f}')
        val_acc, val_loss = eval_model(model, val_loader, device)
        print(f'Val   loss {val_loss:.4f} accuracy {val_acc:.4f}')

        if val_acc > best_accuracy:
            best_accuracy = val_acc
            # Save the best model
            output_dir = args.output_dir
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            print(f'Saved best model to {output_dir}')

    print('Training complete.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train mental health text classification model')
    parser.add_argument('--data_path', type=str, required=True, help='Path to CSV file with text and label columns')
    parser.add_argument('--output_dir', type=str, default='./model', help='Directory to save the trained model')
    parser.add_argument('--max_len', type=int, default=256, help='Maximum sequence length for BERT tokenizer')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--gradient_clip', type=float, default=1.0, help='Gradient clipping max norm')
    parser.add_argument('--early_stopping_patience', type=int, default=3, help='Early stopping patience epochs')
    parser.add_argument('--mixed_precision', action='store_true', help='Use mixed precision training if CUDA available')
    args = parser.parse_args()

    main(args)
