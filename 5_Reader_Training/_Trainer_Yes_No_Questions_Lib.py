import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertForSequenceClassification
from tqdm import tqdm
from datetime import datetime
import os
from  Datasets_Models import *

def load_dataset_yes_no(filepath):
    try:
        with open(filepath, 'r') as file:
            data = json.load(file)
            if any('type' in item and item['type'] == 'yesno' for item in data):
                filtered_data = [item for item in data if item.get('type') == 'yesno']
                return filtered_data
            return data
    except FileNotFoundError:
        print("The file was not found.")
    except json.JSONDecodeError:
        print("The file is not in proper JSON format.")
    except Exception as e:
        print(f"An error occurred: {e}")

def train_model(model, criterion, optimizer, scheduler, train_loader, num_epochs, device, result_file, validation_loader):
    best_accuracy = 0
    best_model_path = ""
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{num_epochs}"):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            if logits.shape[-1] == 1:
                loss = criterion(logits.view(-1), labels.float())
            else:
                loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Testing
        validate_model(model, validation_loader, device, result_file)
        scheduler.step(loss)  # Update learning rate.

        # Validation step
        accuracy = validate_model(model, validation_loader, device, result_file)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_path = os.path.join(os.path.dirname(result_file),
                                           f"best_model.pt")
            torch.save(model.state_dict(), best_model_path)

        print(f"Epoch {epoch + 1} Training Loss: {train_loss / len(train_loader)}")
        result_file.write(f"Epoch {epoch + 1} Training Loss: {train_loss / len(train_loader)}\n")
        return  best_model_path

def validate_model(model, validation_loader, device, result_file):
    model.eval()
    val_correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for batch in tqdm(validation_loader, desc="Testing"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # Check the number of output logits to determine classification type
            if logits.shape[-1] == 1:
                # Binary classification with single logit per instance
                preds = torch.sigmoid(logits.squeeze(-1)) >= 0.5
                preds = preds.long()  # Convert boolean to long type (0 or 1)
            else:
                # Multi-class classification
                preds = torch.argmax(logits, dim=1)

            val_correct_predictions += (preds == labels).sum().item()
            total_predictions += labels.size(0)

    test_acc = val_correct_predictions / total_predictions
    print(f'Validation Accuracy: {test_acc}')
    result_file.write(f'Validation Accuracy: {test_acc}\n')
    return  test_acc


def test_model(model, test_loader, device, result_file):
    model.eval()
    test_correct_predictions = 0
    total_predictions = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # Check the number of output logits to determine classification type
            if logits.shape[-1] == 1:
                # Binary classification with single logit per instance
                preds = torch.sigmoid(logits.squeeze(-1)) >= 0.5
                preds = preds.long()  # Convert boolean to long type (0 or 1)
            else:
                # Multi-class classification
                preds = torch.argmax(logits, dim=1)

            test_correct_predictions += (preds == labels).sum().item()
            total_predictions += labels.size(0)

    test_acc = test_correct_predictions / total_predictions
    print(f'Test Accuracy: {test_acc}')
    result_file.write(f'Test Accuracy: {test_acc}\n')
    return test_acc

label_map = {"yes": 1, "no": 0, "maybe": 2}

models = ["bert-base-uncased","cambridgeltl/SapBERT-from-PubMedBERT-fulltext","dmis-lab/biobert-v1.1","michiyasunaga/BioLinkBERT-base"]

datasets = ["BioASQ","PubmedQA"]
dataPath = "../Pre_Processed_Datasets/"

batch_size = 8
learning_rate = 3e-5
number_epoch = 5

# Those Parameters for learning rate schedular
factor = 0.1
patience = 100
device = 'cuda'