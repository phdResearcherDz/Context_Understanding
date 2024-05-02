import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertForSequenceClassification, get_linear_schedule_with_warmup
from tqdm import tqdm
from datetime import datetime
import os
from  Datasets_Models import *

def load_dataset_yes_no(filepath):
    try:
        with open(filepath, 'r') as file:
            data = json.load(file)
            filtered_data = [item for item in data if item.get('answer') == 'yes' or item.get('answer') == 'no']
            return filtered_data

    except FileNotFoundError:
        print("The file was not found.")
    except json.JSONDecodeError:
        print("The file is not in proper JSON format.")
    except Exception as e:
        print(f"An error occurred: {e}")

def train_model(model, criterion, optimizer, scheduler, train_loader, num_epochs, device, result_file, validation_loader):
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=total_steps)

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

            loss = criterion(logits.view(-1), labels.float())

            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

        # Testing

        # Validation step
        accuracy = validate_model(model, validation_loader, device, result_file)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_path = os.path.join(os.path.dirname(result_file.name),
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
        for batch in tqdm(validation_loader, desc="Validation"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # Binary classification with single logit per instance
            preds = torch.sigmoid(logits.squeeze(-1)) >= 0.5
            preds = preds.long()  # Convert boolean to long type (0 or 1)

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

            # Binary classification with single logit per instance
            preds = torch.sigmoid(logits.squeeze(-1)) >= 0.5
            preds = preds.long()  # Convert boolean to long type (0 or 1)

            test_correct_predictions += (preds == labels).sum().item()
            total_predictions += labels.size(0)

    test_acc = test_correct_predictions / total_predictions
    print(f'Test Accuracy: {test_acc}')
    result_file.write(f'Test Accuracy: {test_acc}\n')
    return test_acc

label_map = {"yes": 1, "no": 0, "maybe": 2}
#"Tolerblanc/biogpt_Readmission"
models = ["dmis-lab/biobert-v1.1"]#"cambridgeltl/SapBERT-from-PubMedBERT-fulltext",,"michiyasunaga/BioLinkBERT-base"

datasets = ["PubmedQA"] #,"BioASQ"
dataPath = "./Pre_Processed_Datasets/"

batch_size = 8
learning_rate = 3e-5 #2e-5
number_epoch = 5

# Those Parameters for learning rate schedular
factor = 0.1
patience = 100
device = 'cuda'