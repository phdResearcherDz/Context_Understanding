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


def load_dataset(filepath):
    try:
        with open(filepath, 'r') as file:
            data = json.load(file)
            filtered_data = [item for item in data if item.get('type') == 'yesno']
            return filtered_data
    except FileNotFoundError:
        print("The file was not found.")
    except json.JSONDecodeError:
        print("The file is not in proper JSON format.")
    except Exception as e:
        print(f"An error occurred: {e}")


class MyDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def train_model(model, criterion, optimizer, scheduler, train_loader, num_epochs, device, result_file,validation_loader):
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
        print(f"Epoch {epoch + 1} Training Loss: {train_loss / len(train_loader)}")
        result_file.write(f"Epoch {epoch + 1} Training Loss: {train_loss / len(train_loader)}\n")
        # Testing
        validate_model(model, validation_loader, device, result_file)
        scheduler.step(loss)  # Update learning rate.


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


def main():
    result_folder = f"results/base_models_result"
    os.makedirs(result_folder, exist_ok=True)
    datasets = ["BioASQ"]
    dataPath = "../Pre_Processed_Datasets/"

    for dataset in datasets:
        train_data = load_dataset(dataPath + f"{dataset}/train.json")
        val_data = load_dataset(dataPath + f"{dataset}/test.json")
        test_data = load_dataset(dataPath + f"{dataset}/test.json")
        models = ["bert-base-uncased","cambridgeltl/SapBERT-from-PubMedBERT-fulltext","dmis-lab/biobert-v1.1","michiyasunaga/BioLinkBERT-base"]

        label_map = {"yes": 1, "no": 0, "maybe": 2}

        for model_name in models:
            num_epochs = 3
            device = 'cuda'

            model_folder = os.path.join(result_folder, model_name.replace("/", "_"))
            os.makedirs(model_folder, exist_ok=True)
            dataset_folder = os.path.join(model_folder, dataset)
            os.makedirs(dataset_folder, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            timestamp_folder = os.path.join(dataset_folder, timestamp)
            os.makedirs(timestamp_folder, exist_ok=True)
            result_file_path = os.path.join(dataset_folder, "results.txt")

            with open(result_file_path, "w") as result_file:
                result_file.write(f"**************** - {model_name} {dataset}- ****************\n")
                tokenizer = BertTokenizerFast.from_pretrained(model_name)
                train_encodings = tokenizer([(item["question"], item["context"]) for item in train_data],
                                            truncation=True, padding=True, max_length=512)
                test_encodings = tokenizer([(item["question"], item["context"]) for item in test_data],
                                           truncation=True, padding=True, max_length=512)

                train_labels = [label_map[item["answer"]] for item in train_data]
                test_labels = [label_map[item["answer"]] for item in test_data]

                train_dataset = MyDataset(train_encodings, train_labels)
                test_dataset = MyDataset(test_encodings, test_labels)

                train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
                test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

                if dataset == "PubmedQA":
                    # Model
                    criterion = nn.CrossEntropyLoss()
                    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)
                else:
                    # Model
                    criterion = nn.BCEWithLogitsLoss()
                    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=1)

                optimizer = optim.AdamW(model.parameters(), lr=3e-5)
                model.to(device)
                scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=100, verbose=True)

                # Training
                train_model(model, criterion, optimizer, scheduler, train_loader, num_epochs, device, result_file,test_loader)

                # Testing
                test_model(model, test_loader, device, result_file)


if __name__ == '__main__':
    main()
