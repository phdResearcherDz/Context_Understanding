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


def load_dataset(filepath):
    try:
        with open(filepath, 'r') as file:
            data = json.load(file)
            # Filter data to only include entries where the type is 'yesno'
            filtered_data = [item for item in data if item.get('type') == 'yesno']
            return filtered_data
    except FileNotFoundError:
        print("The file was not found.")
    except json.JSONDecodeError:
        print("The file is not in proper JSON format.")
    except Exception as e:
        print(f"An error occurred: {e}")


def main():
    result_folder = f"results/base_models_result"
    os.makedirs(result_folder, exist_ok=True)

    datasets = ["BioASQ"]

    dataPath = "./Pre_Processed_Datasets/"

    for dataset in datasets:

        train_data = load_dataset(dataPath + f"{dataset}/5_enhanced_formated/train_enhanced.json")
        val_data = load_dataset(dataPath + f"{dataset}/5_enhanced_formated/test_enhanced.json")
        test_data = load_dataset(dataPath + f"{dataset}/5_enhanced_formated/test_enhanced.json")
        models = ["bert-base-uncased","cambridgeltl/SapBERT-from-PubMedBERT-fulltext","dmis-lab/biobert-v1.1",
                  "michiyasunaga/BioLinkBERT-base"]

        # Define Dataset
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

        # Label Mapping
        label_map = {"yes": 1, "no": 0, "maybe": 2}

        for model_name in models:
            best_val_loss = float('inf')  # Best validation loss for early stopping
            patience = 10  # Number of epochs to wait before stopping
            num_epochs = 3

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

                print("**************** - " + model_name + " - " + dataset + "- ****************")
                # Tokenizer

                tokenizer = BertTokenizerFast.from_pretrained(model_name)

                train_encodings = tokenizer([(item["question"], item["context"] + " [SEP] " + item["enhanced_text"]) for item in train_data],
                                            truncation=True, padding=True, max_length=512)
                val_encodings = tokenizer([(item["question"], item["context"] + " [SEP] " + item["enhanced_text"]) for item in val_data],
                                          truncation=True, padding=True, max_length=512)
                test_encodings = tokenizer([(item["question"], item["context"] + " [SEP] " + item["enhanced_text"]) for item in test_data],
                                           truncation=True, padding=True, max_length=512)

                train_labels = [label_map[item["answer"]] for item in train_data]
                val_labels = [label_map[item["answer"]] for item in val_data]
                test_labels = [label_map[item["answer"]] for item in test_data]

                # Dataset
                train_dataset = MyDataset(train_encodings, train_labels)
                val_dataset = MyDataset(val_encodings, val_labels)
                test_dataset = MyDataset(test_encodings, test_labels)

                # DataLoader
                train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
                test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

                model = None
                criterion = None
                if dataset == "pubmedqa_hf":
                    # Model
                    criterion = nn.CrossEntropyLoss()
                    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)
                else:
                    # Model
                    criterion = nn.BCEWithLogitsLoss()
                    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=1)

                model.to('cuda')
                # Optimizer
                optimizer = optim.AdamW(model.parameters(), lr=3e-5)  # 3e-5
                scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=100, verbose=True)

                # # Total number of training steps
                total_steps = len(train_loader) * num_epochs

                # Training Loop
                for epoch in range(num_epochs):
                    model.train()
                    train_loss = 0
                    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{num_epochs}"):
                        optimizer.zero_grad()
                        input_ids = batch['input_ids'].to('cuda')
                        attention_mask = batch['attention_mask'].to('cuda')
                        labels = batch['labels'].to('cuda')

                        outputs = model(input_ids, attention_mask=attention_mask)
                        logits = outputs.logits

                        if dataset == "pubmedqa_hf":
                            # Compute loss for multi-class classification
                            loss = criterion(logits, labels)
                        else:
                            # Compute loss for binary classification
                            loss = criterion(logits.view(-1), labels.float())

                        loss.backward()
                        optimizer.step()

                        train_loss += loss.item()
                    result_file.write(f"Epoch {epoch + 1} Training Loss: {train_loss / len(train_loader)}\n")
                    print(f"Epoch {epoch + 1} Training Loss: {train_loss / len(train_loader)}")
                    scheduler.step(loss)  # Update the learning rate.

                    # Evaluation Loop for Validation set
                    model.eval()
                    val_loss = 0
                    val_correct_predictions = 0
                    total_predictions = 0
                    with torch.no_grad():
                        for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}/{num_epochs}"):
                            input_ids = batch['input_ids'].to('cuda')
                            attention_mask = batch['attention_mask'].to('cuda')
                            labels = batch['labels'].to('cuda')

                            # Forward pass
                            outputs = model(input_ids, attention_mask=attention_mask)
                            logits = outputs.logits

                            if dataset == "pubmedqa_hf":
                                # Compute loss for multi-class classification
                                loss = criterion(logits, labels)
                            else:
                                # Compute loss for binary classification
                                # Reshape logits and cast labels to float for BCEWithLogitsLoss
                                loss = criterion(logits.view(-1), labels.float())

                            val_loss += loss.item()

                            if dataset == "pubmedqa_hf":
                                # Use argmax for multi-class classification to get predictions
                                preds = torch.argmax(logits, dim=1)
                            else:
                                # For binary classification, use a threshold (e.g., 0.5) to convert logits to predictions
                                preds = torch.sigmoid(logits).view(-1) > 0.5

                            val_correct_predictions += (preds == labels).sum().item()
                            total_predictions += labels.size(0)
                    # Calculate validation accuracy
                    val_accuracy = val_correct_predictions / total_predictions
                    # Calculate the average validation loss
                    val_loss /= len(val_loader)

                    # Display validation loss and accuracy
                    print(f'Epoch {epoch + 1} Validation Loss: {val_loss}')
                    print(f'Epoch {epoch + 1} Validation Accuracy: {val_accuracy}')
                    result_file.write(f'Epoch {epoch + 1} Validation Loss: {val_loss}\n')
                    result_file.write(f'Epoch {epoch + 1} Validation Accuracy: {val_accuracy}\n')

                    # Step the scheduler based on validation loss
                    scheduler.step(val_loss)

                    # Early stopping logic based on validation loss
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        early_stop_counter = 0  # Reset the counter
                        # Save the best model
                        best_model_path = os.path.join(timestamp_folder, "best_model.pt")
                        torch.save(model.state_dict(), best_model_path)
                    else:
                        early_stop_counter += 1

                    # Uncomment below lines if you want to use early stopping
                    if early_stop_counter >= patience:
                        print("Early stopping")
                        break  # Break the training loop

                # Test Loop
                model.load_state_dict(torch.load(best_model_path))
                model.eval()
                test_acc = 0
                with torch.no_grad():
                    for batch in tqdm(test_loader, desc="Testing"):
                        input_ids = batch['input_ids'].to('cuda')
                        attention_mask = batch['attention_mask'].to('cuda')
                        labels = batch['labels'].to('cuda')

                        # Forward pass
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

                        test_acc += (preds == labels).sum().item()

                test_acc = test_acc / len(test_dataset)
                print(f'Test Accuracy: {test_acc}')
                print("************** - End - *****************")
                result_file.write(f'Test Accuracy: {test_acc}\n')
                result_file.write("************** - End - *****************\n")

                # Save test accuracy to a separate file for accumulating results
                with open(os.path.join(result_folder, "test_accuracies.txt"), "a") as acc_file:
                    acc_file.write(f"{model_name},{dataset},{test_acc}\n")


if __name__ == '__main__':
    main()
