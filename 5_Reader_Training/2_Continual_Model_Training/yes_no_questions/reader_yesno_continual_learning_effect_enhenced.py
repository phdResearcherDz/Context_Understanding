import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertForSequenceClassification, get_linear_schedule_with_warmup
from tqdm import tqdm
from datetime import datetime
import os
from sklearn.model_selection import train_test_split


def main():
    result_folder = f"results/base_models_result_continual_learning_enhanced"
    os.makedirs(result_folder, exist_ok=True)
    for dataset in datasets:

        full_train_data = load_dataset_yes_no(dataPath + f"{dataset}/5_enhanced_formated/train_enhanced.json")

        # Splitting the training data into two equal parts
        first_half_train_data, second_half_train_data = train_test_split(full_train_data, test_size=0.5,
                                                                         random_state=42)
        # Further splitting the second half into four parts
        quarter_batches = np.array_split(second_half_train_data, 4)


        val_data = load_dataset_yes_no(dataPath + f"{dataset}/5_enhanced_formated/test_enhanced.json")
        test_data = load_dataset_yes_no(dataPath + f"{dataset}/5_enhanced_formated/test_enhanced.json")
        # model_name = 'michiyasunaga/BioLinkBERT-base'

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
                train_encodings = tokenizer([(item["question"], item["context"] + " [SEP] " + item["enhanced_text"]) for item in first_half_train_data],
                                            truncation=True, padding=True, max_length=512)
                val_encodings = tokenizer([(item["question"], item["context"] + " [SEP] " + item["enhanced_text"]) for item in val_data],
                                          truncation=True, padding=True, max_length=512)
                test_encodings = tokenizer([(item["question"], item["context"] + " [SEP] " + item["enhanced_text"]) for item in test_data],
                                           truncation=True, padding=True, max_length=512)
                train_labels = [label_map[item["answer"]] for item in first_half_train_data]
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

                print('First Part Training')
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
                print(f'Test First Half Loss: {val_loss}')
                print(f'Test First Half Accuracy: {val_accuracy}')
                result_file.write(f'Test First Half loss: {val_loss}\n')
                result_file.write(f'Test First Half Accuracy: {val_accuracy}\n')

                # Step the scheduler based on validation loss
                scheduler.step(val_loss)

                print('Continual Learning Part')
                # Continual finetuning
                for i, quarter_data in enumerate(quarter_batches):
                    quarter_encodings = tokenizer([(item["question"], item["context"] + " [SEP] " + item["enhanced_text"]) for item in quarter_data],
                                                  truncation=True, padding=True, max_length=512)
                    quarter_labels = [label_map[item["answer"]] for item in quarter_data]
                    quarter_dataset = MyDataset(quarter_encodings, quarter_labels)
                    quarter_loader = DataLoader(quarter_dataset, batch_size=8, shuffle=True)
                    for epoch in range(num_epochs):
                        model.train()
                        train_loss = 0
                        for batch in tqdm(quarter_loader, desc=f"Part {i} - Training Epoch {epoch + 1}/{num_epochs}"):
                            optimizer.zero_grad()
                            input_ids = batch['input_ids'].to('cuda')
                            attention_mask = batch['attention_mask'].to('cuda')
                            labels = batch['labels'].to('cuda')

                            outputs = model(input_ids, attention_mask=attention_mask)
                            logits = outputs.logits

                            if dataset == "pubmedqa_hf":
                                loss = criterion(logits, labels)
                            else:
                                loss = criterion(logits.view(-1), labels.float())

                            loss.backward()
                            optimizer.step()

                            train_loss += loss.item()
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
                                loss = criterion(logits, labels)
                            else:
                                loss = criterion(logits.view(-1), labels.float())

                            val_loss += loss.item()

                            if dataset == "pubmedqa_hf":
                                preds = torch.argmax(logits, dim=1)
                            else:
                                preds = torch.sigmoid(logits).view(-1) > 0.5

                            val_correct_predictions += (preds == labels).sum().item()
                            total_predictions += labels.size(0)
                    # Calculate validation accuracy
                    val_accuracy = val_correct_predictions / total_predictions
                    # Calculate the average validation loss
                    val_loss /= len(val_loader)

                    # Display validation loss and accuracy
                    result_file.write(f'Part {i+1} Test Loss: {val_loss}\n')
                    result_file.write(f'Part {i+1} Test Accuracy: {val_accuracy}\n')

                    # Step the scheduler based on validation loss
                    scheduler.step(val_loss)



import numpy as np
from sklearn.model_selection import train_test_split

from ..._Trainer_Yes_No_Questions_Lib import *
from ...Split_Dataset import *


def main():
    result_folder = f"results/base_models_result_continual_learning_enhanced"
    os.makedirs(result_folder, exist_ok=True)

    for dataset in datasets:

        first_half_train_data,second_half_train_data  = split_dataset_file(dataset,file_name="train.js",output_file_type="train")

        val_data = load_dataset_yes_no(dataPath + f"{dataset}/test.json")
        test_data = load_dataset_yes_no(dataPath + f"{dataset}/test.json")
        for model_name in models:
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
                train_encodings = tokenizer([(item["question"], item["context"]) for item in first_half_train_data],
                                            truncation=True, padding=True, max_length=512)
                val_encodings = tokenizer([(item["question"], item["context"]) for item in val_data],
                                          truncation=True, padding=True, max_length=512)
                test_encodings = tokenizer([(item["question"], item["context"]) for item in test_data],
                                           truncation=True, padding=True, max_length=512)

                train_labels = [label_map[item["answer"]] for item in first_half_train_data]
                val_labels = [label_map[item["answer"]] for item in val_data]
                test_labels = [label_map[item["answer"]] for item in test_data]

                # Dataset
                train_dataset = YesNoDataset(train_encodings, train_labels)
                val_dataset = YesNoDataset(val_encodings, val_labels)
                test_dataset = YesNoDataset(test_encodings, test_labels)

                # DataLoader
                train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
                test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

                if dataset == "PubmedQA":
                    # Model
                    criterion = nn.CrossEntropyLoss()
                    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)
                else:
                    # Model
                    criterion = nn.BCEWithLogitsLoss()
                    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=1)

                optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
                model.to(device)
                scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=100, verbose=True)

                # Training
                best_model_path = train_model(model, criterion, optimizer, scheduler, train_loader, number_epoch,
                                              device, result_file, test_loader)

                # Load best model and test
                model.load_state_dict(torch.load(best_model_path))

                # Testing
                test_acc = test_model(model, test_loader, device, result_file)
                # Save test accuracy to a separate file for accumulating results
                with open(os.path.join(result_folder, "test_accuracies.txt"), "a") as acc_file:
                    acc_file.write(f"{model_name},{dataset},{test_acc}\n")

            #------------------------------------------------------------------------------------------------------
            # Continual fine-tuning on quarter batches
            #------------------------------------------------------------------------------------------------------
            number_parts = 4
            result_continual_file_path = os.path.join(dataset_folder, "results_continual_parts.txt")
            with open(result_continual_file_path, "w") as result_continual_file:
                num_iterations = 4  # Define how many times you want to repeat the splitting and training process
                test_accuracies = []

                for iteration in range(num_iterations):
                    # Randomly shuffle second half data before splitting to ensure randomness in splits
                    np.random.shuffle(second_half_train_data)
                    quarter_batches = np.array_split(second_half_train_data, 4)

                    iteration_accuracies = []

                    for i, quarter_data in enumerate(quarter_batches):
                        quarter_encodings = tokenizer([(item["question"], item["context"]) for item in quarter_data],
                                                      truncation=True, padding=True, max_length=512)
                        quarter_labels = [label_map[item["answer"]] for item in quarter_data]
                        quarter_dataset = YesNoDataset(quarter_encodings, quarter_labels)
                        quarter_loader = DataLoader(quarter_dataset, batch_size=8, shuffle=True)

                        # Training
                        best_model_path = train_model(model, criterion, optimizer, scheduler, quarter_loader,
                                                      number_epoch,
                                                      device, result_file, test_loader)

                        # Load best model and test
                        model.load_state_dict(torch.load(best_model_path))

                        # Testing the model
                        test_acc = test_model(model, test_loader, device, result_file)
                        iteration_accuracies.append(test_acc)

                    # Store the average accuracy of this iteration
                    test_accuracies.append(iteration_accuracies)
                    print(f"test accuracy for iteration {iteration + 1}: {iteration_accuracies}")

                overall_mean_acc = []
                for iteration_accuracies in test_accuracies:
                    for i in range(number_parts):
                        part_acc = iteration_accuracies[i] + part_acc
                    overall_mean_acc.append(part_acc/num_iterations)
                print(f"Overall mean test accuracy after {num_iterations} iterations: {overall_mean_acc}")


if __name__ == '__main__':
    main()

if __name__ == '__main__':
    main()