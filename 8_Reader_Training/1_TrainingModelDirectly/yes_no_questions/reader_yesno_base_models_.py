import shutil

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from _Libreries._Trainer_Yes_No_Questions_Lib import *


def main_base_model():
    result_folder = f"results/base_models_result"
    os.makedirs(result_folder, exist_ok=True)

    for dataset in datasets:
        print(dataPath + f"{dataset}/train.json")
        train_data = load_dataset_yes_no(dataPath + f"{dataset}/train.json")
        test_data = load_dataset_yes_no(dataPath + f"{dataset}/test.json")
        for model_name in models:
            model_folder = os.path.join(result_folder, model_name.replace("/", "_"))
            os.makedirs(model_folder, exist_ok=True)
            dataset_folder = os.path.join(model_folder, dataset)
            os.makedirs(dataset_folder, exist_ok=True)
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

                train_dataset = YesNoDataset(train_encodings, train_labels)
                test_dataset = YesNoDataset(test_encodings, test_labels)

                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

                criterion = nn.BCEWithLogitsLoss()
                model = BertForSequenceClassification.from_pretrained(model_name, num_labels=1)

                optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
                model.to(device)
                scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=100, verbose=True)

                # Training
                best_model_path = train_model(model, criterion, optimizer, scheduler, train_loader, number_epoch, device, result_file,test_loader)

                # Load best model and test
                model.load_state_dict(torch.load(best_model_path))

                # Testing
                test_acc = test_model(model, test_loader, device, result_file)

                # Save test accuracy to a separate file for accumulating results
                with open(os.path.join(result_folder, "test_accuracies.txt"), "a") as acc_file:
                    acc_file.write(f"{model_name},{dataset},{test_acc}\n")

if __name__ == '__main__':
    main_base_model()