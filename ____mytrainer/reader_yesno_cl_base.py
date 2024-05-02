import shutil

import numpy as np

from Split_Dataset import *


def main_cl_base():
    result_folder = f"results/results_cl_base"

    if os.path.exists(result_folder):
        # Remove the folder and all its contents
        shutil.rmtree(result_folder)

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
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
                test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

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
                # Save test accuracy to a separate file for accumulating results
                with open(os.path.join(result_folder, "test_accuracies.txt"), "a") as acc_file:
                    test_acc = test_model(model, test_loader, device, acc_file)
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
                                                      device, result_continual_file, test_loader)

                        # Load best model and test
                        model.load_state_dict(torch.load(best_model_path))

                        # Testing the model
                        test_acc = test_model(model, test_loader, device, result_continual_file)
                        iteration_accuracies.append(test_acc)

                    # Store the average accuracy of this iteration
                    test_accuracies.append(iteration_accuracies)
                    print(f"test accuracy for iteration {iteration + 1}: {iteration_accuracies}")

                overall_mean_acc = []
                for iteration_accuracies in test_accuracies:
                    part_acc = 0
                    for i in range(number_parts):
                        part_acc = iteration_accuracies[i] + part_acc
                    overall_mean_acc.append(part_acc/num_iterations)
                print(f"Overall mean test accuracy after {num_iterations} iterations: {overall_mean_acc}")
                result_continual_file.write(f"Overall mean test accuracy after {num_iterations} iterations: {overall_mean_acc}")

if __name__ == '__main__':
    main_cl_enhanced_context_model()
