import os
import json


def process_dataset(root_path, dataset_name,folder_prefix):
    dataset_folders = [folder for folder in os.listdir(root_path)  if folder.startswith(folder_prefix)]
    jsonl_file_path = os.path.join(root_path,"dev.json")
    list_folder_names = []
    for folder in dataset_folders:
        dev_questions = set()
        with open(jsonl_file_path, 'r') as file:
            for line in file:
                data = json.loads(line)
                dev_questions.add(data['sentence1'])

        # Define the path for the current folder
        current_folder_path = os.path.join(root_path,folder)

        # Define the output folder path based on the dataset and method
        method_name = folder.split("_WITH_")[-1] if "_WITH_" in folder else folder

        output_folder = f"{dataset_name}_{method_name}"
        output_folder_path = os.path.join(".", output_folder)
        list_folder_names.append(output_folder)
        # Create the output folder if it doesn't exist
        if not os.path.exists(output_folder_path):
            os.makedirs(output_folder_path)

        # Process each type of file (train and test)
        for type_ in ["train", "test"]:
            file_path = os.path.join(current_folder_path, f"{type_}.json")

            # Load data from JSON file
            with open(file_path, "r") as f:
                data = json.load(f)

            # Convert data
            converted_data = []
            converted_data_jb = []
            for idx, item in enumerate(data):
                id_str = f"converted_{idx}"
                sentence1 = item["question"]
                sentence2 = f'{item["context"]}'
                label = item["answer"]
                if label not in ["yes","no","maybe"]:
                    continue
                json_obj = {
                    "id": id_str,
                    "sentence1": sentence1,
                    "sentence2": sentence2,
                    "label": label
                }
                converted_data.append(json_obj)
                converted_data_jb.append(json.dumps(json_obj))

            if type_ == "train":
                output_file_path_dev = os.path.join(output_folder_path, f"dev.json")
                output_file_path_train = os.path.join(output_folder_path, f"train.json")

                updated_train_data = []
                dev_data = []
                for item in converted_data:
                    if item["sentence1"] in dev_questions:
                        dev_data.append(json.dumps(item))
                    else:
                        updated_train_data.append(json.dumps(item))

                with open(output_file_path_train, "w") as f:
                    f.write("\n".join(updated_train_data))

                with open(output_file_path_dev, "w") as f:
                    f.write("\n".join(dev_data))
            else:
                output_file_path = os.path.join(output_folder_path, f"{type_}.json")
                with open(output_file_path, "w") as f:
                    f.write("\n".join(converted_data_jb))

    print(list_folder_names)
# Set the root directory and dataset names
root = "../Pre_Processed_Datasets/"

# Call the function for each dataset
process_dataset(os.path.join(root, "base"), "PubmedQA","P")
