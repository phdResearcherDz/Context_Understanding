import os
import json


def process_dataset(root_path, dataset_name, folder_prefix):
    dataset_folders = [folder for folder in os.listdir(root_path) if folder.startswith(folder_prefix)]
    list_folder_names = []
    for folder in dataset_folders:

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
        for type_ in ["train", "test","dev"]:
            file_path = os.path.join(current_folder_path, f"{type_}.json")

            # Load data from JSON file
            with open(file_path, "r") as f:
                data = json.load(f)

            # Convert data
            converted_data = []
            converted_data_jb = []
            for idx, item in enumerate(data):
                question = item["sent1"].split('. ')[-1]
                item["sent1"] = item["context"]+"[SEP]"+item["metadata"]+"[SEP]"+question
                del item["context"]
                del item["question"]
                del item["context_entities"]
                del item["question_entities"]
                converted_data.append(item)
                converted_data_jb.append(json.dumps(item))

            output_file_path = os.path.join(output_folder_path, f"{type_}.json")
            with open(output_file_path, "w") as f:
                f.write("\n".join(converted_data_jb))

    print(list_folder_names)
# Set the root directory and dataset names
root = "../Pre_Processed_Datasets/"

# Call the function for each dataset
process_dataset(os.path.join(root, "medqa_usmle_hf"), "medqa_usmle_hf", "5_")
