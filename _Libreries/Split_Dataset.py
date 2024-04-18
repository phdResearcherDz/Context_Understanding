from sklearn.model_selection import train_test_split
import pickle
from _Trainer_Yes_No_Questions_Lib import *

def save_data_pickle(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load_data_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def ensure_directory_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
        print(f"Created directory: {folder_path}")
    else:
        print(f"Directory already exists: {folder_path}")

def split_dataset_file(dataset, file_name, output_file_type="train", renew_split=True, random_state=42, test_size=0.5):
    splited_data_path = dataPath + f"{dataset}/splited_data/"
    ensure_directory_exists(splited_data_path)

    first_half_file = splited_data_path + f'first_half_{output_file_type}.pkl'
    second_half_file = splited_data_path + f'second_half_{output_file_type}.pkl'

    # Check for existence of both files
    first_half_exists = os.path.exists(first_half_file)
    second_half_exists = os.path.exists(second_half_file)

    if not renew_split and first_half_exists and second_half_exists:
        try:
            # Try to load existing data splits if renew_split is False
            first_half_data = load_data_pickle(first_half_file)
            second_half_data = load_data_pickle(second_half_file)
            print(f"Loaded existing data splits from {first_half_file} and {second_half_file}.")
            return first_half_data, second_half_data
        except FileNotFoundError:
            print("Split files not found. Splitting dataset anew since renew_split is False but files are missing.")

    # Load full dataset and split if renew_split is True or files were not found
    full_data = load_dataset_yes_no(dataPath + f"{dataset}/{file_name}")
    first_half_data, second_half_data = train_test_split(full_data, test_size=test_size, random_state=random_state)
    save_data_pickle(first_half_data, first_half_file)
    save_data_pickle(second_half_data, second_half_file)
    print(f"Dataset split and saved to {first_half_file} and {second_half_file}.")
    return first_half_data, second_half_data

def load_splited_dataset_file(dataset, output_file_type="train"):
    splited_data_path = dataPath + f"{dataset}/splited_data/"

    first_half_file = splited_data_path + f'first_half_{output_file_type}.pkl'
    second_half_file = splited_data_path + f'second_half_{output_file_type}.pkl'

    # Check for existence of both files
    first_half_exists = os.path.exists(first_half_file)
    second_half_exists = os.path.exists(second_half_file)

    if first_half_exists and second_half_exists:
        first_half_data = load_data_pickle(first_half_file)
        second_half_data = load_data_pickle(second_half_file)
        return first_half_data,second_half_data
    return None,None

if __name__ == '__main__':
    dataset = "BioASQ"
    split_dataset_file(dataset,"train.json",renew_split=False, random_state=60)

    fist,second = load_splited_dataset_file(dataset,"train")
    print(fist[0])