import glob
import json
import os
import random
from tqdm import tqdm

root_folder = ".."
datasets = ["BioASQ2024"]  # Add other datasets like "BioASQ", "PubmedQA" if needed


def load_json_file(filepath):
    """Load data from a JSON file."""
    with open(filepath, 'r') as file:
        return json.load(file)


def save_json_file(data, filepath):
    """Save data to a JSON file."""
    with open(filepath, 'w') as file:
        json.dump(data, file, indent=4)


def random_relation_addition(all_relations, num_relations=10):
    """Randomly select relations from the provided list."""
    if len(all_relations) <= num_relations:
        return all_relations  # Return all relations if less than num_relations
    return random.sample(all_relations, num_relations)


def process_json_random(json_file_path, kg_name, num_relations=10):
    """Process JSON file with random relation addition."""
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    new_data = []
    for item in tqdm(data):
        question_relations = item.get(f"{kg_name}_concepts_question_with_kg_data", {})
        context_relations = item.get(f"{kg_name}_concepts_context_with_kg_data", {})

        # Aggregate all relations
        all_relations = []
        for concept_relations in question_relations.values():
            all_relations.extend(concept_relations.get("triples", []))
        for concept_relations in context_relations.values():
            all_relations.extend(concept_relations.get("triples", []))

        # Randomly select relations
        random_relations = random_relation_addition(all_relations, num_relations)
        item[f"{kg_name}_random_relations"] = random_relations
        new_data.append(item)

    return new_data


def process_dataset_random(dataset, kg_name, num_relations=10):
    """Process dataset with random relation addition."""
    input_directory = f'{root_folder}/Pre_Processed_Datasets/{dataset}/3_extracted_concepts_relations/'
    output_directory = f'{root_folder}/Pre_Processed_Datasets/{dataset}/4_extracted_concepts_relations_random/'
    os.makedirs(output_directory, exist_ok=True)

    for json_file_path in glob.glob(input_directory + '*.json'):
        new_data = process_json_random(json_file_path, kg_name, num_relations)

        # Save processed data
        filename = os.path.basename(json_file_path)
        new_filename = os.path.splitext(filename)[0] + f"_{kg_name}_{dataset}_with_random_relations.json"
        output_filepath = os.path.join(output_directory, new_filename)
        save_json_file(new_data, output_filepath)


def merge_records(data1, data2, record_type="test", kg1="primekg", kg2="hetionet"):
    """Merge records from two datasets."""
    merged_data = []
    for ind1, question in enumerate(data1):
        item = data1[ind1]
        item2 = data2[ind1]
        # Combine relations
        item[f"{kg2}_random_relations"] = item2.get(f"{kg2}_random_relations", [])
        merged_data.append(item)
    return merged_data


def merge_datasets(dataset, kgs, split_type):
    """Merge datasets for a specific split (train/dev/test)."""
    directory_path = f"{root_folder}/Pre_Processed_Datasets/{dataset}/4_extracted_concepts_relations_random/"
    data1 = load_json_file(f'{directory_path}{split_type}_{kgs[0]}_{dataset}_with_random_relations.json')
    data2 = load_json_file(f'{directory_path}{split_type}_{kgs[1]}_{dataset}_with_random_relations.json')

    # Merge data
    merged_data = merge_records(data1, data2, record_type=split_type, kg1=kgs[0], kg2=kgs[1])
    output_file_path = os.path.join(directory_path, f"{split_type}.json")
    save_json_file(merged_data, output_file_path)


def filter_relations_random():
    """Main function to process datasets using random relation addition and merge data."""
    kgs = ["primekg", "hetionet"]
    for dataset in datasets:
        # Process datasets for both KGs
        for kg_name in kgs:
            process_dataset_random(dataset, kg_name, num_relations=10)

        # Merge datasets for train/dev/test splits
        for split_type in ["train", "dev", "test"]:
            merge_datasets(dataset, kgs, split_type)


if __name__ == "__main__":
    filter_relations_random()
