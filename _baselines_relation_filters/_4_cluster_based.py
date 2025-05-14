import glob
import json
import os
from sklearn.cluster import KMeans
import numpy as np
from tqdm import tqdm

root_folder = ".."
datasets = ["BioASQ2024","BioASQ_BLURB","PubmedQA"]   # Add other datasets like "BioASQ", "PubmedQA" if needed


def load_json_file(filepath):
    """Load data from a JSON file."""
    with open(filepath, 'r') as file:
        return json.load(file)


def save_json_file(data, filepath):
    """Save data to a JSON file."""
    with open(filepath, 'w') as file:
        json.dump(data, file, indent=4)


def cluster_based_relation_selection(all_relations, num_clusters=5):
    """
    Cluster relations based on their embeddings and select a representative from each cluster.
    """
    if len(all_relations) <= num_clusters:
        return all_relations  # Return all relations if less than the number of clusters

    # Extract embeddings for clustering
    embeddings = np.array([relation["embedding"] for relation in all_relations if "embedding" in relation])

    if embeddings.shape[0] < num_clusters:
        return all_relations  # Not enough embeddings for clustering

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)

    # Select one representative relation from each cluster
    cluster_centers = kmeans.cluster_centers_
    selected_relations = []
    for cluster_id in range(num_clusters):
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        # Find the closest relation to the cluster center
        closest_index = cluster_indices[np.argmin(np.linalg.norm(embeddings[cluster_indices] - cluster_centers[cluster_id], axis=1))]
        selected_relations.append(all_relations[closest_index])

    return selected_relations


def process_json_cluster_based(json_file_path, kg_name, num_clusters=5):
    """Process JSON file with cluster-based relation selection."""
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

        # Perform cluster-based selection
        selected_relations = cluster_based_relation_selection(all_relations, num_clusters)
        item[f"{kg_name}_clustered_relations"] = selected_relations
        new_data.append(item)

    return new_data


def process_dataset_cluster_based(dataset, kg_name, num_clusters=5):
    """Process dataset with cluster-based relation selection."""
    input_directory = f'{root_folder}/Pre_Processed_Datasets/{dataset}/3_extracted_concepts_relations/'
    output_directory = f'{root_folder}/Pre_Processed_Datasets/{dataset}/4_extracted_concepts_relations_clustered/'
    os.makedirs(output_directory, exist_ok=True)

    for json_file_path in glob.glob(input_directory + '*.json'):
        new_data = process_json_cluster_based(json_file_path, kg_name, num_clusters)

        # Save processed data
        filename = os.path.basename(json_file_path)
        new_filename = os.path.splitext(filename)[0] + f"_{kg_name}_{dataset}_with_clustered_relations.json"
        output_filepath = os.path.join(output_directory, new_filename)
        save_json_file(new_data, output_filepath)


def merge_records(data1, data2, record_type="test", kg1="primekg", kg2="hetionet"):
    """Merge records from two datasets."""
    merged_data = []
    for ind1, question in enumerate(data1):
        item = data1[ind1]
        item2 = data2[ind1]
        # Combine relations
        item[f"{kg2}_clustered_relations"] = item2.get(f"{kg2}_clustered_relations", [])
        merged_data.append(item)
    return merged_data


def merge_datasets_cluster_based(dataset, kgs, split_type):
    """Merge datasets for a specific split (train/dev/test)."""
    directory_path = f"{root_folder}/Pre_Processed_Datasets/{dataset}/4_extracted_concepts_relations_clustered/"
    data1 = load_json_file(f'{directory_path}{split_type}_{kgs[0]}_{dataset}_with_clustered_relations.json')
    data2 = load_json_file(f'{directory_path}{split_type}_{kgs[1]}_{dataset}_with_clustered_relations.json')

    # Merge data
    merged_data = merge_records(data1, data2, record_type=split_type, kg1=kgs[0], kg2=kgs[1])
    output_file_path = os.path.join(directory_path, f"{split_type}.json")
    save_json_file(merged_data, output_file_path)


def filter_relations_cluster_based():
    """Main function to process datasets using cluster-based relation selection and merge data."""
    kgs = ["primekg", "hetionet"]
    for dataset in datasets:
        # Process datasets for both KGs
        for kg_name in kgs:
            process_dataset_cluster_based(dataset, kg_name, num_clusters=5)

        # Merge datasets for train/dev/test splits
        for split_type in ["train", "dev", "test"]:
            merge_datasets_cluster_based(dataset, kgs, split_type)


if __name__ == "__main__":
    filter_relations_cluster_based()
