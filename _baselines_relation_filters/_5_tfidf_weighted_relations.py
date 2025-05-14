import glob
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
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


def tfidf_weighted_relations(all_relations, query, top_n=10):
    """
    Calculate TF-IDF scores for relations and return the top-weighted ones.
    """
    if not all_relations:
        return []  # Return empty if no relations

    # Extract relation texts
    relation_texts = [relation["text"] for relation in all_relations if "text" in relation]

    # If no valid texts, return an empty list
    if not relation_texts:
        return []

    # Include the query as part of the corpus
    corpus = [query] + relation_texts

    # Calculate TF-IDF scores
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # Get scores for relations relative to the query
    scores = tfidf_matrix[0, 1:].toarray().flatten()

    # Sort relations by scores and select top N
    sorted_indices = scores.argsort()[::-1]
    top_relations = [all_relations[i] for i in sorted_indices[:top_n]]

    return top_relations


def process_json_tfidf(json_file_path, kg_name, top_n=10):
    """Process JSON file with TF-IDF weighting."""
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    new_data = []
    for item in tqdm(data):
        query = item.get("query", "")
        question_relations = item.get(f"{kg_name}_concepts_question_with_kg_data", {})
        context_relations = item.get(f"{kg_name}_concepts_context_with_kg_data", {})

        # Aggregate all relations
        all_relations = []
        for concept_relations in question_relations.values():
            all_relations.extend(concept_relations.get("triples", []))
        for concept_relations in context_relations.values():
            all_relations.extend(concept_relations.get("triples", []))

        # Apply TF-IDF weighting
        weighted_relations = tfidf_weighted_relations(all_relations, query, top_n)
        item[f"{kg_name}_tfidf_weighted_relations"] = weighted_relations
        new_data.append(item)

    return new_data


def process_dataset_tfidf(dataset, kg_name, top_n=10):
    """Process dataset with TF-IDF weighting."""
    input_directory = f'{root_folder}/Pre_Processed_Datasets/{dataset}/3_extracted_concepts_relations/'
    output_directory = f'{root_folder}/Pre_Processed_Datasets/{dataset}/4_extracted_concepts_relations_tfidf/'
    os.makedirs(output_directory, exist_ok=True)

    for json_file_path in glob.glob(input_directory + '*.json'):
        new_data = process_json_tfidf(json_file_path, kg_name, top_n)

        # Save processed data
        filename = os.path.basename(json_file_path)
        new_filename = os.path.splitext(filename)[0] + f"_{kg_name}_{dataset}_with_tfidf_relations.json"
        output_filepath = os.path.join(output_directory, new_filename)
        save_json_file(new_data, output_filepath)


def merge_records(data1, data2, record_type="test", kg1="primekg", kg2="hetionet"):
    """Merge records from two datasets."""
    merged_data = []
    for ind1, question in enumerate(data1):
        item = data1[ind1]
        item2 = data2[ind1]
        # Combine relations
        item[f"{kg2}_tfidf_weighted_relations"] = item2.get(f"{kg2}_tfidf_weighted_relations", [])
        merged_data.append(item)
    return merged_data


def merge_datasets_tfidf(dataset, kgs, split_type):
    """Merge datasets for a specific split (train/dev/test)."""
    directory_path = f"{root_folder}/Pre_Processed_Datasets/{dataset}/4_extracted_concepts_relations_tfidf/"
    data1 = load_json_file(f'{directory_path}{split_type}_{kgs[0]}_{dataset}_with_tfidf_relations.json')
    data2 = load_json_file(f'{directory_path}{split_type}_{kgs[1]}_{dataset}_with_tfidf_relations.json')

    # Merge data
    merged_data = merge_records(data1, data2, record_type=split_type, kg1=kgs[0], kg2=kgs[1])
    output_file_path = os.path.join(directory_path, f"{split_type}.json")
    save_json_file(merged_data, output_file_path)


def filter_relations_tfidf():
    """Main function to process datasets using TF-IDF weighting and merge data."""
    kgs = ["primekg", "hetionet"]
    for dataset in datasets:
        # Process datasets for both KGs
        for kg_name in kgs:
            process_dataset_tfidf(dataset, kg_name, top_n=10)

        # Merge datasets for train/dev/test splits
        for split_type in ["train", "dev", "test"]:
            merge_datasets_tfidf(dataset, kgs, split_type)


if __name__ == "__main__":
    filter_relations_tfidf()
