import glob
import json
import os
from sentence_transformers import SentenceTransformer, util
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


def dense_retrieval_relation_ranking(all_relations, query, model, top_n=10):
    """
    Perform dense retrieval to rank relations by relevance to the query.
    """
    if not all_relations or not query:
        return []  # Return empty if no relations or query is empty

    # Extract relation texts
    relation_texts = [relation["text"] for relation in all_relations if "text" in relation]

    if not relation_texts:
        return []  # Return empty if no valid texts

    # Compute embeddings
    query_embedding = model.encode(query, convert_to_tensor=True)
    relation_embeddings = model.encode(relation_texts, convert_to_tensor=True)

    # Compute cosine similarity
    scores = util.pytorch_cos_sim(query_embedding, relation_embeddings).squeeze()

    # Get top-N relations
    top_indices = scores.argsort(descending=True)[:top_n]
    top_relations = [all_relations[i] for i in top_indices]

    return top_relations


def process_json_dense_retrieval(json_file_path, kg_name, model, top_n=10):
    """Process JSON file with dense retrieval for relation ranking."""
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

        # Perform dense retrieval
        ranked_relations = dense_retrieval_relation_ranking(all_relations, query, model, top_n)
        item[f"{kg_name}_dense_retrieval_relations"] = ranked_relations
        new_data.append(item)

    return new_data


def process_dataset_dense_retrieval(dataset, kg_name, model, top_n=10):
    """Process dataset with dense retrieval for relation ranking."""
    input_directory = f'{root_folder}/Pre_Processed_Datasets/{dataset}/3_extracted_concepts_relations/'
    output_directory = f'{root_folder}/Pre_Processed_Datasets/{dataset}/4_extracted_concepts_relations_dense_retrieval/'
    os.makedirs(output_directory, exist_ok=True)

    for json_file_path in glob.glob(input_directory + '*.json'):
        new_data = process_json_dense_retrieval(json_file_path, kg_name, model, top_n)

        # Save processed data
        filename = os.path.basename(json_file_path)
        new_filename = os.path.splitext(filename)[0] + f"_{kg_name}_{dataset}_with_dense_retrieval_relations.json"
        output_filepath = os.path.join(output_directory, new_filename)
        save_json_file(new_data, output_filepath)


def merge_records(data1, data2, record_type="test", kg1="primekg", kg2="hetionet"):
    """Merge records from two datasets."""
    merged_data = []
    for ind1, question in enumerate(data1):
        item = data1[ind1]
        item2 = data2[ind1]
        # Combine relations
        item[f"{kg2}_dense_retrieval_relations"] = item2.get(f"{kg2}_dense_retrieval_relations", [])
        merged_data.append(item)
    return merged_data


def merge_datasets_dense_retrieval(dataset, kgs, split_type):
    """Merge datasets for a specific split (train/dev/test)."""
    directory_path = f"{root_folder}/Pre_Processed_Datasets/{dataset}/4_extracted_concepts_relations_dense_retrieval/"
    data1 = load_json_file(f'{directory_path}{split_type}_{kgs[0]}_{dataset}_with_dense_retrieval_relations.json')
    data2 = load_json_file(f'{directory_path}{split_type}_{kgs[1]}_{dataset}_with_dense_retrieval_relations.json')

    # Merge data
    merged_data = merge_records(data1, data2, record_type=split_type, kg1=kgs[0], kg2=kgs[1])
    output_file_path = os.path.join(directory_path, f"{split_type}.json")
    save_json_file(merged_data, output_file_path)


def filter_relations_dense_retrieval():
    """Main function to process datasets using dense retrieval for relation ranking."""
    kgs = ["primekg", "hetionet"]
    model_name = "sentence-transformers/all-mpnet-base-v2"  # Use any SentenceTransformer model
    model = SentenceTransformer(model_name)

    for dataset in datasets:
        # Process datasets for both KGs
        for kg_name in kgs:
            process_dataset_dense_retrieval(dataset, kg_name, model, top_n=10)

        # Merge datasets for train/dev/test splits
        for split_type in ["train", "dev", "test"]:
            merge_datasets_dense_retrieval(dataset, kgs, split_type)


if __name__ == "__main__":
    filter_relations_dense_retrieval()
