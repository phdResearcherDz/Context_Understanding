from transformers import AutoTokenizer, AutoModel
import glob
import json
import os
import pandas as pd
from neo4j import GraphDatabase
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel
import torch
import glob
import json
import os
from tqdm import tqdm

root_folder = ".."
# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize tokenizer and model and move the model to the specified device
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/msmarco-distilbert-base-tas-b")
model = AutoModel.from_pretrained("sentence-transformers/msmarco-distilbert-base-tas-b")
model.to(device)


def cls_pooling(model_output):
    return model_output.last_hidden_state[:, 0]

# Function to encode text and compute embeddings
def encode(texts):
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    with torch.no_grad():
        model_output = model(**encoded_input, return_dict=True)

    # Free up memory
    del encoded_input
    torch.cuda.empty_cache()
    return cls_pooling(model_output)

def get_relevant_triples(question, triples, top_k=5, batch_size=32):
    if not triples:
        return []

    triple_texts = []
    triples_map = []
    for triple in triples:
        triples_map.append({"source":triple['source'],"relation":triple['relation'],"target_nodes":triple["target_nodes"]})
        triple_text = f"{triple['source']} {triple['relation']} {triple['target_nodes']}"
        triple_texts.append(triple_text)

    question_emb = encode([question])
    question_emb = question_emb.to(device)

    # Process in smaller batches
    scores = []
    for i in range(0, len(triple_texts), batch_size):
        batch_texts = triple_texts[i:i + batch_size]
        triple_embs = encode(batch_texts)
        triple_embs = triple_embs.to(device)

        batch_scores = torch.mm(question_emb, triple_embs.transpose(0, 1))[0].cpu().tolist()
        scores.extend(batch_scores)

        # Clear cache
        del triple_embs
        torch.cuda.empty_cache()

    if not scores:
        return []

    triple_score_pairs = sorted(zip(triples_map, scores), key=lambda x: x[1], reverse=True)


    relevants =  [triple for triple, score in triple_score_pairs[:top_k]]

    return  relevants

def process_json(json_file_path, kg_name):
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    new_data = list()
    for item in tqdm(data):
        question_medical_concepts_relations = item.get(f"{kg_name}_concepts_question_with_kg_data", '')
        context_medical_concepts_relations = item.get(f"{kg_name}_concepts_context_with_kg_data", '')

        relevant_relations_filtered_sm = []

        #select relvant relations from context
        for concept in context_medical_concepts_relations:
            triples = context_medical_concepts_relations[concept]["triples"]
            for triple in triples:
                relevant_relations_filtered_sm.append(triple)

        #select relevant relations from question
        for concept in question_medical_concepts_relations:
            triples = question_medical_concepts_relations[concept]["triples"]
            for triple in triples:
                relevant_relations_filtered_sm.append(triple)
        question = item["question"]
        item[f"{kg_name}_relevant_relations_sm"] = get_relevant_triples(question,relevant_relations_filtered_sm,10)

        item[f"{kg_name}_concepts_question_with_kg_data"] = None
        item[f"{kg_name}_concepts_context_with_kg_data"] = None
        new_data.append(item)
    return new_data

def process_dataset(dataset, kg_name):
    directory_path = f'{root_folder}/Pre_Processed_Datasets/{dataset}/3_extracted_concepts_relations/'
    for json_file_path in glob.glob(directory_path + '*.json'):
        new_data = process_json(json_file_path, kg_name)

        # Create new file path
        directory, filename = os.path.split(json_file_path)
        new_directory = f"{root_folder}/Pre_Processed_Datasets/{dataset}/4_extracted_concepts_relations_filtered_similarity"
        os.makedirs(new_directory,exist_ok=True)

        new_filename = os.path.splitext(filename)[0] + f"_{kg_name}_{dataset}_with_relations_filtered.json"
        new_file_path = os.path.join(new_directory, new_filename)

        # Save new data to the new file
        with open(new_file_path, 'w') as new_file:
            json.dump(list(new_data), new_file, indent=4)


def load_json_file(filepath):
    """Load data from a JSON file."""
    with open(filepath, 'r') as file:
        return json.load(file)





def merge_records(data1, data2,type="test",kg1 ="primekg", kg2 = "hetionet"):

    # Merge the dictionaries
    merged_data = []
    for ind1,question in enumerate(data1):
        item = data1[ind1]
        item2 = data2[ind1]
        item[f"{kg2}_relevant_relations_sm"] = item2[f"{kg2}_relevant_relations_sm"]
        merged_data.append(item)
    return merged_data

