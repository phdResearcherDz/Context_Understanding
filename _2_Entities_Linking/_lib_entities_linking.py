from transformers import AutoTokenizer, AutoModel
import glob
import json
import os
import pandas as pd
from neo4j import GraphDatabase
from tqdm import tqdm
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from Levenshtein import distance as levenshtein_distance

# Function to connect to Neo4j Database
tokenizer = AutoTokenizer.from_pretrained("GanjinZero/UMLSBert_ENG")
model = AutoModel.from_pretrained("GanjinZero/UMLSBert_ENG")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Function to connect to Neo4j Database
def connect_to_neo4j(uri, username, password):
    return GraphDatabase.driver(uri, auth=(username, password))

# Function to fetch nodes from Neo4j
def fetch_nodes(driver):
    with driver.session() as session:
        result = session.execute_read(lambda tx: tx.run("MATCH (n) RETURN n").data())
        nodes = [record['n'] for record in result]
    return pd.DataFrame(nodes)


def extract_entities(entities):
    """ Extracts canonical names and their respective definitions from entities. """
    extracted_info = {}
    for entity in entities:
        for key, value in entity.items():
            if 'canonical_name' in value:
                extracted_info[key] = value['canonical_name']
    return extracted_info

def precompute_node_embeddings(node_descriptions, batch_size=1000, model=model, tokenizer=tokenizer, device=device):
    model.eval()
    all_embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(node_descriptions), batch_size), desc="Computing embeddings"):
            node_desc_batch = node_descriptions[i:i+batch_size]
            node_desc_tokens = tokenizer(node_desc_batch, padding=True, return_tensors='pt').to(device)
            node_desc_embeddings = model(**node_desc_tokens).pooler_output
            all_embeddings.append(node_desc_embeddings)

            # Delete tensors that are no longer needed to free up memory
            del node_desc_tokens, node_desc_embeddings
            torch.cuda.empty_cache()  # Clear memory cache

        # Concatenate all embeddings into a single tensor
        all_embeddings = torch.cat(all_embeddings, dim=0)

    return all_embeddings

def compute_embeddings_similarity(node_desc_embeddings, query_words, batch_size=10, model=model, tokenizer=tokenizer, device=device):
    model.eval()  # Ensure the model is in evaluation mode
    total_similarities = []
    with torch.no_grad():
        for j in range(0, len(query_words), batch_size):
            query_word_batch = query_words[j:j+batch_size]
            query_words_tokens = tokenizer(query_word_batch, padding=True, return_tensors='pt').to(device)
            query_words_embeddings = model(**query_words_tokens).pooler_output

            # Ensure dimensions for cosine similarity are compatible
            if query_words_embeddings.shape[0] < batch_size:
                # Pad the embeddings tensor to have the same first dimension as the batch size
                padding_size = batch_size - query_words_embeddings.shape[0]
                query_words_embeddings = torch.nn.functional.pad(query_words_embeddings, (0, 0, 0, padding_size), "constant", 0)

            similarities = torch.nn.functional.cosine_similarity(query_words_embeddings.unsqueeze(1), node_desc_embeddings.unsqueeze(0), dim=-1)
            total_similarities.append(similarities)
            
            torch.cuda.empty_cache()  # Clear memory cache

        # Concatenate all batch results into a single tensor, handling the case where the last batch is smaller
        if len(total_similarities) > 1:
            max_len = max(t.shape[1] for t in total_similarities)  # Find the maximum length in dimension 1
            padded_similarities = [torch.nn.functional.pad(t, (0, max_len - t.shape[1])) for t in total_similarities]
            total_similarities = torch.cat(padded_similarities, dim=1)
        else:
            total_similarities = torch.cat(total_similarities, dim=1)

    return total_similarities


def process_json_livenshtien(file_path, df_nodes, kg_name, node_name_attribute):
    with open(file_path, 'r') as file:
        data = json.load(file)
    node_names = df_nodes[node_name_attribute].tolist()
    
    new_data = []
    for item in tqdm(data):
        context_entities = extract_entities(item['context_entities'])
        question_entities = extract_entities(item['question_entities'])
        
        item[f"{kg_name}_concepts_context"] = {}
        item[f"{kg_name}_concepts_question"] = {}

        # Function to find closest node name by Levenshtein Distance
        def find_closest_name(entity_name):
            min_distance = float('inf')
            closest_name = None
            for node_name in node_names:
                dist = levenshtein_distance(entity_name.lower(), node_name.lower())
                if dist < min_distance:
                    min_distance = dist
                    closest_name = node_name
            print(min_distance)
            print(f'{entity_name}:{closest_name}')
            return closest_name

        # Context entities matching
        for entity_name in context_entities:
            closest_node = find_closest_name(entity_name)
            item[f"{kg_name}_concepts_context"][entity_name] = closest_node

        # Question entities matching
        for entity_name in question_entities:
            closest_node = find_closest_name(entity_name)
            item[f"{kg_name}_concepts_question"][entity_name] = closest_node

        new_data.append(item)
        print(item[f"{kg_name}_concepts_context"])
        print(item[f"{kg_name}_concepts_question"])

    return new_data


def process_json_ngram(file_path, df_nodes, kg_name, node_name_attribute):
    with open(file_path, 'r') as file:
        data = json.load(file)

    threshold = 0.80
    vectorizer = CountVectorizer(
        analyzer='char_wb',       # or 'char_wb' for more granular analysis within words
        ngram_range=(1, 3),    # Unigrams to trigrams
        stop_words='english',  # Standard or customized stop words
    )
    vectorizer.fit(df_nodes[node_name_attribute])  # Fit only on node names
    node_vectors = vectorizer.transform(df_nodes[node_name_attribute])

    new_data = []
    for item in tqdm(data):
        context_entities = extract_entities(item['context_entities'])
        question_entities = extract_entities(item['question_entities'])

        item[f"{kg_name}_concepts_context"] = {}
        item[f"{kg_name}_concepts_question"] = {}

        # Process context entities
        if context_entities:
            entity_vectors = vectorizer.transform(list(context_entities.values()))
            similarities = cosine_similarity(entity_vectors, node_vectors)

            for idx, entity_name in enumerate(context_entities):
                max_index = similarities[idx].argmax()
                max_similarity_score = similarities[idx, max_index]
                if max_similarity_score > threshold:
                    most_similar_node = df_nodes[node_name_attribute].iloc[max_index]
                    item[f"{kg_name}_concepts_context"][entity_name] = most_similar_node
                    print(f"{entity_name}:{most_similar_node}")

        # Process question entities
        if question_entities:
            entity_vectors = vectorizer.transform(list(question_entities.values()))
            similarities = cosine_similarity(entity_vectors, node_vectors)

            for idx, entity_name in enumerate(question_entities):
                max_index = similarities[idx].argmax()
                max_similarity_score = similarities[idx, max_index]
                if max_similarity_score > threshold:
                    most_similar_node = df_nodes[node_name_attribute].iloc[max_index]
                    item[f"{kg_name}_concepts_question"][entity_name] = most_similar_node
                    print(f"{entity_name}:{most_similar_node}")

        new_data.append(item)

    return new_data

def process_json(file_path, df_nodes, kg_name, node_name_attribute):
    with open(file_path, 'r') as file:
        data = json.load(file)
    node_names = df_nodes[node_name_attribute].tolist()
    node_embeddings = precompute_node_embeddings(node_names)
    similarity_threshold = 0.90
    new_data = []
    for item in tqdm(data):
        context_entities = extract_entities(item['context_entities'])
        question_entities = extract_entities(item['question_entities'])
        
        item[f"{kg_name}_concepts_context"] = {}
        item[f"{kg_name}_concepts_question"] = {}

        if context_entities:
            similarities = compute_embeddings_similarity(node_embeddings, list(context_entities.values()))
            top_indices = similarities.argmax(dim=1).tolist()
            for idx, entity_name in enumerate(context_entities):
                if idx < len(top_indices) and top_indices[idx] < len(df_nodes) and top_indices[idx] > similarity_threshold:
                    similar_concept = df_nodes[node_name_attribute].iloc[top_indices[idx]]
                    item[f"{kg_name}_concepts_context"][entity_name] = similar_concept

        if question_entities:
            similarities_question = compute_embeddings_similarity(node_embeddings, list(question_entities.values()))
            top_indices_question = similarities_question.argmax(dim=1).tolist()
            for idx, entity_name in enumerate(question_entities):
                if idx < len(top_indices_question) and top_indices_question[idx] < len(df_nodes) and top_indices_question[idx] > similarity_threshold:
                    similar_concept_question = df_nodes[node_name_attribute].iloc[top_indices_question[idx]]
                    item[f"{kg_name}_concepts_question"][entity_name] = similar_concept_question

        new_data.append(item)

        print(item[f"{kg_name}_concepts_context"])
        print(item[f"{kg_name}_concepts_question"])
    return new_data

def process_dataset(dataset, df_nodes, kg_name, node_name_attribute):
    directory_path = f'Pre_Processed_Datasets/{dataset}/1_extracted_entities_with_metadata/'
    for json_file_path in glob.glob(directory_path + '*.json'):
        # new_data = process_json(json_file_path, df_nodes,kg_name,node_name_attribute)
        new_data = process_json_ngram(json_file_path, df_nodes,kg_name,node_name_attribute)

        # Create new file path
        directory, filename = os.path.split(json_file_path)
        new_directory = f"Pre_Processed_Datasets/{dataset}/2_kg_linked_entities"
        os.makedirs(new_directory,exist_ok=True)

        new_filename = os.path.splitext(filename)[0] + f"_{kg_name}_{dataset}_with_concept.json"
        new_file_path = os.path.join(new_directory, new_filename)

        # Save new data to the new file
        with open(new_file_path, 'w') as new_file:
            json.dump(list(new_data), new_file, indent=4)



