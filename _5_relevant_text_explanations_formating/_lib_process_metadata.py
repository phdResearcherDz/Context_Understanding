from transformers import AutoTokenizer, AutoModel
import glob
import json
import os
import pandas as pd
from neo4j import GraphDatabase
from tqdm import tqdm
import torch
from transformers import  pipeline

root_folder = "."

def get_all_concepts_with_definition(context_ent, question_ent):
    allConcepts = {}

    # Function to process each list of entity dictionaries
    def process_entities(entities):
        for entity in entities:
            for key, value in entity.items():
                # Check if 'definition' exists and is not None
                if value.get("definition"):
                    allConcepts[key] = value["definition"]

    # Process context entities
    process_entities(context_ent)

    # Process question entities
    process_entities(question_ent)

    return allConcepts
def summarize_text(text):
    # Check if CUDA is available and then specify to use GPU (device 0 by default)
    device = 0 if torch.cuda.is_available() else -1  # -1 means CPU

    # Load the summarization pipeline and set it to use the GPU if available
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device)

    # Perform summarization
    summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
    return summary

def get_item_metadata(context_entities, question_entities, primekg_relevant_relations, hetionet_relevant_relations):
    # Get all concepts with definitions
    concepts_def = get_all_concepts_with_definition(context_entities, question_entities)
    definitions_text = ""
    for defItem, definition in concepts_def.items():
        definitions_text += f"{defItem} defined as following: {definition}. "

    # Process relationships to avoid repeating the source
    relations_dict = {}
    for relation in (primekg_relevant_relations + hetionet_relevant_relations):
        source = relation["source"]
        target_node = relation["target_nodes"]
        relation_type = relation["relation"]
        if source in relations_dict:
            relations_dict[source].append(f"{relation_type} with {target_node}")
        else:
            relations_dict[source] = [f"{relation_type} with {target_node}"]

    # Format relations text
    relations_text = ""
    for source, relations in relations_dict.items():
        relations_text += f"{source} has relations: {', '.join(relations)}. "

    # Compose the final relevant text
    if not definitions_text and not relations_text:
        relevant_text = ""
    else:
        if relations_text and definitions_text:
            relevant_text = f"Relations: {relations_text.strip()} Definitions: {definitions_text.strip()}"
        else:
            if definitions_text:
                relevant_text = f"Definitions: {definitions_text.strip()}"
            if relations_text:
                relevant_text = f"Relations: {relations_text.strip()}"
                    
    return relevant_text


# def get_item_metadata(context_entities,question_entities,primekg_relevant_relation,hetionet_relevant_relation):
#     definitions_text = ""
#     concepts_def = get_all_concepts_with_definition(context_entities,question_entities)    
#     for defItem in concepts_def:
#         definition = concepts_def[defItem]
#         text = f"{defItem} defined as following: {definition}."
#         definitions_text += text
        
#     relations_text = ""
#     for relation in primekg_relevant_relation:
#         text = f"{relation["source"]} has relation {relation["relation"]} with {relation["target_nodes"]}." 
#         relations_text +=text 
#     relevant_text = f"Relations : {relations_text} Definitions : {definitions_text}"
    
#     return relations_text


def process_json(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    new_data = list()
    for item in tqdm(data):
        metadata = ""
        context_entities = item["context_entities"]
        question_entities = item["question_entities"]
        
        primekg_relations = item["primekg_relevant_relations_qc"]
        hetionet_relations = item["hetionet_relevant_relations_qc"]

        metadata = get_item_metadata(context_entities,question_entities,primekg_relations,hetionet_relations)
        if metadata != "":
            summury = summarize_text(metadata)
        new_item = {
            "question": item["question"],
            "context": item["context"],
            "answer": item["answer"],
            "type":item.get("type",""),
            "metadata":summury
        }
        
        new_data.append(new_item)
    return new_data

def process_dataset(dataset):
    directory_path = f'{root_folder}/Pre_Processed_Datasets/{dataset}/4_extracted_concepts_relations_filtered/'
    for json_file_path in glob.glob(directory_path + '*.json'):
        new_data = process_json(json_file_path)

        # Create new file path
        directory, filename = os.path.split(json_file_path)
        new_directory = f"{root_folder}/Pre_Processed_Datasets/{dataset}/5_formated_metadata"
        os.makedirs(new_directory,exist_ok=True)

        new_filename = os.path.splitext(filename)[0] + f".json"
        new_file_path = os.path.join(new_directory, new_filename)

        # Save new data to the new file
        with open(new_file_path, 'w') as new_file:
            json.dump(list(new_data), new_file, indent=4)



def load_json_file(filepath):
    """Load data from a JSON file."""
    with open(filepath, 'r') as file:
        return json.load(file)



