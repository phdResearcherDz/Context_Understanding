from enum import Enum

from transformers import AutoTokenizer, AutoModel
import glob
import json
import os
import pandas as pd
from neo4j import GraphDatabase
from tqdm import tqdm
import torch
from transformers import  pipeline

root_folder = ".."
# Load the summarization pipeline and set it to use the GPU if available
device = 0 if torch.cuda.is_available() else -1  # -1 means CPU
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device)

class MetadataMethod(Enum):
    WITH_DEFINITION_RELATION = 1
    WITH_RELATION = 2
    WITH_RELATION_SUMMARIZE = 3
    WITH_DEFINITION = 4
    WITH_DEFINITION_SUMMARIZE = 5
    WITH_DEFINITION_RELATION_SUMMARIZE = 6

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
    # Assuming the model's tokenizer is needed to count tokens accurately
    tokenizer = summarizer.tokenizer
    tokens = tokenizer.tokenize(text)
    token_count = len(tokens)

    # Define a minimum token count threshold for summarization
    min_token_threshold = 50  # Adjust this value as necessary

    # Ensure we do not proceed if the text is too small or exceeds the model's maximum input size
    model_max_input_size = tokenizer.model_max_length
    if token_count < min_token_threshold or token_count > model_max_input_size:
        return text

    # Set max input length to half of the token count for dynamic summarization
    max_input_length = min(token_count // 2, model_max_input_size)

    # Set a minimum input length ensuring it is always less than max_input_length
    min_input_length = max(25, max_input_length // 2)  # Set some reasonable fraction of max_input_length

    # Adjust min_input_length if necessary to ensure it is less than max_input_length
    if min_input_length >= max_input_length:
        min_input_length = max_input_length // 2

    # Perform summarization with dynamic input length constraints
    summary = summarizer(text, min_length=min_input_length, max_length=max_input_length, do_sample=False)
    return summary[0]["summary_text"]

def get_item_metadata_with_deffinition_relation(context_entities, question_entities, primekg_relevant_relations, hetionet_relevant_relations):
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


def get_item_metadata_with_relation(context_entities, question_entities, primekg_relevant_relations,
                                                hetionet_relevant_relations):
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
    relevant_text= ""
    for source, relations in relations_dict.items():
        relations_text += f"{source} has relations: {', '.join(relations)}. "

    if relations_text:
        relevant_text = f"Relations: {relations_text.strip()}"

    return relevant_text

def get_item_metadata_with_relation_summurize(context_entities, question_entities, primekg_relevant_relations,
                                                hetionet_relevant_relations):
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
    relevant_text = ""
    for source, relations in relations_dict.items():
        relations_text += f"{source} has relations: {', '.join(relations)}. "

    if relations_text:
        relevant_text = f"Relations: {relations_text.strip()}"

    return summarize_text(relevant_text)


def get_item_metadata_with_deffinition(context_entities, question_entities, primekg_relevant_relations,
                                       hetionet_relevant_relations):
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

    relevant_text = ""

    if definitions_text:
        relevant_text = f"Definitions: {definitions_text.strip()}"

    return relevant_text


def get_item_metadata_with_deffinition_summarize(context_entities, question_entities, primekg_relevant_relations,
                                       hetionet_relevant_relations):
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

    relevant_text = ""

    if definitions_text:
        relevant_text = f"Definitions: {definitions_text.strip()}"

    return summarize_text(relevant_text)


def get_item_metadata_with_deffinition_relation_summarize(context_entities, question_entities, primekg_relevant_relations,
                                                hetionet_relevant_relations):
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

    return summarize_text(relations_text)

def call_metadata_method(method, context_entities, question_entities, primekg_relations, hetionet_relations):
    match method:
        case MetadataMethod.WITH_DEFINITION_RELATION:
            return get_item_metadata_with_deffinition_relation(context_entities, question_entities, primekg_relations, hetionet_relations)
        case MetadataMethod.WITH_RELATION:
            return get_item_metadata_with_relation(context_entities, question_entities, primekg_relations, hetionet_relations)
        case MetadataMethod.WITH_RELATION_SUMMARIZE:
            return get_item_metadata_with_relation_summurize(context_entities, question_entities, primekg_relations, hetionet_relations)
        case MetadataMethod.WITH_DEFINITION:
            return get_item_metadata_with_deffinition(context_entities, question_entities, primekg_relations, hetionet_relations)
        case MetadataMethod.WITH_DEFINITION_SUMMARIZE:
            return get_item_metadata_with_deffinition_summarize(context_entities, question_entities, primekg_relations, hetionet_relations)
        case MetadataMethod.WITH_DEFINITION_RELATION_SUMMARIZE:
            return get_item_metadata_with_deffinition_relation_summarize(context_entities, question_entities, primekg_relations, hetionet_relations)
        case _:
            raise ValueError("Invalid method")

def process_json(json_file_path,method,version_filter):
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    print(json_file_path)
    new_data = list()
    for item in tqdm(data):
        metadata = ""
        context_entities = item["context_entities"]
        question_entities = item["question_entities"]
        if version_filter == "similarity":
            version_filter = "sm"
        primekg_relations = item[f"primekg_relevant_relations_{version_filter}"]
        hetionet_relations = item[f"hetionet_relevant_relations_{version_filter}"]

        metadata = call_metadata_method(method, context_entities, question_entities, primekg_relations,
                                        hetionet_relations)
        new_item = {
            "question": item["question"],
            "context": item["context"],
            "answer": item["answer"],
            "type":item.get("type",""),
            "metadata":metadata
        }
        
        new_data.append(new_item)
    return new_data

def process_dataset(dataset,type_formating,version_filter):
    directory_path = f'{root_folder}/Pre_Processed_Datasets/{dataset}/4_extracted_concepts_relations_filtered_{version_filter}/'
    for json_file_path in glob.glob(directory_path + '*.json'):
        new_data = process_json(json_file_path,type_formating,version_filter)

        # Create new file path
        directory, filename = os.path.split(json_file_path)
        new_directory = f"{root_folder}/Pre_Processed_Datasets/{dataset}/5_formated_metadata_{type_formating}_v{version_filter}"
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



