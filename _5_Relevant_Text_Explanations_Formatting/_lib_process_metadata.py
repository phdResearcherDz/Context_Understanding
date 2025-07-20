from enum import Enum

from transformers import AutoTokenizer, AutoModel
import glob
import json
import os
import pandas as pd
from neo4j import GraphDatabase
from tqdm import tqdm
import torch
from transformers import pipeline

root_folder = ".."
# Load the summarization pipeline and set it to use the GPU if available
device = 0 if torch.cuda.is_available() else -1  # -1 means CPU
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device)

class MetadataMethod(Enum):
    WITH_DEFINITION_RELATION = 1
    WITH_RELATION = 2
    # WITH_RELATION_SUMMARIZE = 3
    # WITH_DEFINITION = 4
    # WITH_DEFINITION_SUMMARIZE = 5
    # WITH_DEFINITION_RELATION_SUMMARIZE = 6

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

def summarize_text(text,context,summarizer=summarizer):
    # Assuming the model's tokenizer is needed to count tokens accurately
    tokenizer = summarizer.tokenizer
    tokens = tokenizer.tokenize(text)
    token_count = len(tokens)
    min_token_threshold = 50
    context_token_count = len(tokenizer.tokenize(context))
    if(context_token_count > 512 ):
        return text
    # Define a minimum token count threshold for summarization
      # Adjust this value as necessary

    max_input_length = min(token_count, 512 - context_token_count)
    if max_input_length < min_token_threshold:
        return  text
    min_input_length = int(max_input_length / 2)
    # Ensure we do not proceed if the text is too small or exceeds the model's maximum input size
    model_max_input_size = tokenizer.model_max_length
    if token_count < min_token_threshold or token_count > model_max_input_size:
        return text

    # Perform summarization with dynamic input length constraints
    summary = summarizer(text, min_length=min_input_length, max_length=max_input_length, do_sample=False)

    del summarizer  # Assuming summarizer is a large object using GPU
    torch.cuda.empty_cache()  # Clear cache if using PyTorch
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
                                                hetionet_relevant_relations,context):
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

    return summarize_text(relevant_text,context)


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
                                       hetionet_relevant_relations,context):
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

    return summarize_text(relevant_text,context)


def get_item_metadata_with_deffinition_relation_summarize(context_entities, question_entities, primekg_relevant_relations,
                                                hetionet_relevant_relations,context):
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

    return summarize_text(relevant_text,context)

def call_metadata_method(method, context_entities, question_entities, primekg_relations, hetionet_relations,context):
    match method:
        case MetadataMethod.WITH_DEFINITION_RELATION:
            return get_item_metadata_with_deffinition_relation(context_entities, question_entities, primekg_relations, hetionet_relations)
        case MetadataMethod.WITH_RELATION:
            return get_item_metadata_with_relation(context_entities, question_entities, primekg_relations, hetionet_relations)
        # case MetadataMethod.WITH_RELATION_SUMMARIZE:
        #     return get_item_metadata_with_relation_summurize(context_entities, question_entities, primekg_relations, hetionet_relations,context)
        case MetadataMethod.WITH_DEFINITION:
            return get_item_metadata_with_deffinition(context_entities, question_entities, primekg_relations, hetionet_relations)
        # case MetadataMethod.WITH_DEFINITION_SUMMARIZE:
        #     return get_item_metadata_with_deffinition_summarize(context_entities, question_entities, primekg_relations, hetionet_relations,context)
        # case MetadataMethod.WITH_DEFINITION_RELATION_SUMMARIZE:
        #     return get_item_metadata_with_deffinition_relation_summarize(context_entities, question_entities, primekg_relations, hetionet_relations,context)
        case _:
            raise ValueError("Invalid method")




def process_json(json_file_path,method,version_filter,kg):
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    print(json_file_path)
    new_data = list()
    for item in tqdm(data):
        metadata = ""
        context = item["context"]
        context_entities = item["context_entities"]
        question_entities = item["question_entities"]
        if version_filter == "similarity":
            version_filter = "sm"

        match kg:
            case "both":
                primekg_relations = item[f"primekg_relevant_relations_{version_filter}"]
                hetionet_relations = item[f"hetionet_relevant_relations_{version_filter}"]
            case "primekg":
                primekg_relations = item[f"primekg_relevant_relations_{version_filter}"]
                hetionet_relations = []
            case "hetionet":
                primekg_relations = []
                hetionet_relations = item[f"hetionet_relevant_relations_{version_filter}"]
            case _:
                raise ValueError("Invalid method")

        metadata = call_metadata_method(method, context_entities, question_entities, primekg_relations,
                                        hetionet_relations,context)

        new_item = {
            "original_question":  item["question"],
            "question": item["question"],
            "context": item["context"],
            "answer": item["answer"],
            "type":item.get("type",""),
            "metadata":metadata
        }
        
        new_data.append(new_item)
    return new_data


def process_json_medqa(json_file_path, method, version_filter, kg):
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    print(json_file_path)
    new_data = list()
    for item in tqdm(data):
        metadata = ""
        context = item["context"]
        context_entities = item["context_entities"]
        question_entities = item["question_entities"]
        if version_filter == "similarity":
            version_filter = "sm"

        match kg:
            case "both":
                primekg_relations = item[f"primekg_relevant_relations_{version_filter}"]
                hetionet_relations = item[f"hetionet_relevant_relations_{version_filter}"]
            case "primekg":
                primekg_relations = item[f"primekg_relevant_relations_{version_filter}"]
                hetionet_relations = []
            case "hetionet":
                primekg_relations = []
                hetionet_relations = item[f"hetionet_relevant_relations_{version_filter}"]
            case _:
                raise ValueError("Invalid method")

        metadata = call_metadata_method(method, context_entities, question_entities, primekg_relations,
                                        hetionet_relations, context)
        item["metadata"] = metadata

        # Specify keys to remove
        keys_to_remove = [
            f"primekg_relevant_relations_{version_filter}",
            f"hetionet_relevant_relations_{version_filter}",
            f"primekg_concepts_context",
            f"hetionet_concepts_context",
            f"primekg_concepts_question",
            f"hetionet_concepts_question",
            f"primekg_concepts_question_with_kg_data",
            f"hetionet_concepts_question_with_kg_data",
            f"primekg_concepts_context_with_kg_data",
            f"hetionet_concepts_context_with_kg_data"
        ]

        # Create a new dictionary without specified keys
        filtered_data = {key: value for key, value in item.items() if key not in keys_to_remove}

        # Reassign filtered data back to item
        item = filtered_data

        # Copy the original question if it exists
        item["original_question"] = item.get("question", "")

        new_data.append(item)
    return new_data

# kg = > Both, primekg, hetionet
def process_dataset(dataset,type_formating,version_filter,kg="both"):
    directory_path = f'{root_folder}/Pre_Processed_Datasets/{dataset}/4_extracted_concepts_relations_filtered_{version_filter}/'
    for json_file_path in glob.glob(directory_path + '*.json'):

        # new_data = process_json(json_file_path,type_formating,version_filter,kg)
        new_data = process_json_medqa(json_file_path,type_formating,version_filter,kg)

        # Create new file path
        directory, filename = os.path.split(json_file_path)
        new_directory = f"{root_folder}/Pre_Processed_Datasets/{dataset}/5_formated_metadata_{type_formating}_v{version_filter}_{kg}"
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



