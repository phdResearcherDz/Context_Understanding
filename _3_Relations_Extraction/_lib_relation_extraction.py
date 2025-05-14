from transformers import AutoTokenizer, AutoModel
import glob
import json
import os
import pandas as pd
from neo4j import GraphDatabase
from tqdm import tqdm
import torch

root_folder = "../"


def connect_to_neo4j(uri, username, password):
    return GraphDatabase.driver(uri, auth=(username, password))


def _get_node_details(driver, node_name,node_name_attribute,allowed_attributes):
    with driver.session() as session:
        # Query to get node attributes
        node_attributes_result = session.run("MATCH (n {node_name: $node_name}) RETURN n", node_name=node_name).single()
        if node_attributes_result is None:
            return None  # Node not found

        # Convert to a dictionary and filter based on words_list
        node_attributes_all = dict(node_attributes_result[0].items())
        node_attributes = {key: value for key, value in node_attributes_all.items() if key in allowed_attributes}

        # Query to get related triples where the node is the source
        result = session.run(
            "MATCH (n {node_name: $node_name})-[r]->(m) RETURN type(r) as relation, collect(m."+node_name_attribute+") as nodes",
            node_name=node_name)

        # Process the result into the desired format
        triples = []
        for record in result:
            for node in record["nodes"]:
                triple = {
                    "source": node_name,
                    "relation": record["relation"],
                    "target_nodes": node
                }
                # print(f"source: {node_name} relation: {record["relation"]} target_nodes:{node}" )
                triples.append(triple)

    return {"attributes": node_attributes, "triples": triples}


def get_node_details(driver, node_name, node_name_attribute, allowed_attributes):
    with driver.session() as session:
        # Query to get related triples where the node is the source
        result = session.run(
            "MATCH (n {"+node_name_attribute+": $node_name})-[r]-(m) RETURN type(r) as relation, collect(m." + node_name_attribute + ") as nodes",
            node_name=node_name)

        # Process the result into the desired format
        triples = set()
        for record in result:
            for node in record["nodes"]:
                # Create a dictionary of the triple
                triple = {
                    "source": node_name,
                    "relation": record["relation"],
                    "target_nodes": node
                }
                # Convert dictionary to frozenset of items to make it hashable
                hashable_triple = frozenset(triple.items())
                triples.add(hashable_triple)  # Add hashable triple to set
                # print(f"source: {node_name}, relation: {record['relation']}, target_nodes: {node}")

    # Convert set of frozensets back to list of dicts for output
    list_of_triples = [dict(triple) for triple in triples]
    return {"triples": list_of_triples}


def process_json(driver,json_file_path, kg_name, node_name_attribute,allowed_attributes):
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    new_data = list()
    for item in tqdm(data):
        question_medical_concepts = item.get(f"{kg_name}_concepts_question", '')
        context_medical_concepts = item.get(f"{kg_name}_concepts_context", '')

        question_with_kg = dict()
        context_with_kg = dict()
        if question_medical_concepts:
            for concept in question_medical_concepts:
                node = question_medical_concepts[concept]
                extracted_data_question = get_node_details(driver,node,node_name_attribute,allowed_attributes)
                question_with_kg[concept] = extracted_data_question

        if context_medical_concepts:
            for concept in context_medical_concepts:
                node = context_medical_concepts[concept]
                extracted_data_context = get_node_details(driver, node,node_name_attribute,allowed_attributes)
                context_with_kg[concept] = extracted_data_context

        item[f"{kg_name}_concepts_question_with_kg_data"] = question_with_kg
        item[f"{kg_name}_concepts_context_with_kg_data"] = context_with_kg

        new_data.append(item)
    return new_data

def process_dataset(driver, dataset, kg_name, node_name_attribute,allowed_attributes):
    directory_path = f'{root_folder}/Pre_Processed_Datasets/{dataset}/2_kg_linked_entities/'
    for json_file_path in glob.glob(directory_path + '*.json'):
        new_data = process_json(driver, json_file_path, kg_name, node_name_attribute,allowed_attributes)

        # Create new file path
        directory, filename = os.path.split(json_file_path)
        new_directory = f"{root_folder}/Pre_Processed_Datasets/{dataset}/3_extracted_concepts_relations"
        os.makedirs(new_directory,exist_ok=True)

        new_filename = os.path.splitext(filename)[0] + f"_{kg_name}_{dataset}_with_relations.json"
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
        # item[f"{kg1}_concepts_context_with_kg_data"] = item[f"{kg1}_concepts_context_with_kg_data"]
        # item[f"{kg1}_concepts_question_with_kg_data"] = item[f"{kg1}_concepts_question_with_kg_data"]

        item[f"{kg2}_concepts_context_with_kg_data"] = item2[f"{kg2}_concepts_context_with_kg_data"]
        item[f"{kg2}_concepts_question_with_kg_data"] = item2[f"{kg2}_concepts_question_with_kg_data"]

        merged_data.append(item)
    return merged_data
