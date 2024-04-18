import glob
import json
import os
import threading

from neo4j import GraphDatabase
from tqdm import tqdm

allowed_attributes = [
    "actions", "affinity_nM",  "chromosome", "class_type", "description",
    "log2_fold_change",
    "method",  "similarity",
    "unbiased", "z_score"
]

def connect_to_neo4j(uri, username, password):
    return GraphDatabase.driver(uri, auth=(username, password))


def find_intersection(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    intersection = set1.intersection(set2)
    return list(intersection)


def get_node_details(driver, node_name):
    with driver.session() as session:
        # Query to get node attributes
        node_attributes_result = session.run("MATCH (n {name: $node_name}) RETURN n", node_name=node_name).single()
        if node_attributes_result is None:
            return None  # Node not found

        # Convert to a dictionary and filter based on words_list
        node_attributes_all = dict(node_attributes_result[0].items())
        node_attributes = {key: value for key, value in node_attributes_all.items() if key in allowed_attributes}

        # Query to get related triples where the node is the source
        result = session.run(
            "MATCH (n {name: $node_name})-[r]->(m) RETURN type(r) as relation, collect(m.name) as nodes",
            node_name=node_name)

        # Process the result into the desired format
        triples = []
        for record in result:
            triple = {
                "source": node_name,
                "relation": record["relation"],
                "target_nodes": record["nodes"]
            }
            triples.append(triple)

    return {"attributes": node_attributes, "triples": triples}


def process_json(driver,json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    new_data = list()
    for item in tqdm(data):
        question_medical_concepts_primekg = set(item.get('question_medical_concepts_hetionet', ''))
        context_medical_concepts_primekg = set(item.get('medical_concepts_hetionet', ''))

        question_with_kg = dict()

        for concept in question_medical_concepts_primekg:
            extracted_data_question = get_node_details(driver,concept)
            question_with_kg[concept] = extracted_data_question

        context_with_kg = dict()
        for concept in context_medical_concepts_primekg:
            extracted_data_context = get_node_details(driver, concept)
            context_with_kg[concept] = extracted_data_context

        item["question_concepts_with_kg_hetionet"] = question_with_kg
        item["context_concepts_with_kg_hetionet"] = context_with_kg

        new_data.append(item)
    return new_data

def process_dataset(driver, dataset):
    directory_path = f'../Pre_Processed_Datasets/{dataset}/3_extracted_kg_data/'
    for json_file_path in glob.glob(directory_path + '*.json'):
        new_data = process_json(driver, json_file_path)

        # Create new file path
        directory, filename = os.path.split(json_file_path)
        new_directory = f"../Pre_Processed_Datasets/{dataset}/3_extracted_kg_data/DATA"
        os.makedirs(new_directory,exist_ok=True)

        new_filename = os.path.splitext(filename)[0] + "_with_kg_data.json"
        new_file_path = os.path.join(new_directory, new_filename)

        # Save new data to the new file
        with open(new_file_path, 'w') as new_file:
            json.dump(list(new_data), new_file, indent=4)


def main():
    uri = "bolt://neo4j.het.io:7687"
    username = "neo4j"
    password = ""

    driver = connect_to_neo4j(uri, username, password)

    datasets = ["BIOMRC"]#BioASQ,"QA4MRE-Alz","MEDQA" , "MedMCQA"

    threads = []
    for dataset in datasets:
        thread = threading.Thread(target=process_dataset, args=(driver, dataset))
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

if __name__ == "__main__":
    main()
