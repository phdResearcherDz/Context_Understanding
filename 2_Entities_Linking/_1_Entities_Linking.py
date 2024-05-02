from _lib_entities_linking import *

import threading
import pandas as pd
import re

datasets = ["BioASQ","PubmedQA"]#,"PubmedQA"

def ProcessPrimeKG():
    uri = "bolt://localhost:7687"
    username = "neo4j"
    password = "kira.@1830"

    driver = connect_to_neo4j(uri, username, password)
    df_nodes = fetch_nodes(driver)
    
    for dataset in datasets:
        process_dataset(dataset, df_nodes, "primekg","node_name")


def ProcessHetionet():
    uri = "bolt://neo4j.het.io:7687"
    username = "neo4j"
    password = ""
    
    driver = connect_to_neo4j(uri, username, password)
    df_nodes = fetch_nodes(driver)

    for dataset in datasets:
        process_dataset(dataset, df_nodes, "hetionet", "name")


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
        item[f"{kg2}_concepts_context"] = item2[f"{kg2}_concepts_context"]
        item[f"{kg2}_concepts_question"] = item2[f"{kg2}_concepts_question"]
        merged_data.append(item)
    return merged_data

if __name__ == "__main__":
    print("Process Hetionet")
    print("*"*10)
    ProcessHetionet()

    print("Process PrimeKG")
    print("*"*10)
    ProcessPrimeKG()

    print("Merge Data")
    print("*"*10)
    kgs = ["primekg","hetionet"]
    for dataset in datasets:
        directory_path = f"Pre_Processed_Datasets/{dataset}/2_kg_linked_entities"
        
        # Load data from files
        data1 = load_json_file(f'{directory_path}\\test_{kgs[0]}_{dataset}_with_concept.json')
        data2 = load_json_file(f'{directory_path}\\test_{kgs[1]}_{dataset}_with_concept.json')

        # Merge data
        merged_data = merge_records(data1, data2,type="test",kg1=kgs[0],kg2 = kgs[1])
        output_file_path = os.path.join(directory_path, f"test_{dataset}.json")
        with open(output_file_path, 'w') as outfile:
            json.dump(merged_data, outfile, indent=4)
        
        
        data1 = load_json_file(f'{directory_path}\\train_{kgs[0]}_{dataset}_with_concept.json')
        data2 = load_json_file(f'{directory_path}\\train_{kgs[1]}_{dataset}_with_concept.json')
        
        merged_data = merge_records(data1, data2,type="train",kg1=kgs[0],kg2 = kgs[1])
        output_file_path = os.path.join(directory_path, f"train_{dataset}.json")
        with open(output_file_path, 'w') as outfile:
            json.dump(merged_data, outfile, indent=4)