from transformers import AutoTokenizer, AutoModel
import glob
import json
import os
import pandas as pd
from neo4j import GraphDatabase
from tqdm import tqdm
import torch

root_folder = ".."

def get_all_concepts(context_ent,question_ent):
    allConcepts = []

    for item in context_ent:
        allConcepts.append(item)
    for item in question_ent:
        allConcepts.append(item)
    return allConcepts

def process_json(json_file_path, kg_name):
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    new_data = list()
    for item in tqdm(data):
        question_medical_concepts = item.get(f"{kg_name}_concepts_question", '')
        context_medical_concepts = item.get(f"{kg_name}_concepts_context", '')

        question_medical_concepts_relations = item.get(f"{kg_name}_concepts_question_with_kg_data", '')
        context_medical_concepts_relations = item.get(f"{kg_name}_concepts_context_with_kg_data", '')

        relevant_relations_filtered_qc = []
        all_concepts = get_all_concepts(context_medical_concepts,question_medical_concepts)
        print(all_concepts)
        #select relvant relations from context
        for concept in context_medical_concepts_relations:
            triples = context_medical_concepts_relations[concept]["triples"]
            for triple in triples:
                if triple["target_nodes"] in all_concepts:
                    relevant_relations_filtered_qc.append(triple)

        #select relevant relations from question
        for concept in question_medical_concepts_relations:
            triples = question_medical_concepts_relations[concept]["triples"]
            for triple in triples:
                if triple["target_nodes"] in all_concepts:
                    relevant_relations_filtered_qc.append(triple)

        item[f"{kg_name}_relevant_relations_qc"] = relevant_relations_filtered_qc

        new_data.append(item)
    return new_data

def process_dataset(dataset, kg_name):
    directory_path = f'{root_folder}/Pre_Processed_Datasets/{dataset}/3_extracted_concepts_relations/'
    for json_file_path in glob.glob(directory_path + '*.json'):
        new_data = process_json(json_file_path, kg_name)

        # Create new file path
        directory, filename = os.path.split(json_file_path)
        new_directory = f"{root_folder}/Pre_Processed_Datasets/{dataset}/4_extracted_concepts_relations_filtered_qc"
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
        item[f"{kg2}_relevant_relations_qc"] = item2[f"{kg2}_relevant_relations_qc"]
        merged_data.append(item)
    return merged_data
