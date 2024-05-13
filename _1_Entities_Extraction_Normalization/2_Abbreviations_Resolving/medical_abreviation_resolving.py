from _lib_abreviation_expantion import *

import json
import os
import spacy
from tqdm import tqdm

list_datasets = ["PubmedQA","BioASQ"]#,


def process_json_file(file_path, output_directory):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        new_data = []
        for item in tqdm(data):
            context = item['context']
            question = item['question']
           
            new_question,resolved_question_abbreviations = process_abbreviations(question)
            new_context,resolved_context_abbreviations = process_abbreviations(context)
            print(resolved_context_abbreviations)
            print(resolved_question_abbreviations)
            
            item['context_abr_exapnded'] = new_context
            item['question_abr_exapnded'] = new_question

            item['resolved_question_abbreviations'] = resolved_question_abbreviations
            item['resolved_context_abbreviations'] = resolved_context_abbreviations
            
            new_data.append(item)

    os.makedirs(output_directory, exist_ok=True)
    output_file_path = os.path.join(output_directory, os.path.basename(file_path))
    with open(output_file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)


if __name__ == '__main__':
    for dataset in list_datasets:
        path_dataset = f"Pre_Processed_Datasets/{dataset}"
        output_directory = os.path.join(path_dataset, '1_abbreviation_resolving')

        # Process each JSON file in the directory
        for filename in os.listdir(path_dataset):
            if filename.endswith(".json"):
                file_path = os.path.join(path_dataset, filename)
                process_json_file(file_path, output_directory)
