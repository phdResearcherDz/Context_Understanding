import json
import os
import string

import nltk
import spacy
import stanza
from nltk.corpus import stopwords
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

list_datasets = ["PubmedQA","BioASQ","MedMCQA","MEDQA", "QA4MRE-Alz"]#,
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


def extract_entities(context):
    if context is None:
        return None

    nlp = spacy.load("en_core_sci_sm")
    doc = nlp(context)
    medical_entities = [ent.text for ent in doc.ents]
    return medical_entities


def process_json_file(file_path, output_directory):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        new_data = []
        for item in tqdm(data):
            context = item['context']
            entities = extract_entities(context)
            if entities is None:
                continue
            item['context_entities'] = entities
            new_data.append(item)

    os.makedirs(output_directory, exist_ok=True)
    output_file_path = os.path.join(output_directory, os.path.basename(file_path))
    with open(output_file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)


if __name__ == '__main__':
    for dataset in list_datasets:
        path_dataset = f"../Pre_Processed_Datasets/{dataset}"
        output_directory = os.path.join(path_dataset, '1_extracted_entities')

        # Process each JSON file in the directory
        for filename in os.listdir(path_dataset):
            if filename.endswith(".json"):
                file_path = os.path.join(path_dataset, filename)
                process_json_file(file_path, output_directory)
