import json
import os
import spacy
from tqdm import tqdm

list_datasets = ["PubmedQA","BioASQ"]#,
# nlp = spacy.load("en_core_sci_sm")
nlp = spacy.load("en_core_sci_scibert")





def extract_entities(context):
    if context is None:
        return None
    doc = nlp(context)
    medical_entities = [ent.text for ent in doc.ents]
    return medical_entities


def process_json_file(file_path, output_directory):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        new_data = []
        for item in tqdm(data):
            context = item['context']
            question = item['question']
            entities_context = extract_entities(context)
            entities_question = extract_entities(question)
            if entities_context is None or entities_question is None:
                new_data.append(item)
                continue

            item['context_entities'] = entities_context
            item['question_entities'] = entities_question

            new_data.append(item)

    os.makedirs(output_directory, exist_ok=True)
    output_file_path = os.path.join(output_directory, os.path.basename(file_path))
    with open(output_file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)


if __name__ == '__main__':
    for dataset in list_datasets:
        path_dataset = f"Pre_Processed_Datasets/{dataset}"
        output_directory = os.path.join(path_dataset, '3_extracted_entities')

        # Process each JSON file in the directory
        for filename in os.listdir(path_dataset):
            if filename.endswith(".json"):
                file_path = os.path.join(path_dataset, filename)
                process_json_file(file_path, output_directory)
