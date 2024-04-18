import json
import os
import threading
from tqdm import tqdm
import spacy
import nltk
from nltk.corpus import stopwords

# Download necessary NLTK data
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

list_datasets = ["BIOMRC"]

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
            context = item['question']
            entities = extract_entities(context)
            if entities is None:
                continue
            item['question_entities'] = entities
            new_data.append(item)

    os.makedirs(output_directory, exist_ok=True)
    output_file_path = os.path.join(output_directory, os.path.basename(file_path))
    with open(output_file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)

def process_dataset(dataset):
    path_dataset = f"../Pre_Processed_Datasets/{dataset}/2_extracted_concepts"
    output_directory = os.path.join(path_dataset, '2_extracted_concepts')

    # Process each JSON file in the directory
    for filename in os.listdir(path_dataset):
        if filename.endswith(".json"):
            file_path = os.path.join(path_dataset, filename)
            process_json_file(file_path, output_directory)

if __name__ == '__main__':
    threads = []
    for dataset in list_datasets:
        # Create a new thread for each dataset
        thread = threading.Thread(target=process_dataset, args=(dataset,))
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()
