
import json
import os
import spacy
import scispacy
import re

import torch
from scispacy.linking import EntityLinker
from scispacy.abbreviation import AbbreviationDetector
from torch.nn.functional import cosine_similarity
from tqdm import tqdm
import numpy as np
from scipy.spatial.distance import cosine
from transformers import AutoTokenizer, AutoModel

# relevant_types = {"T047", "T191", "T061", "T060", "T074", "T058", "T190", "T020"}
# Define relevant biomedical labels
list_datasets = ["medqa_usmle_hf"]#"PubmedQA","BioASQ"
relevant_labels = {
    "AMINO_ACID", "ANATOMICAL_SYSTEM", "CANCER", "CELL", "CELLULAR_COMPONENT",
   "GENE_OR_GENE_PRODUCT", "IMMATERIAL_ANATOMICAL_ENTITY",
    "ORGAN", "PATHOLOGICAL_FORMATION", "SIMPLE_CHEMICAL", "TISSUE",
    "DISEASE", "CHEMICAL", "DNA", "CELL_TYPE", "CELL_LINE", "RNA", "PROTEIN",
    "GGP", "SO", "TAXON", "CHEBI", "GO", "CL"
}

# Load NER models
nlp_craft = spacy.load("en_ner_craft_md")
nlp_jnlpba = spacy.load("en_ner_jnlpba_md")
nlp_bc5cdr = spacy.load("en_ner_bc5cdr_md")
nlp_bionlp13cg = spacy.load("en_ner_bionlp13cg_md")

# Load SciBERT with UMLS linking capabilities
nlp_text_entities_extractor = spacy.load("en_core_sci_scibert")
nlp_text_entities_extractor.add_pipe("scispacy_linker", config={
    "resolve_abbreviations": True,
    "linker_name": "umls",
    "filter_for_definitions": True,
    "threshold": 0.95,
    "max_entities_per_mention": 5
})

linker = nlp_text_entities_extractor.get_pipe("scispacy_linker")
tokenizer = AutoTokenizer.from_pretrained("monologg/biobert_v1.1_pubmed")
model = AutoModel.from_pretrained("monologg/biobert_v1.1_pubmed")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)



def is_full_word(context, entity):
    # Regex pattern to match the entity as a full word with potential surrounding punctuation
    pattern = r'\b' + re.escape(entity) + r'\b'
    return re.search(pattern, context) is not None



def compute_embeddings(text):
    # Tokenize the text and move the tensors to the same device as the model
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Perform the forward pass and compute the embeddings
    with torch.no_grad():
        outputs = model(**inputs)

    # Compute the mean of the last hidden state to get the embeddings
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings

def context_relevance_score(entity_embedding, context_embedding):
    similarity = cosine_similarity(entity_embedding, context_embedding)
    return similarity.item()

# Extract entities using the NER models and filter based on relevant labels

def find_closest_concept(entity, context, window_size=10):
    # Extract context around the entity
    start_index = max(entity.start - window_size, 0)
    end_index = min(entity.end + window_size, len(context))
    context_window = context[start_index:end_index]

    context_embedding = compute_embeddings(context_window)

    # Initialize variables to find the closest concept
    min_distance = float('inf')
    closest_concept = None
    best_cui = ""
    # Check all possible UMLS concepts linked to this entity
    for cui, _ in entity._.kb_ents:
        concept = linker.kb.cui_to_entity[cui]
        concept_embedding = compute_embeddings(concept.canonical_name)

        # Calculate similarity (1 - cosine distance)
        distance = context_relevance_score(concept_embedding, context_embedding)
        if distance < min_distance:
            min_distance = distance
            closest_concept = concept
            best_cui = cui
    return closest_concept,best_cui

# Retrieve metadata for entities, checking for relevant labels and UMLS linkage
def get_biomedical_entity_metadata(text, initial_entities):
    doc = nlp_text_entities_extractor(text)
    list_entities = []
    processed_cuis = set()

    for text, label in initial_entities:
        for ent in doc.ents:
            if ent.text == text:
                closest_concept, closest_cui = find_closest_concept(ent, text)
                if closest_concept and closest_cui not in processed_cuis:
                    processed_cuis.add(closest_cui)
                    entity_obj = {
                        "cui": closest_cui,
                        "concept_id": closest_concept.concept_id,
                        "canonical_name": closest_concept.canonical_name,
                        "definition": closest_concept.definition
                    }
                    list_entities.append({text: entity_obj})
                    print({text: closest_concept.canonical_name})
    return list_entities


def extract_entities(context):
    if context is None:
        return None

    # Collect entities from each NER model
    entities = set()
    for model in [nlp_craft, nlp_jnlpba, nlp_bc5cdr, nlp_bionlp13cg]:
        doc = model(context)
        for ent in doc.ents:
            if is_full_word(context, ent.text):
                if ent.label_ in relevant_labels:
                    entities.add((ent.text, ent.label_))

    # Process these entities to get more metadata from UMLS
    return get_biomedical_entity_metadata(context, entities)


def process_json_file(file_path, output_directory):
    new_data = []

    # Read the entire file content to handle multiple JSON objects
    with open(file_path, 'r', encoding='utf-8') as file:
        file_content = file.read()

    # Attempt to load the JSON data
    try:
        data = json.loads(file_content)
    except json.JSONDecodeError:
        print("Error loading JSON: The file may contain multiple JSON objects or is malformed.")
        return

    # Process each item in the data
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

    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    output_file_path = os.path.join(output_directory, os.path.basename(file_path))

    # Write the modified data back to a new file
    with open(output_file_path, 'w', encoding='utf-8') as file:
        json.dump(new_data, file, indent=4)

def entities_normalization():
    for dataset in list_datasets:
        print(f"Process Dataset {dataset}")
        path_dataset = f"../Pre_Processed_Datasets/{dataset}"
        output_directory = os.path.join(path_dataset, '1_extracted_entities_with_metadata')

        path_preprocessed = f"{path_dataset}/0_Preprocessed"
        # Process each JSON file in the directory
        for filename in os.listdir(path_preprocessed):
            if filename.endswith(".json"):
                file_path = os.path.join(path_preprocessed, filename)
                process_json_file(file_path, output_directory)

if __name__ == '__main__':
    entities_normalization()

