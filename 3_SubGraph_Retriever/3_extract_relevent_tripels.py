from transformers import AutoTokenizer, AutoModel
import torch
import glob
import json
import os
from tqdm import tqdm

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize tokenizer and model and move the model to the specified device
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/msmarco-distilbert-base-tas-b")
model = AutoModel.from_pretrained("sentence-transformers/msmarco-distilbert-base-tas-b")
model.to(device)

# Function for CLS Pooling
def cls_pooling(model_output):
    return model_output.last_hidden_state[:, 0]

# Function to encode text and compute embeddings
def encode(texts):
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    with torch.no_grad():
        model_output = model(**encoded_input, return_dict=True)

    # Free up memory
    del encoded_input
    torch.cuda.empty_cache()
    return cls_pooling(model_output)

# Function to find relevant triples based on similarity
def get_relevant_triples(question, triples, top_k=5, batch_size=32):
    if not triples:
        return []

    triple_texts = []
    triple_mapped = []
    for triple in triples:
        for target in triple['target_nodes']:
            elm = f"{triple['source']} {triple['relation']} {target}"
            triple_mapped.append({"source":triple['source'],"relation":triple['relation'],"target":target})
            triple_texts.append(elm)
    question_emb = encode([question])
    question_emb = question_emb.to(device)

    # Process in smaller batches
    scores = []
    for i in range(0, len(triple_texts), batch_size):
        batch_texts = triple_texts[i:i + batch_size]
        triple_embs = encode(batch_texts)
        triple_embs = triple_embs.to(device)

        batch_scores = torch.mm(question_emb, triple_embs.transpose(0, 1))[0].cpu().tolist()
        scores.extend(batch_scores)

        # Clear cache
        del triple_embs
        torch.cuda.empty_cache()

    if not scores:
        return []

    triple_score_pairs = sorted(zip(triple_mapped, scores), key=lambda x: x[1], reverse=True)
    # Include scores in the returned data
    return [{'triple': triple, 'score': score} for triple, score in triple_score_pairs[:top_k]]
def process_json(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    new_data = []
    for item in tqdm(data):
        question = item['question']

        for key in ['question_concepts_with_kg_primekg', 'context_concepts_with_kg_primekg',
                    'question_concepts_with_kg_hetionet', 'context_concepts_with_kg_hetionet']:
            for concept, info in item.get(key, {}).items():
                relevant_triples_with_scores = get_relevant_triples(question, info['triples'])
                item[key][concept]['triples'] = relevant_triples_with_scores

                for triple_info in relevant_triples_with_scores:
                    triple = triple_info['triple']
                    score = triple_info['score']

        new_data.append(item)
    return new_data


# Function to process each dataset
def process_dataset(dataset):
    directory_path = f'../Pre_Processed_Datasets/{dataset}/3_extracted_kg_data/'
    for json_file_path in glob.glob(directory_path + '*.json'):
        new_data = process_json(json_file_path)

        directory, filename = os.path.split(json_file_path)
        new_directory = f"../Pre_Processed_Datasets/{dataset}/4_extracted_kg_data_relevant/"
        os.makedirs(new_directory, exist_ok=True)
        new_filename = os.path.splitext(filename)[0] + "_with_kg_data_relevant.json"
        new_file_path = os.path.join(new_directory, new_filename)
        with open(new_file_path, 'w') as new_file:
            json.dump(new_data, new_file, indent=4)

# Main function
def main():
    datasets = ["BIOMRC"]
    for dataset in datasets:
        process_dataset(dataset)

if __name__ == "__main__":
    main()
