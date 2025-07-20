import json
import os

dataset_path  = "./Pre_Processed_Datasets/medqa_usmle_hf/"

def process_medqa_umls(file_path,output_dir):
    # Open the JSONL file
    with open(f'{dataset_path}{file_path}', 'r') as file:
        list_data = []
        # Iterate through each line in the file
        for line in file:
            # Parse the JSON object from the line
            data = json.loads(line)
            
            # Get the value of the "sent1" field
            text = data["sent1"]
            
            # Split the text into sentences
            sentences = text.split('. ')
            
            # Extract the passage before the question
            passage = ". ".join(sentences[:-1])  # Join all sentences except the last one
            
            # Extract the question
            question = sentences[-1]
            endings = [data[f"ending{i}"] for i in range(4)]
            question +="[SEP]"+ ", ".join(endings)
            
            data["question"] = question
            data["context"] = passage
            list_data.append(data)
            # Save the modified data into a JSON file
        out_path = f"{dataset_path}{output_dir}"
        os.makedirs(out_path,exist_ok = True)
        with open(f"{out_path}{file_path}", 'w') as outfile:
            json.dump(list_data, outfile, indent=4)
            
files = ["dev.json","test.json","train.json"]

for file in files:
    process_medqa_umls(file,"0_Preprocessed/")