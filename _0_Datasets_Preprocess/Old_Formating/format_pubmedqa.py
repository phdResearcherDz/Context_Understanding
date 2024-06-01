import json
import os

def extract_data_from_file(json_file):
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)

    extracted_data = []
    for key, value in data.items():
        question_text = value["QUESTION"]
        context = " ".join(value["CONTEXTS"])
        answer = value["final_decision"]

        extracted_data.append({
            "question": question_text,
            "context": context,
            "answer": answer,
        })

    return extracted_data

def process_folder(folder_path):
    all_extracted_data = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            extracted_data = extract_data_from_file(file_path)
            all_extracted_data.extend(extracted_data)

    return all_extracted_data

# Specify the folder containing your JSON files
train_data = '../Datasets/PubmedQA/pubmedqa-master/data/ori_pqal.json'
test_data = '../Datasets/PubmedQA/pubmedqa-master/data/ori_pqal.json'


# Extract data from all files in the folder
consolidated_data = extract_data_from_file(train_data)
consolidated_data2 = extract_data_from_file(test_data)

# Save the consolidated data to a new JSON file
with open("../Pre_Processed_Datasets/PubmedQA/train.json", "w") as outfile:
    json.dump(consolidated_data, outfile, indent=4)

with open("../Pre_Processed_Datasets/PubmedQA/test.json", "w") as outfile:
    json.dump(consolidated_data2, outfile, indent=4)
