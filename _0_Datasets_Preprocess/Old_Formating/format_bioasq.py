import json
import os


def extract_data(json_file):
    with open(json_file, 'r',encoding='utf-8') as file:
        data = json.load(file)

    extracted_data = []
    for question in data["questions"]:
        question_text = question["body"]
        context = ", ".join(snippet["text"] for snippet in question["snippets"])
        answer = question.get("exact_answer", None)
        type = question["type"]
        id = question["id"]
        if type == "yesno":
            extracted_data.append({
                "id": id,
                "question": question_text,
                "context": context,
                "answer": answer,
                "type": type,
            })

    return extracted_data


def extract_train_data(path):
    extracted_info_train = extract_data(path)

    # To save the extracted data as a new JSON file
    with open("../../Pre_Processed_Datasets/BioASQ2024/train.json", "w",encoding='utf-8') as outfile:
        json.dump(extracted_info_train, outfile, indent=4)

def process_folder(folder_path):
    all_extracted_data = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            extracted_data = extract_data(file_path)
            all_extracted_data.extend(extracted_data)

    return all_extracted_data
def extract_test_data(path):
    # Specify the folder containing your JSON files
    folder_path = path
    # Extract data from all files in the folder
    consolidated_data = process_folder(folder_path)
    # Save the consolidated data to a new JSON file
    with open("test_.json", "w") as outfile:
        json.dump(consolidated_data, outfile, indent=4)

if __name__ == '__main__':
    extract_train_data("../../Pre_Processed_Datasets/raw_datasets/BioASQ 2024/training12b_new.json")
    extract_test_data("../../Pre_Processed_Datasets/raw_datasets/BioASQ 2024/test")
