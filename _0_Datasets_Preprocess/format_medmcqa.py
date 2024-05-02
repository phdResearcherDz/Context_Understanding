import json

def process_file(file_path):
    extracted_data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                data = json.loads(line)
                question_text = data["question"]
                context = data.get("exp", None)
                choices = [data.get(f"op{i}", None) for i in ['a', 'b', 'c', 'd']]
                answer_index = data.get("cop", None)
                answer_text = choices[answer_index] if 0 <= answer_index < len(choices) else None

                extracted_data.append({
                    "question": question_text,
                    "context": context,
                    "choices": choices,
                    "answer_index": answer_index,
                    "answer": answer_text
                })
            except json.JSONDecodeError:
                continue  # Skip lines that are not valid JSON

    return extracted_data

# Specify the path to your file
dev_path = '../Datasets/MedMCQA/dev.json'
test_path = '../Datasets/MedMCQA/test.json'
train_path = '../Datasets/MedMCQA/train.json'

# Extract data from the file
dev_data = process_file(dev_path)
#test_data = process_file(test_path)
train_data = process_file(train_path)

# Optionally, save the extracted data to a new JSON file
with open("../Pre_Processed_Datasets/MEDQA/dev.json", "w") as outfile:
    json.dump(dev_data, outfile, indent=4)

# with open("test.json", "w") as outfile:
#     json.dump(test_data, outfile, indent=4)

with open("../Pre_Processed_Datasets/MEDQA/train.json", "w") as outfile:
    json.dump(train_data, outfile, indent=4)


