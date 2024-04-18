import json
import re


def read_jsonl_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            yield json.loads(line)
def separate_context_and_question(data):
    updated_data = []
    for item in data:
        text = item['question']
        # Find the last WH-question
        matches = list(re.finditer(r'\b(What|Which|Who|Where|When|Why|How)\b', text,re.IGNORECASE))
        if matches:
            last_match = matches[-1]
            context = text[:last_match.start()].strip()
            question = text[last_match.start():].strip()
        else:
            context = text
            question = "Choose from the following options?"

        updated_item = {
            "question": question,
            "context": context,
            "choices": list(item["options"].values()),
            "answer_index": ord(item["answer_idx"]) - ord('A'),
            "answer": item["answer"]
        }
        updated_data.append(updated_item)

    return updated_data
type = "test"
# Path to your JSON file
file_path = f'../Datasets/MedQA/{type}.jsonl'

# Read data from file
data = read_jsonl_file(file_path)

# Process data to separate context and question
separated_data = separate_context_and_question(data)

# Convert to JSON
json_output = json.dumps(separated_data, indent=4)

# Print or save the output
# Optionally, save to a new file
with open(f'{type}.json', 'w', encoding='utf-8') as f:
    json.dump(separated_data, f, indent=4)
