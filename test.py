import json

# Load the source data (the file you want to split)
with open("Pre_Processed_Datasets/BioASQ2024/train.json", "r") as f:
    source_data = json.load(f)

# Load the reference data (JSONL format)
reference_questions = set()
with open("Pre_Processed_Datasets/BioASQ/dev.json", "r") as f:
    for line in f:
        entry = json.loads(line)
        reference_questions.add(entry["sentence1"])  # Collect questions for fast matching

# Initialize train and dev lists
train_data = []
dev_data = []

# Split the source data based on matching questions
for record in source_data:
    if record["question"] in reference_questions:
        dev_data.append(record)
    else:
        train_data.append(record)

# Save the dev data to a new file
with open("dev.json", "w") as f:
    json.dump(dev_data, f, indent=4)

# Save the train data to a new file
with open("train.json", "w") as f:
    json.dump(train_data, f, indent=4)

print(f"Dev data: {len(dev_data)} records")
print(f"Train data: {len(train_data)} records")
