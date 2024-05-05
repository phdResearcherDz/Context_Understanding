import json

dataset = "PubmedQA\\5_formated_metadata"
root = ".."
for type in ["train","test"]:
    type_ = type
    path = f"{root}\\Pre_Processed_Datasets\\{dataset}\\"
    # Load data from JSON file
    with open(path+f"{type_}.json", "r") as f:
        data = json.load(f)

    # Convert data
    converted_data = []
    for idx, item in enumerate(data):
        id_str = f"converted_{idx}"
        sentence1 = item["question"]
        sentence2 = f'{item["context"]}[SEP]{item["metadata"]}'
        label = item["answer"]
        json_obj = {
            "id": id_str,
            "sentence1": sentence1,
            "sentence2": sentence2,
            "label": label
        }
        converted_data.append(json.dumps(json_obj))

    # Write to a JSONL file
    with open(f"{type_}.json", "w") as f:
        f.write("\n".join(converted_data))
