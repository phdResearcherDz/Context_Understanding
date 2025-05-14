import json

# Load the first JSON file
with open('ori_pqal.json', 'r') as file:
    first_data = json.load(file)

# Load the second JSON file
with open('test_ground_truth.json', 'r') as file:
    second_data = json.load(file)

# Extract matching elements (present in both files)
matching_data = {key: first_data[key] for key in second_data if key in first_data}

# Extract non-matching elements (present in the first file but not in the second file)
non_matching_data = {key: first_data[key] for key in first_data if key not in second_data}

# Save the matching elements to a new file
with open('../../Pre_Processed_Datasets/BioASQ2024/test.json', 'w') as file:
    json.dump(matching_data, file, indent=4)

# Save the non-matching elements to another new file
with open('train.json', 'w') as file:
    json.dump(non_matching_data, file, indent=4)
