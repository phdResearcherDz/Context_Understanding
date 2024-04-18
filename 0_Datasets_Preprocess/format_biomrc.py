#BioMRC - From Huffing Face

from datasets import load_dataset

# Replace 'dataset_name' with the actual name of the dataset
dataset = load_dataset('biomrc',"biomrc_small_A")

train_dataset = dataset['train']
test_dataset = dataset.get('test', None)
validation_dataset = dataset.get('validation', None)

# Export the train dataset to a JSON file
train_dataset.to_json('train_dataset.json')

# Similarly, export test and validation sets if they exist
if test_dataset is not None:
    test_dataset.to_json('test_dataset.json')
if validation_dataset is not None:
    validation_dataset.to_json('validation_dataset.json')
