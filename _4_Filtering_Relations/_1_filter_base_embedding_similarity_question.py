from _lib_relation_filter_similarity import *

datasets = ["medqa_usmle_hf"]#"BioASQ", "PubmedQA"


def ProcessPrimeKG():
    for dataset in datasets:
        process_dataset(dataset, kg_name="primekg")


def ProcessHetionet():
    for dataset in datasets:
        process_dataset(dataset, kg_name="hetionet")


def filter_relations_sm():
    ProcessPrimeKG()
    ProcessHetionet()

    kgs = ["primekg", "hetionet"]
    for dataset in datasets:
        directory_path = f"{root_folder}/Pre_Processed_Datasets/{dataset}/4_extracted_concepts_relations_filtered_similarity"

        # Load data from files
        data1 = load_json_file(f'{directory_path}\\test_{kgs[0]}_{dataset}_with_relations_filtered.json')
        data2 = load_json_file(f'{directory_path}\\test_{kgs[1]}_{dataset}_with_relations_filtered.json')

        # Merge data
        merged_data = merge_records(data1, data2, type="test", kg1=kgs[0], kg2=kgs[1])
        output_file_path = os.path.join(directory_path, f"test.json")
        with open(output_file_path, 'w') as outfile:
            json.dump(merged_data, outfile, indent=4)

        data1 = load_json_file(f'{directory_path}\\dev_{kgs[0]}_{dataset}_with_relations_filtered.json')
        data2 = load_json_file(f'{directory_path}\\dev_{kgs[1]}_{dataset}_with_relations_filtered.json')

        # Merge data
        merged_data = merge_records(data1, data2, type="dev", kg1=kgs[0], kg2=kgs[1])
        output_file_path = os.path.join(directory_path, f"dev.json")
        with open(output_file_path, 'w') as outfile:
            json.dump(merged_data, outfile, indent=4)

        data1 = load_json_file(f'{directory_path}\\train_{kgs[0]}_{dataset}_with_relations_filtered.json')
        data2 = load_json_file(f'{directory_path}\\train_{kgs[1]}_{dataset}_with_relations_filtered.json')

        merged_data = merge_records(data1, data2, type="train", kg1=kgs[0], kg2=kgs[1])
        output_file_path = os.path.join(directory_path, f"train.json")
        with open(output_file_path, 'w') as outfile:
            json.dump(merged_data, outfile, indent=4)


if __name__ == "__main__":
    filter_relations_sm()
