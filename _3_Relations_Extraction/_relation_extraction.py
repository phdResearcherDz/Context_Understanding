# from _lib_relation_extraction import *
from _3_Relations_Extraction._lib_relation_extraction import *
datasets = ["BioASQ2024"] # "medqa_usmle_hf" ,


def ProcessPrimeKG():
    uri = "bolt://localhost:7687"
    username = "neo4j"
    password = "kira.@1830"
    allowed_attributes = [
    # "atc_1", "atc_2", "atc_3", "atc_4", "category", "clogp", "description", "half_life", "indication",
    # "mayo_causes", "mayo_complications", "mayo_prevention", "mayo_risk_factors",
    # "mayo_see_doc", "mayo_symptoms", "mechanism_of_action", "molecular_weight",
    # "mondo_definition", "mondo_name", "orphanet_clinical_description",
    # "orphanet_definition", "orphanet_epidemiology", "orphanet_management_and_treatment",
    # "orphanet_prevalence", "pathway", "pharmacodynamics", "protein_binding", "state", "tpsa", "umls_description"
    ]

    driver = connect_to_neo4j(uri, username, password)

    for dataset in datasets:
        process_dataset(driver, dataset,kg_name="primekg",node_name_attribute="node_name",allowed_attributes=allowed_attributes)



def ProcessHetionet():
    uri = "bolt://neo4j.het.io:7687"
    username = "neo4j"
    password = ""
    allowed_attributes = [
    # "actions", "affinity_nM",  "chromosome", "class_type", "description",
    # "log2_fold_change",
    # "method",  "similarity",
    # "unbiased", "z_score"
    ]
    
    driver = connect_to_neo4j(uri, username, password)
    for dataset in datasets:
        process_dataset(driver, dataset,kg_name="hetionet",node_name_attribute="name",allowed_attributes=allowed_attributes)



def relation_extraction():
    ProcessPrimeKG()
    ProcessHetionet()


    kgs = ["primekg","hetionet"]
    for dataset in datasets:
        directory_path = f"{root_folder}/Pre_Processed_Datasets/{dataset}/3_extracted_concepts_relations"
    
        # Load data from files
        data1 = load_json_file(f'{directory_path}\\test_{kgs[0]}_{dataset}_with_relations.json')
        data2 = load_json_file(f'{directory_path}\\test_{kgs[1]}_{dataset}_with_relations.json')
    
        # Merge data
        merged_data = merge_records(data1, data2,type="test")
        output_file_path = os.path.join(directory_path, f"test.json")
        with open(output_file_path, 'w') as outfile:
            json.dump(merged_data, outfile, indent=4)

        data1 = load_json_file(f'{directory_path}\\dev_{kgs[0]}_{dataset}_with_relations.json')
        data2 = load_json_file(f'{directory_path}\\dev_{kgs[1]}_{dataset}_with_relations.json')

        # Merge data
        merged_data = merge_records(data1, data2, type="dev")
        output_file_path = os.path.join(directory_path, f"dev.json")
        with open(output_file_path, 'w') as outfile:
            json.dump(merged_data, outfile, indent=4)

        data1 = load_json_file(f'{directory_path}\\train_{kgs[0]}_{dataset}_with_relations.json')
        data2 = load_json_file(f'{directory_path}\\train_{kgs[1]}_{dataset}_with_relations.json')
    
        merged_data = merge_records(data1, data2,type="train")
        output_file_path = os.path.join(directory_path, f"train.json")
        with open(output_file_path, 'w') as outfile:
            json.dump(merged_data, outfile, indent=4)

        os.remove(f'{directory_path}\\train_{kgs[0]}_{dataset}_with_relations.json')
        os.remove(f'{directory_path}\\train_{kgs[1]}_{dataset}_with_relations.json')

        os.remove(f'{directory_path}\\test_{kgs[0]}_{dataset}_with_relations.json')
        os.remove(f'{directory_path}\\test_{kgs[1]}_{dataset}_with_relations.json')

        os.remove(f'{directory_path}\\dev_{kgs[0]}_{dataset}_with_relations.json')
        os.remove(f'{directory_path}\\dev_{kgs[1]}_{dataset}_with_relations.json')

if __name__ == "__main__":
    relation_extraction()