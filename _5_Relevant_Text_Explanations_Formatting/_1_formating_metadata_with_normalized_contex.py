from _lib_process_metadata_with_notmalized_context import *

datasets = ["BioASQ","PubmedQA"]  #"


def ProcessMetaData():
    for method in MetadataMethod:
        for dataset in datasets:
            process_dataset(dataset,method,"qc")
            # process_dataset(dataset,method,"qc",kg="primekg")
            # process_dataset(dataset,method,"qc",kg="hetionet")
            process_dataset(dataset,method,"similarity")
            # process_dataset(dataset,method,"similarity",kg="primekg")
            # process_dataset(dataset,method,"similarity",kg="hetionet")



if __name__ == "__main__":
    ProcessMetaData()