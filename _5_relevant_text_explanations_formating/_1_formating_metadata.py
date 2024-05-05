from _lib_process_metadata import *

datasets = ["BioASQ","PubmedQA"]  #


def ProcessMetaData():
    for method in MetadataMethod:
        for dataset in datasets:
            process_dataset(dataset,method,"qc")
            process_dataset(dataset,method,"similarity")



if __name__ == "__main__":
    ProcessMetaData()