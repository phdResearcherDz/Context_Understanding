from _lib_process_metadata import *

datasets = ["PubmedQA"]  # "BioASQ", 


def ProcessMetaData():
    for dataset in datasets:
        process_dataset(dataset)



if __name__ == "__main__":
    ProcessMetaData()