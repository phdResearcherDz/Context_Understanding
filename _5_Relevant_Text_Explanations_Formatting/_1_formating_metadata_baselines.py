from _lib_process_metadata_baselines import *

datasets = ["BioASQ2024","BioASQ_BLURB","PubmedQA"]  #"


def ProcessMetaData():
    for method in MetadataMethod:
        for dataset in datasets:
            process_dataset(dataset,method,"clustered")
            process_dataset(dataset,method,"dense_retrieval")
            process_dataset(dataset,method,"random")
            process_dataset(dataset,method,"tfidf")
            process_dataset(dataset,method,"top_n")



if __name__ == "__main__":
    ProcessMetaData()