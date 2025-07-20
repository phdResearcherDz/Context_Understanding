from _lib_process_metadata import *

datasets = ["BioASQ2024","BioASQ_BLURB","PubmedQA"]  #"


def ProcessMetaData():
    for method in MetadataMethod:
        for dataset in datasets:
            # process_dataset(dataset,method,"qc")
            # process_dataset(dataset,method,"qc",kg="primekg")
            # process_dataset(dataset,method,"qc",kg="hetionet")
            # process_dataset(dataset,method,"similarity")
            # process_dataset(dataset,method,"similarity",kg="primekg")
            # process_dataset(dataset,method,"similarity",kg="hetionet")
            #
            # process_dataset(dataset,method,"clustered",kg="primekg")
            # process_dataset(dataset,method,"clustered",kg="hetionet")

            process_dataset(dataset,method,"clustered")
            process_dataset(dataset,method,"dense_retrieval")
            process_dataset(dataset,method,"random")
            process_dataset(dataset,method,"tfidf")
            process_dataset(dataset,method,"top_n")



if __name__ == "__main__":
    ProcessMetaData()