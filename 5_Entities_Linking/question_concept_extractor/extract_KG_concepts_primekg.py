import glob
import json
import os
import threading
import pandas as pd
from neo4j import GraphDatabase
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


# Function to connect to Neo4j Database
def connect_to_neo4j(uri, username, password):
    return GraphDatabase.driver(uri, auth=(username, password))


# Function to fetch nodes from Neo4j
def fetch_nodes(driver):
    with driver.session() as session:
        result = session.execute_read(lambda tx: tx.run("MATCH (n) RETURN n").data())
        nodes = [record['n'] for record in result]
    return pd.DataFrame(nodes)


# Function to compute n-gram similarity
def batch_similarity_check(vectorizer, df_nodes, words):
    node_vectors = vectorizer.transform(df_nodes['node_name'])
    word_vectors = vectorizer.transform(words)
    similarities = cosine_similarity(word_vectors, node_vectors)
    return similarities


def filter_context_words(words):
    # Create a new set with words that are not numbers
    filtered_words = {word for word in words if not word.isdigit()}
    return filtered_words

def find_intersection(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    intersection = set1.intersection(set2)
    return list(intersection)

def get_node_details(driver, node_name):
    with driver.session() as session:
        # Query to get node attributes
        node_attributes_result = session.run("MATCH (n {node_name: $node_name}) RETURN n", node_name=node_name).single()
        if node_attributes_result is None:
            return None  # Node not found
        node_attributes = dict(node_attributes_result[0].items())  # Convert to a dictionary that is JSON serializable

        # Query to get related triples where the node is the source
        result = session.run("MATCH (n {node_name: $node_name})-[r]->(m) RETURN type(r) as relation, collect(m.node_name) as nodes", node_name=node_name)

        # Process the result into the desired format
        triples = []
        for record in result:
            triple = {
                "source": node_name,
                "relation": record["relation"],
                "target_nodes": record["nodes"]
            }
            triples.append(triple)

    return {"attributes": node_attributes, "triples": triples}


# Function to process JSON file
def process_json(driver, file_path, df_nodes):
    with open(file_path, 'r') as file:
        data = json.load(file)

    vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(2, 3))
    vectorizer.fit(df_nodes['node_name'])  # Fit only on node names
    node_vectors = vectorizer.transform(df_nodes['node_name'])

    new_data = list()  # Represent the new set of items and questions
    for item in tqdm(data):
        context_entities = set(item.get('question_entities', []))
        if(len(context_entities) == 0) :
            item["question_medical_concepts_primekg"] = []
            new_data.append(item)
            continue
        filtered_words = filter_context_words(context_entities)

        word_vectors = vectorizer.transform(filtered_words)
        similarities = cosine_similarity(word_vectors, node_vectors)

        # Find the most similar node for each word
        most_similar_nodes = df_nodes['node_name'][similarities.argmax(axis=1)]
        high_similarity_flags = similarities.max(axis=1) > 0.89  # Threshold check
        medical_concepts = most_similar_nodes[high_similarity_flags]
        item["question_medical_concepts_primekg"] = list(medical_concepts)
        new_data.append(item)

    return new_data


# Main function

def process_dataset(driver, dataset, df_nodes):
    directory_path = f'../../Pre_Processed_Datasets/{dataset}/2_extracted_concepts/'
    for json_file_path in glob.glob(directory_path + '*.json'):
        new_data = process_json(driver, json_file_path, df_nodes)

        # Create new file path
        directory, filename = os.path.split(json_file_path)
        new_directory = f"../../Pre_Processed_Datasets/{dataset}/2_extracted_concepts/QE"
        os.makedirs(new_directory,exist_ok=True)

        new_filename = os.path.splitext(filename)[0] + "_with_concept_question.json"
        new_file_path = os.path.join(new_directory, new_filename)

        # Save new data to the new file
        with open(new_file_path, 'w') as new_file:
            json.dump(list(new_data), new_file, indent=4)


def main1():
    uri = "bolt://localhost:7687"
    username = "neo4j"
    password = "kira.@1830"

    driver = connect_to_neo4j(uri, username, password)
    df_nodes = fetch_nodes(driver)

    datasets = ["BIOMRC"]#, "QA4MRE-Alz","MEDQA", "MedMCQA"

    threads = []
    for dataset in datasets:
        thread = threading.Thread(target=process_dataset, args=(driver, dataset, df_nodes))
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

if __name__ == "__main__":
    main1()
