# import ollama
# response = ollama.chat(model='medllama2', messages=[
#   {
#     'role': 'user',
#     'content': 'what is the drogue for cancer disease?',
#   },
# ])
# print(response['message']['content'])

#
# import pandas as pd
# from neo4j import GraphDatabase
#
# # Neo4j connection details
# uri = "bolt://localhost:7687"
# username = "neo4j"
# password = "kira.@1830"
#
# # Function to establish connection to Neo4j
# def get_neo4j_session(uri, username, password):
#     driver = GraphDatabase.driver(uri, auth=(username, password))
#     return driver.session()
#
# # Function to create nodes and relationships in Neo4j
# def create_graph(session, drkg_file, entity2src_file, relation_glossary_file):
#     # Read the data
#     drkg_data = pd.read_csv(drkg_file, sep='\t', header=None)
#     entity2src_data = pd.read_csv(entity2src_file, sep='\t', header=None, dtype=str).fillna('')
#     relation_glossary_data = pd.read_csv(relation_glossary_file, sep='\t', header=None)
#
#     # Create nodes and relationships from DRKG file
#     for index, row in drkg_data.iterrows():
#         session.run("MERGE (a:Entity {name: $name1}) "
#                     "MERGE (b:Entity {name: $name2}) "
#                     "MERGE (a)-[:RELATES_TO {type: $relation}]->(b)",
#                     name1=row[0], name2=row[2], relation=row[1])
#
#     # Add source information from entity2src file
#     for index, row in entity2src_data.iterrows():
#         entity = row[0]
#         sources = row[1:]
#         for source in sources:
#             if source:
#                 session.run("MATCH (e:Entity {name: $entity}) "
#                             "MERGE (s:Source {name: $source}) "
#                             "MERGE (e)-[:HAS_SOURCE]->(s)",
#                             entity=entity, source=source)
#
#     # Add relation descriptions from relation_glossary file
#     for index, row in relation_glossary_data.iterrows():
#         relation_type = row[0]
#         description = row[4]
#         session.run("MERGE (r:Relation {type: $type}) "
#                     "SET r.description = $description",
#                     type=relation_type, description=description)
#
# # Main execution
# session = get_neo4j_session(uri, username, password)
# create_graph(session, 'extra_materials/DRKG/drkg.tsv', 'extra_materials/DRKG/entity2src.tsv', 'extra_materials/DRKG/relation_glossary.tsv')
#session.close()

import requests

API_URL = "https://api-inference.huggingface.co/models/dmis-lab/biobert-large-cased-v1.1-squad"
headers = {"Authorization": "Bearer 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"}


def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()


output = query({
    "inputs": {
        "question": "Which splicing factors have been associated with alternative splicing in PLN R14del hearts?",
        "context": "Our work suggests that an intricate interplay of programs controlling gene expression levels and AS is fundamental to organ development, especially for the brain and heart., Bioinformatical analysis pointed to the tissue-specific splicing factors Srrm4 and Nova1 as likely upstream regulators of the observed splicing changes in the PLN-R14del cardiomyocytes. "
    },
})

print(output)

