# import xml.etree.ElementTree as ET
#
# # Load and parse the XML file
# tree = ET.parse('_materials/mesh/desc2024.xml')
# root = tree.getroot()
#
# # Dictionary to hold abbreviations
# abbreviation_dict = {}
#
# # Loop through each descriptor record
# for allowable_qualifier in root.findall('.//AllowableQualifier'):
#     # Find QualifierName and Abbreviation elements
#     qualifier_name_element = allowable_qualifier.find('.//QualifierName/String')
#     abbreviation_element = allowable_qualifier.find('Abbreviation')
#
#     # Extract text content if elements exist
#     if qualifier_name_element is not None and abbreviation_element is not None:
#         qualifier_name = qualifier_name_element.text
#         abbreviation = abbreviation_element.text
#
#         # Add to the dictionary
#         abbreviation_dict[abbreviation] = qualifier_name
# # Print the abbreviation dictionary
# word =  "Srrm4"
#
# # Check if 'IP' exists in the dictionary before accessing it
# if word in abbreviation_dict:
#     print(abbreviation_dict[word])
# else:
#     print(f"Abbreviation {word} not found")
# num_records = len(abbreviation_dict)
# print(num_records)
# text = "In recent studies, researchers have found that DM (Diabetes Mellitus) is linked to an increased risk of developing other chronic conditions such as HTN (Hypertension). Furthermore, it has been observed that patients suffering from T1D (Type 1 Diabetes) often require a different treatment regimen compared to those with Type 2 Diabetes. Managing these conditions effectively is crucial for improving patient outcomes."
import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
model = AutoModel.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext").cuda()

# replace with your own list of entity names
all_names = ["covid-19", "Coronavirus infection", "high fever", "Tumor of posterior wall of oropharynx"]

bs = 128 # batch size during inference
all_embs = []
for i in tqdm(np.arange(0, len(all_names), bs)):
    toks = tokenizer.batch_encode_plus(all_names[i:i+bs],
                                       padding="max_length",
                                       max_length=25,
                                       truncation=True,
                                       return_tensors="pt")
    toks_cuda = {}
    for k,v in toks.items():
        toks_cuda[k] = v.cuda()
    cls_rep = model(**toks_cuda)[0][:,0,:] # use CLS representation as the embedding
    all_embs.append(cls_rep.cpu().detach().numpy())

all_embs = np.concatenate(all_embs, axis=0)
