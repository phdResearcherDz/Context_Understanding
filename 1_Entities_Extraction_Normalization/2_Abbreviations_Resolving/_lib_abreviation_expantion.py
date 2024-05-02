import spacy
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from torch.nn.functional import cosine_similarity

# Load the spaCy model and the abbreviation dictionary
nlp = spacy.load("en_core_web_trf")
path_dictionary = "_materials/abbreviation_database/abbreviations.csv"
df = pd.read_csv(path_dictionary, delimiter='|')
abbr_dict = df.groupby('SF')['LF'].apply(list).to_dict()
# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("monologg/biobert_v1.1_pubmed")
model = AutoModel.from_pretrained("monologg/biobert_v1.1_pubmed")

# tokenizer = AutoTokenizer.from_pretrained("GanjinZero/UMLSBert_ENG")
# model = AutoModel.from_pretrained("GanjinZero/UMLSBert_ENG")

# Ensure the model is on the correct device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Model is running on {device}")

def get_embeddings(text):

    # Tokenize the text and move the tensors to the same device as the model
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Perform the forward pass and compute the embeddings
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Compute the mean of the last hidden state to get the embeddings
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings


def context_relevance_score(expansion, context):
    expansion_embedding = get_embeddings(expansion)
    context_embedding = get_embeddings(context)
    similarity = cosine_similarity(expansion_embedding, context_embedding)
    return similarity.item()

def process_abbreviations(text, abbr_dict=abbr_dict, min_score=0.75):
    resolved_abbreviations = {}
    doc = nlp(text)
    all_expansions = {}
    window_size = 10
    for ent in doc.ents:
        abbr = ent.text.lower()
        if abbr in abbr_dict:
            # context = doc[max(ent.start - window_size, 0):min(ent.end + window_size, len(doc))].text
            context = text
            for expansion in abbr_dict[abbr]:
                score = context_relevance_score(expansion, context)
                if abbr not in all_expansions or score > all_expansions[abbr]['score']:
                    all_expansions[abbr] = {'expansion': expansion, 'score': score}

    # Determine the best expansion for each abbreviation based on the highest score observed
    for abbr, data in all_expansions.items():
        if data['score'] >= min_score:
            resolved_abbreviations[abbr] = data['expansion']

    # Replace all occurrences of each abbreviation with its best expansion
    new_text = text
    for abbr, expansion in resolved_abbreviations.items():
        new_text = new_text.replace(abbr, expansion)

    return new_text, resolved_abbreviations

if __name__ == '__main__':
    text = "64% deletions, 18% duplications and 18% point mutations. , Of these, 406 (70.5%) were exonic deletions, 64 (11.1%) were exonic duplications, and one was a deletion/duplication complex rearrangement (0.2%). Small mutations were identified in 105 cases (18.2%), most being nonsense/frameshift types (75.2%). Mutations in splice sites, however, were relatively frequent (20%). , gene deletion rate was 54.3% (391/720), and gene duplication rate was 10.6% (76/720), The rate of deletion mutant occurred in Exon 45-54 was 71.9% (281/391) in all gene deletion patients; meanwhile, the rate of gene duplication occurred in Exon 1-40 was 82.9% (63/76) in all gene duplication ones., In all the patients with gene deletion and duplication, the rate of DMD and IMD was 90.6% (423/467), and BMD, 9.4% (44/467)., The Duchenne Muscular dystrophy (DMD) is the most frequent muscle disorder in childhood caused by mutations in the Xlinked dystrophin gene (about 65% deletions, about 7% duplications, about 26% point mutations and about 2% unknown mutations). , The Duchenne Muscular dystrophy (DMD) is the most frequent muscle disorder in childhood caused by mutations in the Xlinked dystrophin gene (about 65% deletions, about 7% duplications, about 26% point mutations and about 2% unknown mutations)., While in earlier studies equal mutation rates in males and females had been reported, a breakdown by mutation types can better explain the sex ratio of mutations: Point mutations and duplications arise preferentially during spermatogenesis whereas deletions mostly arise in oogenesis., l spectrum. This information is extremely beneficial for basic scientific research, genetic diagnosis, trial planning, clinical care, and gene therapy.METHODS: We collected data from 1400 patients (1042 patients with confirmed unrelated Duchenne muscular dystrophy [DMD] or Becker muscular dystrophy [BMD]) registered in the Chinese Genetic Disease Registry from March 2012 to August 2017 and analyzed the genetic mutational characteristics of these patients.RESULTS: Large deletions were the most frequent type of mutation (72.2%), followed by nonsense mutations (11.9%), exon duplications (8.8%), small deletions (3.0%), splice-site mutations (2.1%), small insertions (1.3%), missense mutations (0.6%), and a combination mutation of a dele, Deletions were the most common mutation type (256, 79%), followed by point mutations (45, 13.9%) and duplications (23, 7.1%)."
    expanded_text, resolved_abbreviations = process_abbreviations(text, abbr_dict)
    print("Expanded Text:", expanded_text)
    print("Resolved Abbreviations:", resolved_abbreviations)
