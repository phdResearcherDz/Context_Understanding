import spacy
import pandas as pd

# Load the spaCy model and the abbreviation dictionary
nlp = spacy.load("en_core_web_lg")
path_dictionary = "_materials/abbreviation_database/abbreviations.csv"
df = pd.read_csv(path_dictionary, delimiter='|')
abbr_dict = df.groupby('SF')['LF'].apply(list).to_dict()



def select_expansion(abbr, context,resolved_abbreviations,number_abbreviation_resolved):
    min_score = 0.6
    if abbr in resolved_abbreviations:
        # If abbreviation has been resolved before, use the same expansion
        return resolved_abbreviations[abbr]

    if abbr not in abbr_dict:
        return abbr  # Return the abbreviation unchanged if not found in the dictionary

    expansions = abbr_dict[abbr]
    best_score = 0
    best_expansion = abbr

    for expansion in expansions:
        score = context_relevance_score(expansion, context)
        if score > best_score:
            best_score = score
            best_expansion = expansion

    if best_score < min_score:
        return abbr

    # Store the best expansion for reuse
    resolved_abbreviations[abbr] = best_expansion
    number_abbreviation_resolved += 1
    return best_expansion

def context_relevance_score(expansion, context):
    expansion_doc = nlp(expansion)
    context_doc = nlp(context)
    similarity = context_doc.similarity(expansion_doc)
    
    return similarity



def expand_abbreviations(text):
    resolved_abbreviations = {}
    number_abbreviation_resolved = 0
    # Initialize 
    doc = nlp(text)
    expanded_text = []
    window_size = 3
    new_text = str(text)
    for token in doc.ents:
        if token.text.lower() in abbr_dict:
            start = max(token.start - window_size, 0)  # Use ent.start and ent.end for entity boundaries
            end = min(token.end + window_size, len(doc))
            context = doc[start:end].text
            expansion = select_expansion(token.text, context,resolved_abbreviations,number_abbreviation_resolved)
            expanded_text.append(expansion)
            new_text = new_text.replace(token.text,expansion)
        else:
            expanded_text.append(token.text)

    return new_text,resolved_abbreviations

if __name__ == '__main__':
    text = "64% deletions, 18% duplications and 18% point mutations. , Of these, 406 (70.5%) were exonic deletions, 64 (11.1%) were exonic duplications, and one was a deletion/duplication complex rearrangement (0.2%). Small mutations were identified in 105 cases (18.2%), most being nonsense/frameshift types (75.2%). Mutations in splice sites, however, were relatively frequent (20%). , gene deletion rate was 54.3% (391/720), and gene duplication rate was 10.6% (76/720), The rate of deletion mutant occurred in Exon 45-54 was 71.9% (281/391) in all gene deletion patients; meanwhile, the rate of gene duplication occurred in Exon 1-40 was 82.9% (63/76) in all gene duplication ones., In all the patients with gene deletion and duplication, the rate of DMD and IMD was 90.6% (423/467), and BMD, 9.4% (44/467)., The Duchenne Muscular dystrophy (DMD) is the most frequent muscle disorder in childhood caused by mutations in the Xlinked dystrophin gene (about 65% deletions, about 7% duplications, about 26% point mutations and about 2% unknown mutations). , The Duchenne Muscular dystrophy (DMD) is the most frequent muscle disorder in childhood caused by mutations in the Xlinked dystrophin gene (about 65% deletions, about 7% duplications, about 26% point mutations and about 2% unknown mutations)., While in earlier studies equal mutation rates in males and females had been reported, a breakdown by mutation types can better explain the sex ratio of mutations: Point mutations and duplications arise preferentially during spermatogenesis whereas deletions mostly arise in oogenesis., l spectrum. This information is extremely beneficial for basic scientific research, genetic diagnosis, trial planning, clinical care, and gene therapy.METHODS: We collected data from 1400 patients (1042 patients with confirmed unrelated Duchenne muscular dystrophy [DMD] or Becker muscular dystrophy [BMD]) registered in the Chinese Genetic Disease Registry from March 2012 to August 2017 and analyzed the genetic mutational characteristics of these patients.RESULTS: Large deletions were the most frequent type of mutation (72.2%), followed by nonsense mutations (11.9%), exon duplications (8.8%), small deletions (3.0%), splice-site mutations (2.1%), small insertions (1.3%), missense mutations (0.6%), and a combination mutation of a dele, Deletions were the most common mutation type (256, 79%), followed by point mutations (45, 13.9%) and duplications (23, 7.1%)."
    expanded_text,_ = expand_abbreviations(text)
    print(f"Resolved Abbreviations: {_}")