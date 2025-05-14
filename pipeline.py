# from _1_Entities_Extraction_Normalization.Concept_Normalization import entities_normalization
# from _2_Entities_Linking._1_Entities_Linking import entities_linking
from _3_Relations_Extraction._relation_extraction import relation_extraction
from _4_Filtering_Relations._1_filter_base_embedding_similarity_question import filter_relations_sm
from _4_Filtering_Relations._2_filter_base_relation_between_entities import filter_relations_qc

if __name__ == '__main__':
    #entities_normalization()
    #entities_linking()
    relation_extraction()
    filter_relations_sm()
    filter_relations_qc()