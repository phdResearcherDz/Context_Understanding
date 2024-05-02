import spacy

nlp = spacy.load("en_core_web_sm")
nlp_coref = spacy.load("en_coreference_web_trf")

doc = nlp("The cats were startled by the dog as it growled at them.")

# use replace_listeners for the coref componentspi
nlp_coref.replace_listeners("transformer", "coref", ["model.tok2vec"])
nlp_coref.replace_listeners("transformer", "span_resolver", ["model.tok2vec"])

# we won't copy over the span cleaner
nlp.add_pipe("coref", source=nlp_coref)
text = "Jessica bought a new car. She loves its color and often drives it to her office. Jessica's colleague, Tom, admires the car too. He thinks it suits her well."
doc = nlp(text)

print(doc.spans)