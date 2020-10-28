""" --------------------------------------------- JSON_Testing.py ---------
    |
    |   Purpose: Simple code to test the trained model.
    |
    |   Developer:
    |       Carlos García - https://github.com/cxrlos
    |
    *------------------------------------------------------------------ """
import spacy
import os

model_path = os.getcwd() + '/model'
print("Loading from", model_path)
nlp_updated = spacy.load(model_path)
doc = nlp_updated("Fridge can be ordered in FlipKart" )
print(doc)
print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
