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
nlp = spacy.load(model_path)
print("Loading from", model_path)
doc = nlp("Invoice Denk's Environmental Services 317 Chester Drive Lower Burrell, PA 15068 Date 5/4/2009 Invoice # 1614 South Pike Cinemas 130 Cinema Way Sarver, PA 16055 Amount 1,175.00T Sales Tax 70.50 Total $1,245.50 ")
# doc = nlp("Andrea Pirlo played sooccer the other day whilst buying a house for $5000")
print(doc)
print([(X.text, X.label_) for X in doc.ents])
