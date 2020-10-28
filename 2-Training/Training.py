""" --------------------------------------------- JSON_Testing.py ---------
    |
    |   Purpose: Test the default en_core_web_sm model with our JSON 
    |       dataset.
    |
    |   Developer:
    |       Carlos García - https://github.com/cxrlos
    |
    *------------------------------------------------------------------ """

import spacy
import random
import os
import JSON_to_spaCy
from spacy.util import minibatch, compounding

nlp = spacy.load('en_core_web_sm')
ner = nlp.get_pipe("ner") # Getting pipeline component

# Read the file names in the labelling directory to pass them through the 
# conversion algorithm
files = []
for (path, dirnames, filenames) in os.walk('../../Labeling/Data/experiment_results/labelling'):
    files.extend(os.path.join(path, name) for name in filenames)
random.shuffle(files)

# Go through each JSON file, process it with the JSON_to_spaCy() function and 
# store it into a spaCy compatible array
training_set = []
for x in files:
    current_file, json_text = JSON_to_spaCy.ConvertJSON(x)
    for y in current_file:
        training_set.extend((json_text, {"entities": [(y.start, y.end, y.d_type)]}))

# TODO: Make previous assignation work as the following structure
training_set = [
    ("Walmart is a leading e-commerce company", {"entities": [(0, 7, "ORG")]}),
    ("I reached Chennai yesterday.", {"entities": [(19, 28, "GPE")]}),
    ("I recently ordered a book from Amazon", {"entities": [(24,32, "ORG")]}),
    ("I was driving a BMW", {"entities": [(16,19, "PRODUCT")]}),
    ("I ordered this from ShopClues", {"entities": [(20,29, "ORG")]}),
    ("Fridge can be ordered in Amazon ", {"entities": [(0,6, "PRODUCT")]}),
    ("I bought a new Washer", {"entities": [(16,22, "PRODUCT")]}),
    ("I bought a old table", {"entities": [(16,21, "PRODUCT")]}),
    ("I bought a fancy dress", {"entities": [(18,23, "PRODUCT")]}),
    ("I rented a camera", {"entities": [(12,18, "PRODUCT")]}),
    ("I rented a tent for our trip", {"entities": [(12,16, "PRODUCT")]}),
    ("I rented a screwdriver from our neighbour", {"entities": [(12,22, "PRODUCT")]}),
    ("I repaired my computer", {"entities": [(15,23, "PRODUCT")]}),
    ("I got my clock fixed", {"entities": [(16,21, "PRODUCT")]}),
    ("I got my truck fixed", {"entities": [(16,21, "PRODUCT")]}),
    ("Flipkart started it's journey from zero", {"entities": [(0,8, "ORG")]}),
    ("I recently ordered from Max", {"entities": [(24,27, "ORG")]}),
    ("Flipkart is recognized as leader in market",{"entities": [(0,8, "ORG")]}),
    ("I recently ordered from Swiggy", {"entities": [(24,29, "ORG")]})
]

print(training_set)
for i in training_set:
    print(i)

for _, annotations in training_set:
  for ent in annotations.get("entities"):
    ner.add_label(ent[2])

pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
unaffected_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]

with nlp.disable_pipes(*unaffected_pipes):
  for iteration in range(30):
    random.shuffle(training_set)
    losses = {}
    batches = minibatch(training_set, size=compounding(4.0, 32.0, 1.001))
    for batch in batches:
        texts, annotations = zip(*batch)
        nlp.update(texts, annotations, drop=0.5, losses=losses)
        print("Losses", losses)
out_path = os.getcwd() + '/model'
nlp.to_disk(out_path)
print("Saved model to", out_path)

