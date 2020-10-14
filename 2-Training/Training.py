""" --------------------------------------------- JSON_Testing.py ---------
    |
    |   Purpose: Test the default en_core_web_sm model with our JSON 
    |       dataset.
    |
    |   Developer:
    |       Carlos García - https://github.com/cxrlos
    |
    *------------------------------------------------------------------ """

import json
import spacy
import random
from spacy.util import minibatch, compounding
from pathlib import Path

nlp = spacy.load('en_core_web_sm')
ner = nlp.get_pipe("ner") # Getting pipeline component

class InputData:
    def __init__(self, d_text, label):
        self.d_text = d_text
        self.label = label

with open('../../Data/experiment_results/labelling/1-1-tags.json') as f:
    data = json.load(f)

# ("Walmart is a leading e-commerce company", {"entities": [(0, 7, "ORG")]})
training_set = []
json_str = json.dumps(data)
for x in data:
    this_word = nlp(x['text'])

    for ent in (x['tags'][2]): # Add elements to NER
        print("\n",x['text'], "| ", ent)
        this_text = InputData(x['text'], ent)
        training_set.append(this_text)
        ner.add_label(ent)

    for ent in this_word.ents:
        print("spaCy pred:", ent.label_)

# Disable pre-built pipelines so that they are not affected in the training
# process, delete to fine tune these (change names of tags to correspond to the
# spaCy ones
pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
unaffected_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]    

with nlp.disable_pipes(*unaffected_pipes):
  for iteration in range(30):
    random.shuffle(training_set)
    losses = {}
    batches = minibatch(training_set, size=compounding(4.0, 32.0, 1.001))
    for batch in batches:
        # texts, annotations = zip(*batch)
        # nlp.update(
        #             texts,  # batch of texts
        #             annotations,  # batch of annotations
        #             drop=0.5,  # dropout - make it harder to memorise data
        #             losses=losses,
        #         )
        print("Losses", losses)
