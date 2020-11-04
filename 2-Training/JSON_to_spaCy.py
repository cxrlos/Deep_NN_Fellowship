""" --------------------------------------------- JSON_to_spaCy.py --------
    |
    |   Purpose: Helper code that transforms the Google standard to the
    |       format that spaCy uses for training
    |
    |   Developer:
    |       Carlos García - https://github.com/cxrlos
    |
    *------------------------------------------------------------------ """

import json

# Class for each instance of an entity detection, it will be passed this way to
# better fit the spaCy training format. 
class DetectedEnt:
    def __init__(self, start, end, d_type):
        self.start = start # Start pos relative to the concatenated JSON text
        self.end = end # Ending pos relative to the concatenated JSON text
        self.d_type = d_type # Entity type according to labelling

def ConvertJSON(path):
    with open(path) as f:
        data = json.load(f)
    json_text = '' # Will contain the JSON's text in a single string
    counter = 0 # To keep track of word positioning
    cl_store = [] # Array to be returned

    for x in data:
        word_len = len(x['text'])
        json_text +=(x['text'] + ' ')
        this_value = DetectedEnt(counter, counter+word_len, x['tags'][2])
        cl_store.append(this_value)
        counter += word_len + 1

    return cl_store, json_text
