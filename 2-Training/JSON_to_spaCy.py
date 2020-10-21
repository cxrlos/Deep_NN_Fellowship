""" --------------------------------------------- JSON_to_spaCy.py --------
    |
    |   Purpose: Helper code that transforms the Google standard to the
    |       format that spaCy uses for training
    |
    |   Developer:
    |       Carlos García - https://github.com/cxrlos
    |
    *------------------------------------------------------------------ """

# TODO: Implement a dir-wise iterative pass (or call from main)
# TODO: De-hardcode class storage
# TODO: Return method

import json
import spacy

class DetectedEnt:
    def __init__(self, start, end, d_type):
        self.start = start 
        self.end = end 
        self.d_type = d_type

with open('../../Data/experiment_results/labelling/1-1-tags.json') as f:
    data = json.load(f)

training_set = []
json_text = ''
counter = 0

# Maybe implement a linked list which will receive the class object
cl_store = []

# Hard-coded storage for the time being
hc_st= []
hc_end= []
hc_type = []

# ("Walmart is a leading e-commerce company", {"entities": [(0, 7, "ORG")]})
for x in data:
    word_len = len(x['text'])
    json_text +=(x['text'])
    this_value = DetectedEnt(counter, counter+word_len, x['tags'][2])
    hc_st.append(counter)
    hc_end.append(counter+word_len)
    hc_type.append(x['tags'][2])
    counter += word_len + 1
print(json_text, '\n')

for i in range(0, len(hc_st)):
    print(hc_st[i], hc_end[i], hc_type[i])

# Return should pass class and text
