""" --------------------------------------------- JSON_Testing.py ---------
    |
    |   Purpose: Test the default en_core_web_sm model with our JSON 
    |       dataset.
    |
    |   Developer:
    |       Carlos García - https://github.com/cxrlos
    |
    *------------------------------------------------------------------ """

import os
import spacy
import json
import utils
import en_core_web_sm
nlp = en_core_web_sm.load()

arr = os.listdir('../../Data/detected_text')

for i in arr:
    if i.endswith(".json"):
        print("FILE RUNNING: ",i)

        # Data in a local folder
        input_file = '../../Data/detected_text/' + i
        with open(input_file) as f:
            data = json.load(f)

        # Using the utils script, get the data and run it through the pre-trained
        # spacy model
        tokens = utils.convert_google_to_standard(data)
        for x in tokens:
            sentence = nlp(x['text'])
            # For each sentence, add the label value to the JSON token 
            for X in sentence.ents:
                x['entity'] = X.label_
                print(X.text,':', x['entity'])

        output_file = '../../Data/entity_detection/' + i
        with open(output_file, 'w') as outfile:
            json.dump(tokens, outfile, indent=2)
        print('\n')
