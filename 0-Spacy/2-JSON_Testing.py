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
import json
import en_core_web_sm
nlp = en_core_web_sm.load()


def main():
    sentence = nlp("Former Italy star Andrea Pirlo took to the pitch for the final time to say goodbye to football in a star-studded testimonial \"Night of the Master\" tournament at the San Siro on Monday. Champions and former teammates from all over the world – from Paolo Maldini to Gianluigi Buffon, and Roberto Baggio to Frank Lampard – joined the retiring legend to celebrate the final act of a glittering career. \"I thank my friends and all the people who came to the stadium to enjoy a fantastic evening of entertainment, sport and charity,\" the 39-year-old said as he received a standing ovation from 50,000 fans.")
    print([(X.text, X.label_) for X in sentence.ents])
    
    # Data in a local folder
    with open('../Data/tags_our_format/2-1-tags.json') as f:
        data = json.load(f)
    json_str = json.dumps(data)
    print(type (json_str))
    print(json_str)

if __name__ == "__main__":
    main()
