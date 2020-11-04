""" ------------------------------------------- First_Approach.py ---------
    |
    |   Purpose: First Spacy implementation, testing capibilities and NLP
    |       NER performance.
    |
    |   Developer:
    |       Carlos García - https://github.com/cxrlos
    |
    *------------------------------------------------------------------ """

import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
nlp = en_core_web_sm.load()

def LocalhostGraphs(inFile):
    displacy.serve(inFile, style='ent')
    options = {"compact": True, "color": "black", "font": "Source Sans Pro"}
    displacy.serve(inFile, style="dep", options=options)

def main():
# Classification with context
    testSetence = nlp(" Former Italy star Andrea Pirlo took to the pitch for the final time to say goodbye to football in a star-studded testimonial \"Night of the Master\" tournament at the San Siro on Monday. Champions and former teammates from all over the world – from Paolo Maldini to Gianluigi Buffon, and Roberto Baggio to Frank Lampard – joined the retiring legend to celebrate the final act of a glittering career. \"I thank my friends and all the people who came to the stadium to enjoy a fantastic evening of entertainment, sport and charity,\" the 39-year-old said as he received a standing ovation from 50,000 fans.")
    print([(X.text, X.label_) for X in testSetence.ents])
    LocalhostGraphs(testSetence)
    contextFree = nlp("Italy Pirlo 200 Av. Gaetano Scirea 123 Alessandro Del Piero")
    LocalhostGraphs(contextFree)

if __name__ == "__main__":
    main()
