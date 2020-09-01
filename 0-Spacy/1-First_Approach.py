""" ------------------------------------------- First_Approach.py ---------
    |
    |   Purpose: First Spacy implementation, testing capibilities and NLP
    |       context-free performance.
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

def LocalhostGraphs():
    displacy.serve(doc2, style='ent')
    options = {"compact": True, "color": "black", "font": "Source Sans Pro"}
    displacy.serve(doc, style="dep", options=options)

# Classification with context
doc = nlp('Juventus Andrea Pirlo\'s role before the call came from Andrea Agnelli to replace the axed Maurizio Sarri. Pirlo has got his feet under the table and with the new season a matter of weeks away, the scrutiny over his system and tactics is intensifying.')
doc2 = nlp('It appears increasingly likely that Pirlo is going to lean towards a 3-4-1-2 system. The move harks back to the Antonio Conte era, the start of this dominant run that has them eyeing a tenth straight Serie A title this season. The three central defenders, at least initially, will be Merih Demiral, Leonardo Bonucci and Giorgio Chiellini, awaiting the return of Matthjis De Ligt by the end of October as he recovers from surgery.')
print([(X.text, X.label_) for X in doc.ents])
LocalhostGraphs()
