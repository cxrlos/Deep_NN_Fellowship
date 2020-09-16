""" --------------------------------------------- JSON_Testing.py ---------
    |
    |   Purpose: Test the default en_core_web_sm model with our JSON 
    |       dataset.
    |
    |   Developer:
    |       Carlos García - https://github.com/cxrlos
    |
    *------------------------------------------------------------------ """

# import spacy
import json
import utils
# import en_core_web_sm
# nlp = en_core_web_sm.load()


def main():
    # sentence = nlp("Former Italy star Andrea Pirlo took to the pitch for the final time to say goodbye to football in a star-studded testimonial \"Night of the Master\" tournament at the San Siro on Monday. Champions and former teammates from all over the world – from Paolo Maldini to Gianluigi Buffon, and Roberto Baggio to Frank Lampard – joined the retiring legend to celebrate the final act of a glittering career. \"I thank my friends and all the people who came to the stadium to enjoy a fantastic evening of entertainment, sport and charity,\" the 39-year-old said as he received a standing ovation from 50,000 fans.")
    # print([(X.text, X.label_) for X in sentence.ents])
    
    # iterar por cada fichero

    # Data in a local folder
    with open('./detected_text/2-1.json') as f:
        data = json.load(f)

    # json_str = json.dumps(data)
    # print(type (json_str))
    # print(json_str)
    # for x in data:
    #     print(x['text'])
    #     print(x['boundingBox'])
    #     print(x['tags'])

    tokens = utils.convert_google_to_standard(data)
    for x in tokens:
        print(x['text'])
        print(x['boundingBox'])
        print(x['words'])
        # ent = spacy.entitty(token['text'])
        tags = [['label'], ['general'], [str(ent)]]
        x['tags'] = tags

    outfile = f('./detected_text/2-1.json')
    with open(url, 'w') as outfile:
        json.dump(tokens, outfile, indent=4)



if __name__ == "__main__":
    main()
