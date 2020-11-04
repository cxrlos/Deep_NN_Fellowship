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
import matplotlib.pyplot as plt
import sys, time
from spacy.util import minibatch, compounding

nlp = spacy.load('en_core_web_sm')
ner = nlp.get_pipe("ner") 

max_iterations = 20

# To better visualize the learning progress, copied from 
# https://www.pythonexample.org/gui/how-to-draw-a-progress-bar-in-python/
class ProgressBar:
    def __init__(self, count = 0, total = 0, width = 50):
        self.count = count
        self.total = total
        self.width = width
    def move(self):
        self.count += 1
    def log(self, s):
        sys.stdout.write(' ' * (self.width + 9) + '\r')
        sys.stdout.flush()
        progress = int(self.width * self.count / self.total)
        sys.stdout.write('{0:3}/{1:3}: '.format(self.count, self.total))
        sys.stdout.write('#' * progress + '-' * (self.width - progress) + '\r')

# Read the file names in the labelling directory to pass them through the 
# conversion algorithm
files = []
for (path, dirnames, filenames) in os.walk('../../Labelling/Data/experiment_results/labelling'):
    files.extend(os.path.join(path, name) for name in filenames)
random.shuffle(files)

# Go through each JSON file, process it with the JSON_to_spaCy() function and 
# store it into a spaCy compatible array
training_set = []
for x in files:
    current_file, json_text = JSON_to_spaCy.ConvertJSON(x)
    for y in current_file:
        training_set.extend([(json_text, {"entities": [(y.start, y.end, y.d_type)]})])
    # print(training_set, '\n')


# TODO: Make previous assignation work as the following structure
# training_set = [
#     ("Walmart is a leading e-commerce company", {"entities": [(0, 7, "ORG")]}),
#     ("I reached Chennai yesterday.", {"entities": [(19, 28, "GPE")]}),
#     ("I recently ordered a book from Amazon", {"entities": [(24,32, "ORG")]}),
# ]

for _, annotations in training_set:
  for ent in annotations.get("entities"):
    ner.add_label(ent[2])

pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
unaffected_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]

bar = ProgressBar(total = max_iterations)
graph_data = []

print('Training in process, please wait for a moment')
with nlp.disable_pipes(*unaffected_pipes):
  for iteration in range(max_iterations):
    bar.log('' + str(iteration))
    random.shuffle(training_set)
    losses = {}
    batches = minibatch(training_set, size=compounding(4.0, 32.0, 1.001))
    for batch in batches:
        texts, annotations = zip(*batch)
        nlp.update(texts, annotations, drop=0.5, losses=losses)
        # print("Losses", losses)
        graph_data.append(losses['ner'])
    bar.move()
    bar.log('' + str(iteration+1))

out_path = os.getcwd() + '/model'
nlp.to_disk(out_path)
print("Saved model to", out_path)

# Plot loss graph to detemine easily thee correcctness of the learning params
plt.plot(graph_data,':')
plt.xlabel('Number of iterations')
plt.ylabel('Loss')
plt.title('Loss vs Iteratitons')
plt.show()

