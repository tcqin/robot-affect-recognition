# -*- coding: utf-8 -*-

""" Use DeepMoji to score texts for emoji distribution.

The resulting emoji ids (0-63) correspond to the mapping
in emoji_overview.png file at the root of the DeepMoji repo.

Writes the result to a csv file.
"""
from __future__ import print_function, division
import example_helper
import json
import csv
import numpy as np
from deepmoji.sentence_tokenizer import SentenceTokenizer
from deepmoji.model_def import deepmoji_emojis
from deepmoji.global_variables import PRETRAINED_PATH, VOCAB_PATH

OUTPUT_PATH = 'test_sentences.csv'
INPUT_SENTENCE_PATH = 'animations_and_transcripts'
INPUT_FILE = 'storyteller_story_anim.json'

# These are test sentences
TEST_SENTENCES = [u'I love mom\'s cooking',
                  u'I love how you never reply back..',
                  u'I love cruising with my homies',
                  u'I love messing with yo mind!!',
                  u'I love you and now you\'re just gone..',
                  u'This is shit',
                  u'This is the shit']
NUM_TEST_SENTENCES = len(TEST_SENTENCES)

with open('{}/{}'.format(INPUT_SENTENCE_PATH, INPUT_FILE)) as f:
    examples = json.load(f)

# Load sentences from the input file
for message in examples.keys():
    script = examples[message]['script']
    TEST_SENTENCES.append(script)

def top_elements(array, k):
    ind = np.argpartition(array, -k)[-k:]
    return ind[np.argsort(array[ind])][::-1]


maxlen = 30
batch_size = 32

print('Tokenizing using dictionary from {}'.format(VOCAB_PATH))
with open(VOCAB_PATH, 'r') as f:
    vocabulary = json.load(f)
st = SentenceTokenizer(vocabulary, maxlen)
tokenized, _, _ = st.tokenize_sentences(TEST_SENTENCES)

print('Loading model from {}.'.format(PRETRAINED_PATH))
model = deepmoji_emojis(maxlen, PRETRAINED_PATH)
model.summary()

print('Running predictions.')
prob = model.predict(tokenized)

# Find top emojis for each sentence. Emoji ids (0-63)
# correspond to the mapping in emoji_overview.png
# at the root of the DeepMoji repo.
print('Writing results to {}'.format(OUTPUT_PATH))
scores = []
for i, t in enumerate(TEST_SENTENCES):
    t_tokens = tokenized[i]
    t_score = [t]
    t_prob = prob[i]
    ind_top = top_elements(t_prob, 5)
    t_score.append(sum(t_prob[ind_top]))
    t_score.extend(ind_top)
    t_score.extend([t_prob[ind] for ind in ind_top])
    scores.append(t_score)
    print(t_score)

# Sort the input file sentences by decreasing Top5Pcts
new_input_scores = sorted(scores[NUM_TEST_SENTENCES:],
                          key=lambda x: x[1],
                          reverse=True)

# Separate the original test sentences
original_scores = scores[:NUM_TEST_SENTENCES]

with open(OUTPUT_PATH, 'wb') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', lineterminator='\n')
    writer.writerow(['Text', 'Top5%',
                     'Emoji_1', 'Emoji_2', 'Emoji_3', 'Emoji_4', 'Emoji_5',
                     'Pct_1', 'Pct_2', 'Pct_3', 'Pct_4', 'Pct_5'])
    for i, row in enumerate(original_scores):
        try:
            writer.writerow(row)
        except Exception:
            print("Exception at row {}!".format(i))
    for i, row in enumerate(new_input_scores):
        try:
            writer.writerow(row)
        except Exception:
            print("Exception at row {}!".format(i))
