# %%
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from einops import rearrange

from collections import defaultdict
from nltk.tokenize import word_tokenize
import re

import torch
from torch.utils import data

from torch.nn.utils.rnn import pad_sequence

import csv

# %%
verbose = True
min_length = 10
min_count = 0
RAW_FILE = "Shakespeare.txt"
unk_token = "<UNK>"

if verbose: print("Loading in files...")
with open(RAW_FILE, 'rb') as f:
    corpus = [str(line) for line in f.readlines()]
corpus = "".join(corpus)
corpus = re.sub("\\\\xe2\\\\x80\\\\x99", "'", corpus) # Insert apostrophies where they are required 
corpus = re.sub("\\\\n", " NEWLINE ", corpus).lower() # Marks newlines
corpus = re.sub("'b'|b'|(\\\\[a-z]+)+|(\\[a-z]+)+|_", " ", corpus)# Remove formatting text
corpus = re.sub(",", "", corpus) # There were lots of commas, no reason to have them
corpus = re.split("[0-9]+|\.|\?|\!", corpus) # Split by lines 
corpus = corpus[10:] # Remove the opening lines
if verbose: print("Done!\nTokenizing sentences...")
corpus = [word_tokenize(line) for line in corpus]
corpus = [line for line in corpus if len(line) >= min_length] # tokenize the sentences, and only include sufficiently long ones
if verbose: print("Done!")
print("Example sentence:")
print(re.sub("newline", "\\n", " ".join(corpus[0])))

# %%
vocab_list = defaultdict(lambda: 0)
for sentence in corpus:
    for word in sentence:
        vocab_list[word] += 1

for i, sentence in enumerate(corpus):
    for j, word in enumerate(sentence):
            if int(vocab_list[word]) <= min_count:
                corpus[i][j] = unk_token
                del vocab_list[word]


vocab_list = list(vocab_list.keys())

if verbose:
    print(f'Dataset info:')
    print('====================')
    print(f"Number of sentences: {len(corpus)}")
    print(f"Distinct tokens with >{min_count} in vocab_list: {len(vocab_list)}")
    print(f"Average sentence length (tokens): {sum([len(sentence) for sentence in corpus])/len(corpus):.2f}")
    print(f"Minimum sentence length (tokens): {min([len(sentence) for sentence in corpus]):.0f}")
    print(f"Maximum sentence length (tokens): {max([len(sentence) for sentence in corpus]):.0f}")


# %%
with open("corpus.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(corpus)
# %%
with open("corpus.csv", "r", newline="") as f:
    corpus = f.readlines()
corpus = [line.split(',') for line in corpus]
# %%
