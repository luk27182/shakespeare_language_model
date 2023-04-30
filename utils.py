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

RAW_FILE  = "Shakespeare.txt"

CHECKPOINTS_FOLDER = "checkpoints/"
CORPUS_FILE = CHECKPOINTS_FOLDER+"corpus.pth"
VOCAB_LIST_FILE = CHECKPOINTS_FOLDER+"vocab_list.pth"
W2V_FILE = CHECKPOINTS_FOLDER+"w2v.pth"

W2V_VEC_SIZE = 128

def reload_sp_corpus(min_length=5, min_count=5, unk_token="<UNK>", verbose=False):
    """
    Loads in and formats the file Shakespeare.txt

    Inputs:
        verbose: enables printed checkpoints and information about the dataset.
        min_length: The minimum length of a sentence (in tokens) needed to be added to the corpus.
        min_count: The minimum number of times a word must appear to be included
        unk_token: The token to replace tokens which do not appear at least min_token times

    Outputs:
        corpus: A list of strings, where each line is a sentence (until stanza break or punctuation)
        vocab_list: A list of tuples of the form [word, count], describing the counts of each word in the corpus
    """
    if verbose: print("Loading in files...")
    with open(RAW_FILE, 'rb') as f:
        corpus = [str(line) for line in f.readlines()]
    corpus = "".join(corpus)
    corpus = re.sub("\\\\xe2\\\\x80\\\\x99", "'", corpus) # Insert apostrophies where they are required 
    corpus = re.sub("'b'|b'|(\\\\[a-z]+)+|_", " ", corpus).lower() # Remove formatting text
    corpus = re.sub(",", "", corpus) # There were lots of commas, no reason to have them
    corpus = re.split("[0-9]+|\.|\?|\!", corpus) # Split by lines 
    corpus = corpus[10:] # Remove the opening lines
    
    if verbose: print("Done!\nTokenizing sentences...")
    corpus = [word_tokenize(line) for line in corpus]
    corpus = [line for line in corpus if len(line) >= min_length] # tokenize the sentences, and only include sufficiently long ones
    if verbose: print("Done!")

    " ".join(corpus[0]) #First example sentence from corpus

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
    
    torch.save(obj=corpus, f=CORPUS_FILE)
    torch.save(obj=vocab_list, f=VOCAB_LIST_FILE)
    return corpus, vocab_list

# class Vocab():
#     def __init__(self, vocab_list):
#         self.vocab_list = ["<PAD>", "<s>"] + vocab_list

#     def __len__(self):
#         return len(self.vocab_list)
    
#     def word_to_num(self, input):
#         input = ["<s>"] + input + ["</s>"]
#         output = []
#         for word in input:
#             output.append(self.vocab_list.index(word))
    
#         return torch.tensor(output)
    
#     def num_to_word(self, input):
#         output = []
#         for idx in input:
#             output.append(self.vocab_list[idx])
#         return output

# %%
def add_padding(batch):
    original = [pair for pair in batch]
    padded = pad_sequence(original, padding_value=0)
    
    #return padded
    return rearrange(padded, "length batch -> batch length")

class Autoregressive_TextDataset(data.Dataset):
    def __init__(self, corpus, w2v):        
        self.corpus = corpus
        self.max_length = max([len(sentence) for sentence in self.corpus])
        self.w2v = w2v

    def __len__(self):        
        return int(len(self.corpus))

    def __getitem__(self, idx):
        output = [0]+[self.w2v.wv.get_index(word)+1 for word in self.corpus[idx]]+[0]      
        return torch.tensor(output)
    
# %%

