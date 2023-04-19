# %%
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from collections import defaultdict
from nltk.tokenize import word_tokenize
import re
import torch

RAW_FILE  = "Shakespeare.txt"

CHECKPOINTS_FOLDER = "checkpoints/"
CORPUS_FILE = CHECKPOINTS_FOLDER+"corpus.pth"
VOCAB_FILE = CHECKPOINTS_FOLDER+"vocab.pth"
W2V_FILE = CHECKPOINTS_FOLDER+"w2v.pth"

W2V_VEC_SIZE = 64

def reload_sp_corpus(min_length=5, min_count=5, unk_token="UNK", verbose=False):
    """
    Loads in and formats the file Shakespeare.txt

    Inputs:
        verbose: enables printed checkpoints and information about the dataset.
        min_length: The minimum length of a sentence (in tokens) needed to be added to the corpus.
        min_count: The minimum number of times a word must appear to be included
        unk_token: The token to replace tokens which do not appear at least min_token times

    Outputs:
        corpus: A list of strings, where each line is a sentence (until stanza break or punctuation)
        vocab: A list of tuples of the form [word, count], describing the counts of each word in the corpus
    """
    if verbose: print("Loading in files...")
    with open(RAW_FILE, 'rb') as f:
        corpus = [str(line) for line in f.readlines()]
    corpus = "".join(corpus)
    corpus = re.sub("\\\\xe2\\\\x80\\\\x99", "'", corpus) # Insert apostrophies where they are required 
    corpus = re.sub("'b'|b'|(\\\\[a-z]+)+|_", " ", corpus).lower() # Remove formatting text
    corpus = re.split("[0-9]+|\.|\?|\!", corpus) # Split by lines 
    corpus = corpus[10:] # Remove the opening lines

    if verbose: print("Done!\nTokenizing sentences...")
    corpus = [word_tokenize(line) for line in corpus]
    corpus = [line for line in corpus if len(line) >= min_length] # tokenize the sentences, and only include sufficiently long ones
    if verbose: print("Done!")

    " ".join(corpus[0]) #First example sentence from corpus

    vocab = defaultdict(lambda: 0)
    for sentence in corpus:
        for word in sentence:
            vocab[word] += 1

    for i, sentence in enumerate(corpus):
        for j, word in enumerate(sentence):
            if int(vocab[word]) <= min_count:
                corpus[i][j] = unk_token


    vocab = defaultdict(lambda: 0)
    for sentence in corpus:
        for word in sentence:
            vocab[word] += 1
    vocab = [(key, vocab[key]) for key in vocab.keys()]
    vocab = sorted(vocab, key=lambda x:x[1], reverse=True)
    
    if verbose:
        print(f'Dataset info:')
        print('====================')
        print(f"Number of sentences: {len(corpus)}")
        print(f"Distinct tokens with >{min_count} in vocab: {len(vocab)}")
        print(f"Average sentence length (tokens): {sum([len(sentence) for sentence in corpus])/len(corpus):.2f}")
        print(f"Minimum sentence length (tokens): {min([len(sentence) for sentence in corpus]):.0f}")
        print(f"Maximum sentence length (tokens): {max([len(sentence) for sentence in corpus]):.0f}")
    
    torch.save(obj=corpus, f=CORPUS_FILE)
    torch.save(obj=vocab, f=VOCAB_FILE)
    return corpus, vocab