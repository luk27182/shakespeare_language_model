# %%
import os
import utils
import torch

CORPUS_FILE = "sp_corpus.pth"
VOCAB_FILE = "sp_vocab.pth"

if not os.path.isfile(CORPUS_FILE) or not os.path.isfile(VOCAB_FILE):
    utils.reload_sp_corpus(verbose=True)
corpus, vocab = torch.load(CORPUS_FILE), torch.load(VOCAB_FILE)

print(corpus[:3])
print(vocab[:10])