# %%
import os
import utils
from utils import CORPUS_FILE, VOCAB_FILE
import torch

if not os.path.isfile(CORPUS_FILE) or not os.path.isfile(VOCAB_FILE):
    utils.reload_sp_corpus(verbose=True)
corpus, vocab = torch.load(CORPUS_FILE), torch.load(VOCAB_FILE)

print("example sentence:", corpus[:3])
print("vocab counts:", vocab[:10])