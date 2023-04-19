# %%
import os
import utils
from utils import W2V_FILE, CORPUS_FILE, VOCAB_FILE, W2V_VEC_SIZE
import torch
from gensim.models import Word2Vec


if not os.path.isfile(W2V_FILE):
    if not os.path.isfile(CORPUS_FILE) or not os.path.isfile(VOCAB_FILE):
        corpus, vocab = utils.reload_sp_corpus(verbose=True)
    else:
        corpus, vocab = torch.load(CORPUS_FILE), torch.load(VOCAB_FILE)
    print("Training W2V model on corpus...")
    w2v = Word2Vec(sentences=corpus, vector_size=W2V_VEC_SIZE, window=5, min_count=10, epochs=200)
    print("done!")
    w2v.add_null_word()
    w2v.save(W2V_FILE)
else:
    w2v = Word2Vec.load(W2V_FILE)

# %%
print("king:man::woman:")
for answer in w2v.wv.most_similar(positive=['king','woman'], negative=['man']):
    print(answer)
print('====================')

# %%

print("tall:short::tree:")
for answer in w2v.wv.most_similar(positive=['powerful','woman'], negative=['man']):
    print(answer)
print('====================')
# %%
