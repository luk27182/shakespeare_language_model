# %%
import os
from gensim.models import Word2Vec

W2V_FILE = "w2v.pth"
CORPUS_FILE = "corpus.csv"


# %%
with open("corpus.csv", "r", newline="") as f:
    corpus = f.readlines()
corpus = [line.split(',') for line in corpus]

# %%
if not os.path.isfile(W2V_FILE):
    print("Training W2V model on corpus...")
    w2v = Word2Vec(sentences=corpus, vector_size=64, window=5, epochs=20)
    print("done!")
    w2v.add_null_word()
    w2v.save(W2V_FILE)
else:
    w2v = Word2Vec.load(W2V_FILE)

# %%
print("Words most similar to 'knight'")
for answer in w2v.wv.most_similar("knight"):
    print(answer)
print('====================')

# %%
print("Words most similar to 'ten'")
for answer in w2v.wv.most_similar("ten"):
    print(answer)
print('====================')

# %%
print("king:man::woman:")
for answer in w2v.wv.most_similar(positive=['king','woman'], negative=['man']):
    print(answer)
print('====================')

# %%

print("man:powerful::woman:")
for answer in w2v.wv.most_similar(positive=['powerful','woman'], negative=['man']):
    print(answer)
print('====================')
# %%
