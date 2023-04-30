# %%
import os

import utils
from utils import CORPUS_FILE, VOCAB_LIST_FILE, W2V_FILE
from utils import Autoregressive_TextDataset, add_padding

from gensim.models import Word2Vec
from torch import nn
import torch.nn.functional as F


w2v = Word2Vec.load(W2V_FILE)

import numpy as np

import torch
from torch.utils import data

if not os.path.isfile(CORPUS_FILE) or not os.path.isfile(VOCAB_LIST_FILE):
    utils.reload_sp_corpus(verbose=True)
corpus, vocab_list = torch.load(CORPUS_FILE), torch.load(VOCAB_LIST_FILE)

print("example sentence:", corpus[:3])
print("vocab:", vocab_list[:10])

# %%
# vocab = Vocab(vocab_list)
# sentence = "the queen said <UNK>"
# sentence_tokenized = sentence.split(" ")

# print("Sequence of idx:", vocab.word_to_num(sentence_tokenized))
# print("Reconstructed sentence:", vocab.num_to_word(vocab.word_to_num(sentence_tokenized)))

# %%
ds = Autoregressive_TextDataset(corpus, w2v)
dl = data.DataLoader(ds, batch_size=64, shuffle=True, drop_last=True, collate_fn=add_padding)

for example in dl:
    break
print(example.shape) # BATCH x LENGTH
# %%


# %%
class LSTM_basic(nn.Module):
    def __init__(self, hidden_dim, num_layers, embedding_matrix):
        super(LSTM_basic, self).__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = embedding_matrix.shape[0]
        self.embedding_dim = embedding_matrix.shape[1]

        self.embed = torch.cat([torch.zeros(1, self.embedding_dim), embedding_matrix], dim=0)
        #self.embed = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.lstm = nn.GRU(self.embedding_dim, self.hidden_dim, num_layers=num_layers)
        self.mlp = nn.Linear(hidden_dim, self.vocab_size)

    def forward(self, source):
        # embedded = torch.zeros(source.size(0), self.embedding_dim)
        # lstm_out, (h, c) = self.lstm(embedded)
        # out = self.mlp(lstm_out)
        # out = nn.Softmax(dim=-1)(out)
        # output_sequence = out.unsqueeze(dim=1)

        # idx=0
        # while idx < source.size(-1):
        #     if torch.randn(1).item() < 1:
        #         next_tokens = source[:, idx]
        #     else:
        #         next_tokens = out.argmin(dim=-1)
        #     embedded = self.embed[next_tokens]

        #     lstm_out, (h, c) = self.lstm(embedded, (h, c))
        #     out = self.mlp(lstm_out)
        #     out = nn.Softmax(dim=-1)(out)
        #     output_sequence = torch.cat([output_sequence, out.unsqueeze(dim=1)], dim=1)
        #     idx += 1

        embedded = self.embed[source]
        lstm_out, _ = self.lstm(embedded)
        logits = self.mlp(lstm_out)

        return logits
    
w2v_embedding = torch.from_numpy(np.array(w2v.wv))


 # %%
def generate_lstm(model, temperature=0.01):
    numerical = torch.tensor([0])
    for n in range(20):
        model_out = model(numerical.unsqueeze(dim=0))[0][-1]-1
        model_out -= torch.sum(F.one_hot(numerical, num_classes=model.vocab_size), dim=0)



        model_out /= (temperature)
        model_out = nn.Softmax(dim=0)(model_out)

        next_word = torch.multinomial(input=model_out, num_samples=1)
        numerical = torch.cat([numerical, next_word])

    return [w2v.wv.index_to_key[n] for n in numerical[1:]]

# %%

#corpus = ["the king is my friend".split(" ") for i in range(100)]
ds = Autoregressive_TextDataset(corpus, w2v)
dl = data.DataLoader(ds, batch_size=32, shuffle=True, drop_last=True, collate_fn=add_padding)
model = LSTM_basic(hidden_dim=256, embedding_matrix=w2v_embedding, num_layers=4)
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01, weight_decay=0.01)

for i in range(1000):
    for n, example in enumerate(dl):
        source = example[:, :-1]
        target = example[:, 1:]

        optimizer.zero_grad()
        preds = nn.Softmax(dim=-1)(model(source))
        mask = (target != 0).type(torch.int32)
        loss = -torch.log(torch.gather(preds, -1, (target-mask).unsqueeze(dim=2))).squeeze(dim=-1)
        loss = torch.sum(loss*mask)/torch.sum(mask)

        loss.backward()
        optimizer.step()

        if (n % 10) == 0:
            print(f"loss: {loss.item():.4f}, sample: {generate_lstm(model, temperature=0.01)}")

# %%
def generate_lstm_greedy(model, temperature=0.01):
    numerical = torch.tensor([0])
    for n in range(20):
        model_out = model(numerical.unsqueeze(dim=0))[0][-1]
        # model_out /= (temperature)
        # model_out = nn.Softmax(dim=0)(model_out)
        next_word = model_out.argmax().unsqueeze(-1)

        #next_word = torch.multinomial(input=model_out, num_samples=1)
        numerical = torch.cat([numerical, next_word])
    
    return [w2v.wv.index_to_key[n-1] for n in numerical[1:]]
generate_lstm_greedy(model)
# %%
