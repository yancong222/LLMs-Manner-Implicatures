# created on Aug 2024
# authors: Yan Cong

import numpy as np
import pandas as pd
import torch
import math
from gensim.test.utils import get_tmpfile
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import gensim.models as gs
import nltk

data = 'dataset'
root = 'root folder'
all_sentence_files = os.listdir(data)
fname = get_tmpfile("glove.6B.300d.w2v.txt")
model = KeyedVectors.load_word2vec_format(fname)
glove = model.wv

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def calc_glove(sent):
  sentence_embeddings = []
  for word in sent.strip('.').split(' '):
    if word.lower() in glove:
      sentence_embeddings.append(glove[word.lower()])  
  return sentence_embeddings

def mean_embedding_of_sentence(sent):
  sentence_embeddings = calc_glove(sent)
  if len(sentence_embeddings) > 0:
    return np.average(sentence_embeddings,0) 
  else:
    return np.NaN

def cos_sim(sent1, sent2):
  emb1 = mean_embedding_of_sentence(sent1)
  emb2 = mean_embedding_of_sentence(sent2)
  cos_sim = cosine_similarity(emb1, emb2)
  return cos_sim

def spearman(sent1, sent2):
  emb1 = mean_embedding_of_sentence(sent1)
  emb2 = mean_embedding_of_sentence(sent2)
  sp = stats.spearmanr(emb1, emb2)[0]
  return sp

df['glove_cos_sim_negAntonym_inf1'] = df.apply(lambda x: cos_sim(x.neg_antonym, x.inference1), axis = 1)
df['glove_cos_sim_negAntonym_inf2'] = df.apply(lambda x: cos_sim(x.neg_antonym, x.inference2), axis = 1)




