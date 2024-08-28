# created on Aug 2024
# authors: Yan Cong

import pandas as pd
import numpy as np
import torch
from transformers import GPT2TokenizerFast
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
from sklearn.manifold import TSNE
import matplotlib
import matplotlib.pyplot as plt
from openai.embeddings_utils import get_embedding, cosine_similarity

import openai
openai.api_key = 'your openai api key' 

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_sim_spearman_from_gpt3(seq1, seq2):
  results = []
  embedding1 = get_embedding(seq1, engine = 'text-embedding-ada-002') 
  embedding2 = get_embedding(seq2, engine = 'text-embedding-ada-002') 
  cos = cosine_similarity(embedding1, embedding2)
  results.append(cos)
  sp = stats.spearmanr(embedding1, embedding2)[0] 
  results.append(sp)
  return results

for i in snil_gpt3.index:
  s1=snil_gpt3['sentence1'][i]
  s2=snil_gpt3['sentence2'][i]
  sp = get_sim_spearman_from_gpt3(s1, s2)
  snil_gpt3['cosine_sim_gpt3'][i] = temp[0]
  snil_gpt3['spearman_gpt3'][i] = temp[1]
