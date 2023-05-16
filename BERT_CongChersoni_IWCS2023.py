import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('stsb-roberta-large')

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_sim_from_bert(seq1, seq2):
  # encode sentences to get their embeddings
  embedding1 = model.encode(seq1, convert_to_tensor=True)
  embedding2 = model.encode(seq2, convert_to_tensor=True)
  # compute similarity scores of two embeddings
  cosine_scores = util.pytorch_cos_sim(embedding1, embedding2)
  return cosine_scores.item()

def get_spearman_from_bert(seq1, seq2):
  # encode sentences to get their embeddings
  embedding1 = model.encode(seq1, convert_to_tensor=True)
  embedding2 = model.encode(seq2, convert_to_tensor=True)
  # compute spearman scores of two embeddings
  spearman_scores = stats.spearmanr(embedding1.to("cpu"), embedding2.to("cpu"))[0]
  return spearman_scores.item()

  for i in df.index:
    seq1 = df['sentence1'][i]
    seq2 = df['sentence2'][i]
    df['cosine_sim_bert'][i] = get_sim_from_bert(seq1, seq2)

  for i in df.index:
    seq1 = df['sentence1'][i]
    seq2 = df['sentence2'][i]
    df['spearman_bert'][i] = get_spearman_from_bert(seq1, seq2)