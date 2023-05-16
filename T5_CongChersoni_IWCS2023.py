import pandas as pd
import numpy as np
import torch
from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration, T5Model
T5_PATH = "t5-large" 
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
t5_tokenizer = T5Tokenizer.from_pretrained(T5_PATH)
t5_config = T5Config.from_pretrained(T5_PATH)
t5_mlm = T5ForConditionalGeneration.from_pretrained(T5_PATH, config=t5_config).to(DEVICE)
  
# encode sentences using tensorflow
import tensorflow_hub as hub
import tensorflow as tf
import tensorflow_text as text  
hub_url = "https://tfhub.dev/google/sentence-t5/st5-base/1" 
encoder = hub.KerasLayer(hub_url)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_sent_emb(s):
  temp = []
  s = s.replace('[', '')
  s = s.replace(']', '')
  temp.append(s)
  result = encoder(temp)[0].numpy()
  return result

def get_cosine_from_t5(seq1, seq2):
  # encode sentences to get their embeddings
  embedding1 = get_sent_emb(seq1)[0] # for T5 [[emb]]
  embedding2 = get_sent_emb(seq2)[0]
  # compute spearman scores of two embeddings
  cosine_sim = cosine_similarity(embedding1, embedding2)
  return cosine_sim

def get_spearman_from_t5(seq1, seq2):
  # encode sentences to get their embeddings
  embedding1 = get_sent_emb(seq1)[0] # for T5 [[emb]]
  embedding2 = get_sent_emb(seq2)[0]
  # compute spearman scores of two embeddings
  spearman_scores = stats.spearmanr(embedding1, embedding2)[0]
  return spearman_scores

for i in df.index:
  seq1 = df['sentence1'][i]
  seq2 = df['sentence2'][i]
  df['cosine_sim_t5'][i] = get_cosine_from_t5(seq1, seq2)

for i in df.index:
    seq1 = df['sentence1'][i]  
    seq2 = df['sentence2'][i] 
    df['spearman_t5'][i] = get_spearman_from_t5(seq1, seq2)
