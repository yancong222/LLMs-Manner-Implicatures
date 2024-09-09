'''
LLMs-surprisals will be calculated using the Minicons utility
https://github.com/kanishkamisra/minicons 
'''

# !pip install minicons
import pandas as pd
import numpy as np
import os
import math
import csv
import shutil, sys
from minicons import scorer
def response_mean_surp(response, model):
  surp_scores = []
  input = [response] # response.replace('?', '.').replace('!', '.').split('.')
  #print('input: ', input)
  for sent in input:
    # to remove the last empty string element
    if len(sent) > 0:
      #print('sent length: ', len(sent.split()))
      # Sequence Surprisal, normalized by number of tokens - lambda x: -x.mean(0).item()
      score = model.sequence_score([sent], reduction = lambda x: -x.mean(0).item())
      surp_scores.append(score[0])
      #print('mean', surp_scores)
  return round(np.mean(surp_scores),3)


'''
Naturalness Metric Explanation:
utterance_surprisal: The surprisal score for the given utterance.
interpretation_a_surprisal: The surprisal score for interpretation (a) ("He is very smart").
interpretation_b_surprisal: The surprisal score for interpretation (b) ("He is not smart at all").
The function calculates the absolute difference in surprisal scores between the utterance and each interpretation.
The output dictionary contains:
difference_a: The difference between the utterance and interpretation (a).
difference_b: The difference between the utterance and interpretation (b).
preferred_interpretation: The interpretation with the smaller surprisal difference, 
indicating the LLM’s preferred implied meaning.
This function allows us to evaluate whether the LLMs show pragmatic sensitivity by comparing surprisal scores.
'''

def calculate_naturalness(utterance_surprisal: float, interpretation_a_surprisal: float, interpretation_b_surprisal: float) -> dict:
    """
    Calculate the naturalness metric based on surprisal scores.

    Parameters:
    utterance_surprisal (float): Surprisal score for the utterance.
    interpretation_a_surprisal (float): Surprisal score for interpretation (a).
    interpretation_b_surprisal (float): Surprisal score for interpretation (b).

    Returns:
    dict: A dictionary with surprisal differences.
    """
    difference_a = abs(utterance_surprisal - interpretation_a_surprisal)
    difference_b = abs(utterance_surprisal - interpretation_b_surprisal)
    
    return {
        "difference_a": difference_a,
        "difference_b": difference_b,
        "preferred_interpretation": "b" if difference_b < difference_a else "a"
    }

# Example usage
utterance_surprisal = 2.5
interpretation_a_surprisal = 3.8
interpretation_b_surprisal = 2.9

result = calculate_naturalness(utterance_surprisal, interpretation_a_surprisal, interpretation_b_surprisal)
print(result)
'''
{
    "difference_a": 1.3,
    "difference_b": 0.4,
    "preferred_interpretation": "b"
}
'''



'''
SSM metric Explanation:

- embedding_a: The embedding for the first sentence (e.g., "Alex was not unaware of the issue").
- embedding_b: The embedding for the second sentence (e.g., "Alex was slightly aware of the issue").
- embedding_c: The embedding for the third sentence (e.g., "Alex was aware of the issue").
- The function computes cosine similarities between sentence pairs:
  - similarity_ab: Similarity between sentence A and B.
  - similarity_bc: Similarity between sentence B and C.
- The function then ranks these similarities in descending order.
- The output dictionary contains:
  - similarity_ab: Cosine similarity between sentence A and B.
  - similarity_bc: Cosine similarity between sentence B and C.
  - ranking: The ranked list of similarities, indicating which sentence pair is more similar.

This function allows us to evaluate how well LLMs can differentiate between different shades of meaning in pragmatics.
'''
from sklearn.metrics.pairwise import cosine_similarity

def calculate_ssm(embedding_a: np.ndarray, embedding_b: np.ndarray, embedding_c: np.ndarray) -> dict:
    """
    Calculate the Sensitivity to different Shades of Meaning (SSM) metric using cosine similarity.

    Parameters:
    embedding_a (np.ndarray): Embedding for the first sentence (e.g., "Alex was not unaware of the issue").
    embedding_b (np.ndarray): Embedding for the second sentence (e.g., "Alex was slightly aware of the issue").
    embedding_c (np.ndarray): Embedding for the third sentence (e.g., "Alex was aware of the issue").

    Returns:
    dict: A dictionary with similarity scores and their ranking.
    """
    # Calculate cosine similarities
    similarity_ab = cosine_similarity([embedding_a], [embedding_b])[0][0]
    similarity_bc = cosine_similarity([embedding_b], [embedding_c])[0][0]
    
    # Calculate the ranking
    ranking = sorted(
        [("similarity_ab", similarity_ab), ("similarity_bc", similarity_bc)], 
        key=lambda x: x[1], 
        reverse=True
    )
    
    return {
        "similarity_ab": similarity_ab,
        "similarity_bc": similarity_bc,
        "ranking": ranking
    }

# Example usage
embedding_a = np.array([0.1, 0.3, 0.5])
embedding_b = np.array([0.2, 0.4, 0.6])
embedding_c = np.array([0.3, 0.5, 0.7])

result = calculate_ssm(embedding_a, embedding_b, embedding_c)
print(result)
'''
{
    "similarity_ab": 0.9970544855015816,
    "similarity_bc": 0.9996428223613027,
    "ranking": [
        ("similarity_bc", 0.9996428223613027),
        ("similarity_ab", 0.9970544855015816)
    ]
}
'''
