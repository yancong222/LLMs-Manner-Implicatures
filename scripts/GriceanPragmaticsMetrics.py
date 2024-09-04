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


'''
PRC metrics explanation

To evaluate the PRC metric, we define:

1. Reasoning Step Prompts: These will be custom prompts based on formal reasoning steps that the LLM should follow.
2. Expected Reasoning Steps: The correct reasoning steps according to formal pragmatics.
3. LLM Output Analysis: The comparison between the LLM's generated steps and the expected steps.

- LLM Responses: These are the reasoning steps generated by the LLM when probed with specific prompts.
- Expected Steps: These represent the correct reasoning steps based on formal pragmatics (e.g., deriving "not all" from "some").
- Accuracy: The overall accuracy of the LLM’s responses is calculated as the proportion of correctly generated reasoning steps.
- Step Scores: This array represents the correctness of each individual step (1.0 for correct, 0.0 for incorrect).

### Additional Enhancements:

- Partial Matching: You could implement a more sophisticated comparison, allowing partial credit for reasoning steps that are conceptually correct but not exactly matching the expected step.
- Prompt Engineering: You can design prompts that explicitly probe each reasoning step, providing more granular control over the evaluation process.
- Multilingual Capability: By altering the `expected_steps` and `llm_responses` according to different languages, the function can evaluate PRC in a multilingual context.

This approach allows us to systematically evaluate how well LLMs follow complex pragmatic reasoning processes, aligning with formal pragmatic frameworks.

'''

from typing import List, Dict

def evaluate_prc(llm_responses: List[str], expected_steps: List[str]) -> Dict[str, float]:
    """
    Evaluate the Pragmatic Reasoning Chains (PRC) metric by comparing the LLM's reasoning steps
    to the expected reasoning steps.

    Parameters:
    llm_responses (List[str]): The reasoning steps generated by the LLM.
    expected_steps (List[str]): The correct reasoning steps based on formal pragmatics.

    Returns:
    Dict[str, float]: A dictionary containing the accuracy and step-by-step alignment score.
    """
    correct_steps = 0
    step_scores = []
    
    # Compare each LLM response with the corresponding expected step
    for llm_step, expected_step in zip(llm_responses, expected_steps):
        if llm_step.strip().lower() == expected_step.strip().lower():
            correct_steps += 1
            step_scores.append(1.0)
        else:
            step_scores.append(0.0)
    
    # Calculate overall accuracy
    accuracy = correct_steps / len(expected_steps)
    
    return {
        "accuracy": accuracy,
        "step_scores": step_scores
    }

# Example usage
llm_responses = [
    "The speaker said 'some', which can mean one or more, including all.",
    "You can say 'I ate some of the cookies' even if you ate them all.",
    "But if you had eaten them all, and that was important, you would have said 'all'.",
    "The fact that the speaker didn't say 'all' implies that 'all' doesn't hold.",
    "This leads to the interpretation 'I ate some, but not all'."
]

expected_steps = [
    "The speaker said 'some'. It literally means 'one or more, possibly all.' Thus, 'some' is logically compatible with 'all'.",
    "You can say 'I ate some of the cookies' even when you ate them all.",
    "However, if you in fact ate them all and if that is relevant to the purpose of the current exchange, S would have said 'all'.",
    "The fact that S didn’t choose 'all' then implicates that the stronger 'all' proposition doesn’t hold.",
    "This results in the interpretation 'I ate some, but not all'."
]

result = evaluate_prc(llm_responses, expected_steps)
print(result)
'''
{
    "accuracy": 1.0,
    "step_scores": [1.0, 1.0, 1.0, 1.0, 1.0]
}
'''



'''
IRR metric Explanation:
Explanation:
successful_recoveries: The number of implicature errors that were successfully recovered by the LLM after introducing noise or ambiguity.
total_errors: The total number of implicature errors introduced during the evaluation.
The function calculates the IRR as the ratio of successful_recoveries to total_errors.
The output is a floating-point number representing the IRR, 
which can be interpreted as a percentage (e.g., 0.60 means 60% of implicatures were successfully recovered).
This function allows us to 
evaluate the robustness of LLMs in handling and resolving implicature-related ambiguities 
by measuring how well they can recover from initial errors.
'''
def calculate_irr(successful_recoveries: int, total_errors: int) -> float:
    """
    Calculate the Implicature Recovery Rate (IRR) metric.

    Parameters:
    successful_recoveries (int): The number of successfully recovered implicatures.
    total_errors (int): The total number of implicature errors introduced.

    Returns:
    float: The Implicature Recovery Rate (IRR).
    """
    if total_errors == 0:
        return 0.0
    
    irr = successful_recoveries / total_errors
    return irr

# Example usage
successful_recoveries = 3
total_errors = 5

irr = calculate_irr(successful_recoveries, total_errors)
print(f"Implicature Recovery Rate (IRR): {irr:.2f}")
'''
Implicature Recovery Rate (IRR): 0.60
'''


'''
PSI metric Explanation:
original_accuracy: The accuracy of the LLM's responses when provided with the original context.
changed_accuracy: The accuracy of the LLM's responses after subtle contextual changes, 
such as scrambling nouns or replacing key words with nonsense words.
The function calculates the PSI as the difference between original_accuracy and changed_accuracy.
A higher PSI indicates greater sensitivity to contextual changes, 
meaning the model's performance drops more when the context is altered.
This function allows us to evaluate how well LLMs can adjust to subtle shifts in context, 
reflecting their pragmatic sensitivity.
'''

def calculate_psi(original_accuracy: float, changed_accuracy: float) -> float:
    """
    Calculate the Pragmatic Sensitivity Index (PSI) metric.

    Parameters:
    original_accuracy (float): The accuracy of the LLM's responses in the original context.
    changed_accuracy (float): The accuracy of the LLM's responses after subtle contextual changes.

    Returns:
    float: The Pragmatic Sensitivity Index (PSI).
    """
    psi = original_accuracy - changed_accuracy
    return psi

# Example usage
original_accuracy = 0.85  # 85% accuracy in original context
changed_accuracy = 0.70   # 70% accuracy after subtle contextual changes

psi = calculate_psi(original_accuracy, changed_accuracy)
print(f"Pragmatic Sensitivity Index (PSI): {psi:.2f}")

'''
Pragmatic Sensitivity Index (PSI): 0.15
'''