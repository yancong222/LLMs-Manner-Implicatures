# created on Aug 2024
# authors: Yan Cong

# Load libraries

import pandas as pd
import numpy as np
import re
import math
import os
import string
import shutil, sys, glob


data = ('/dataset/')

"""
Install the Google AI Python SDK
$ pip install google-generativeai
"""

import os
import google.generativeai as genai
from google.colab import userdata
genai.configure(api_key=userdata.get('GoogleAI'))

"""# Set up gemini-1.5-flash"""

# Create the model
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

safety_settings = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_NONE",
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_NONE",
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_NONE",
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_NONE",
  },
]

model = genai.GenerativeModel(
  model_name="gemini-1.5-flash", # gemini-1.5-pro
  generation_config=generation_config,
  safety_settings = safety_settings
)

chat_session = model.start_chat(
  history=[
  ]
)

"""# Prompt Gemini"""

tf = pd.read_csv(data + 'dataset.csv', index_col=0)
tf.head()

for i in range(len(tf)):
    claim = " is equivalent to "
    sent1 = tf['sentence'][i]
    sent2 = tf['neg_antonym'][i]
    input_combined = instruction + task + "sentece 1: " + '"' + sent1 + '"' + claim + "sentece 2: " + '"' + sent2 + '"'
    response = chat_session.send_message(input_combined)
    temp = response.text
    tf['gemini_flash_sentence_neg_antonym_equivalent'][i] = temp.replace(' \n', '').split()[0]
    tf.to_csv(data + 'dataset.csv')
    print('finished line: ', i)

"""# Gemini text-embedding similarity"""

result1 = genai.embed_content(
    model="models/text-embedding-004",
    content="I'm so depressed")

result2 = genai.embed_content(
    model="models/text-embedding-004",
    content="I'm so sad",
    task_type="semantic_similarity")

print(str(result1['embedding'])[:50], '... TRIMMED]')
print(str(result2['embedding'])[:50], '... TRIMMED]')

import numpy as np

def cosine_similarity(v1, v2):
  return round(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)),3)

from scipy.stats import spearmanr

correlation, p_value = spearmanr(result1['embedding'], result2['embedding'])

print("Spearman correlation coefficient:", correlation)
print("P-value:", round(p_value,3))

for i in range(len(df)):
    result1 = genai.embed_content(
    model="models/text-embedding-004",
    content=df['sentence'][i])

    result2 = genai.embed_content(
    model="models/text-embedding-004",
    content=df['neg_antonym'][i],
    task_type="semantic_similarity")

    df['gemini_cos_sim_sent_negAntonym'][i] = cosine_similarity(result1['embedding'], result2['embedding'])
    df['gemini_spearman_sent_negAntonym'][i] = round(spearmanr(result1['embedding'], result2['embedding']).statistic,3)

    df.to_csv(data + "dataset.csv")
