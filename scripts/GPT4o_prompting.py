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

OPENAI_API_PROJECT_KEY = userdata.get('OPENAI_API_PROJECT_KEY')

# Set the API key as an environment variable
os.environ["OPENAI_API_KEY"] = OPENAI_API_PROJECT_KEY #OPENAI_API_KEY

from openai import OpenAI
client = OpenAI()

completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "Write a haiku about recursion in programming."
        }
    ]
)

"""# Prompting"""

for i in range(len(tf)):
    claim = " is equivalent to "
    sent1 = tf['sentence'][i]
    sent2 = tf['neg_antonym'][i]
    input_combined = instruction + task + "sentece 1: " + '"' + sent1 + '"' + claim + "sentece 2: " + '"' + sent2 + '"'
    completion = client.chat.completions.create(
          model="gpt-4o-mini",
          messages=[
        {"role": "system"},
        {
            "role": "user",
            "content": input_combined
        }])
    temp = completion.choices[0].message.content
    tf['gpt4o_sentence_neg_antonym_equivalent'][i] = temp.replace(' \n', '').split()[0]
    tf.to_csv(data + 'dataset.csv')
