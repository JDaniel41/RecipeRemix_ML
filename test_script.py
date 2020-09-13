import tensorflow as tf
import ast
import argparse

import numpy as np
import pandas as pd

import os
import time
import json

# Disable Tensorflow Warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'

# Parse Command-Line Arguments
parser = argparse.ArgumentParser()
parser.add_argument("query")
args = parser.parse_args()

idx2char = []

# Creating a mapping from unique characters to indices
with open("char2idx.json", "r") as json_file:
    char2idx = json.load(json_file)
    json_file.close()

arr_file = open("idx2char.txt", "r")
while 1: 
      
    # read by character 
    char = arr_file.read(1)           
    if not char: 
        break
    idx2char.append(char)
    
arr_file.close() 

model = tf.keras.models.load_model('saved_model/recipe_generator')

def generate_text(model, start_string):
  # Evaluation step (generating text using the learned model)

  # Number of characters to generate
  num_generate = 1000

  # Converting our start string to numbers (vectorizing)
  input_eval = [char2idx[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)

  # Empty string to store our results
  text_generated = []

  # Low temperatures results in more predictable text.
  # Higher temperatures results in more surprising text.
  # Experiment to find the best setting.
  temperature = 1.0

  # Here batch size == 1
  model.reset_states()
  for i in range(num_generate):
    predictions = model(input_eval)
    # remove the batch dimension
    predictions = tf.squeeze(predictions, 0)

    # using a categorical distribution to predict the character returned by the model
    predictions = predictions / temperature
    predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

    # We pass the predicted character as the next input to the model
    # along with the previous hidden state
    input_eval = tf.expand_dims([predicted_id], 0)

    text_generated.append(idx2char[predicted_id])

  return (start_string + ''.join(text_generated))

output_text = generate_text(model, start_string=args.query.lower())
                            
print(output_text)

with open("output.txt", "w") as output_file:
    output_file.write(output_text)
                          