import tensorflow as tf
import ast

import numpy as np
import pandas as pd

import os
import time
import json

"""
df = pd.read_csv('data/RAW_recipes.csv')
recipe_df = df[["steps"]]

recipe_df["processed"] = [". ".join(ast.literal_eval(step_array)) for step_array in recipe_df.steps]
text = ""

for recipe in recipe_df["processed"]:
    text += recipe

# The unique characters in the file
vocab = sorted(set(text))
print ('{} unique characters'.format(len(vocab)))

# Creating a mapping from unique characters to indices
char2idx = {u:i for i, u in enumerate(vocab)}
"""

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

print(idx2char)

"""
checkpoint_dir = './training_checkpoints'

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, None]),
    tf.keras.layers.LSTM(rnn_units,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dense(vocab_size)
  ])
  return model


tf.train.latest_checkpoint(checkpoint_dir)

model = build_model(67, 256, 1024, batch_size=1)

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

model.build(tf.TensorShape([1, None]))

model.summary()

model.save("saved_model/recipe_generator")

"""

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

print(generate_text(model, start_string=u"add"))