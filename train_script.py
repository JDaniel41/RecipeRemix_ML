# Import Libraries
import tensorflow as tf
import ast
import numpy as np
import pandas as pd
import os
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("initial_epoch")
args = parser.parse_args()

# Read in Food.com dataset
df = pd.read_csv('data/RAW_recipes.csv')
recipe_df = df[["steps"]]

recipe_df["processed"] = [". ".join(ast.literal_eval(step_array)) for step_array in recipe_df.steps]
text = ""

for recipe in recipe_df["processed"]:
    text += recipe

# The unique characters in the file
vocab = sorted(set(text))

# Creating a mapping from unique characters to indices
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

# Convert all the text in the Food.com set into their corresponding integers.
text_as_int = np.array([char2idx[c] for c in text])

# The maximum length sentence we want for a single input in characters is 100. In other
# words. Given one single input, we want to predict up to 100 characters in advance.
seq_length = 100

# Create training examples. First we make this into a dataset from the
# integered text. 
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

# Now we return batches of size sequence+1 into this variable. If the last batch
# has less than seq_lenth + 1 elements, we disregard it.
sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

# This function will split the text into an input chunk and the
# target chunk. Remember, our input (the first part) should be
# everything but the last part of the text.
def split_input_target(chunk):
  input_text = chunk[:-1]
  target_text = chunk[1:]
  return input_text, target_text

dataset = sequences.map(split_input_target)

for input_example, target_example in  dataset.take(1):
  print ('Input data: ', repr(''.join(idx2char[input_example.numpy()])))
  print ('Target data:', repr(''.join(idx2char[target_example.numpy()])))

# Batch size
BATCH_SIZE = 64

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 10000

# Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints'

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

# Length of the vocabulary in chars
vocab_size = len(vocab)

# The embedding dimension
embedding_dim = 256

# Number of RNN units
rnn_units = 1024

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

model = build_model(
    vocab_size = len(vocab),
    embedding_dim=embedding_dim,
    rnn_units=rnn_units,
    batch_size=BATCH_SIZE)

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.summary()

def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

model.compile(optimizer='adam', loss=loss)

# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

early_stop_callback = tf.keras.callbacks.EarlyStopping(
    monitor='loss', patience=3
)

EPOCHS=30

print("Starting Training")
history = model.fit(dataset, epochs=EPOCHS, initial_epoch=(int)(args.initial_epoch), callbacks=[checkpoint_callback, early_stop_callback])
