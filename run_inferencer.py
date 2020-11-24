"""Defines TransformerXL model in tf.keras.API."""
import tensorflow as tf
import numpy as np

import utils

from model import TransformerXLModel
#from model import AdaptiveSoftmaxV2
from model import AdaptiveSoftmaxV1
from model_runners import TransformerXLModelInferencer

if __name__ == '__main__':
  stack_size = 6
  num_heads = 8
  hidden_size = 512
  filter_size = 2048
  training = False
  vocab_size = 267735
  cutoffs = [20000, 40000, 200000]

  mems = tf.zeros([1, stack_size, 50, hidden_size], dtype='float32')

  model = TransformerXLModel(vocab_size, stack_size, hidden_size, num_heads, filter_size, 0.1)
  adaptive_softmax = AdaptiveSoftmaxV1(hidden_size, cutoffs + [vocab_size])

  ckpt = tf.train.Checkpoint(model=model, adaptive_softmax=adaptive_softmax)
  latest_ckpt = tf.train.latest_checkpoint('.')
  ckpt.restore(latest_ckpt).expect_partial()
  print('\n\n\n\n', latest_ckpt)


  import functools 
  scoring_fn = functools.partial(adaptive_softmax, mode='softmax')



  primer = """Robert Boulter is an English film , television and theatre actor . He had a guest @-@ starring role on the television series The Bill in 2000 . This was followed by a starring role in the play Herons written by Simon Stephens , which was performed in 2001 at the Royal Court Theatre . He had a guest role in the television series Judge John Deed in 2002 . In 2004 Boulter landed a role as " Craig " in the episode " Teddy 's Story " of the television series The Long Firm ; he starred alongside actors Mark Strong and Derek Jacobi . He was cast in the 2005 theatre productions of the Philip Ridley play Mercury Fur , which was performed at the Drum Theatre in Plymouth and the <unk> Chocolate Factory in London . He was directed by John Tiffany and starred alongside Ben Whishaw , Shane Zaza , Harry Kent , Fraser Ayres , Sophie Stanton and Dominic Hall .""".split()
 
  rev_vocab = {'<eos>': 0}
  vocab = ['<eos>']
  with open('vocab') as f:
    for i, line in enumerate(f):
      vocab.append(line.strip())
      rev_vocab[line.strip()] = i + 1 
  

  def encode(text, rev_vocab):
    return [rev_vocab[s] for s in text]

  def decode(id_list, vocab):
    return ' '.join([vocab[id_] for id_ in id_list])

  text = encode(primer, rev_vocab)
  inputs = tf.constant([text[:51]], dtype='int32')
  #labels = tf.constant([text[1:51]], dtype='int32')
  #outputs, new_memories = model(inputs, mems, training)


  #training_losses = adaptive_softmax(outputs, labels, 'loss')
  #loss = tf.reduce_mean(tf.concat(training_losses, axis=0))

   
  #outputs, new_mems = model(inputs, mems, training)

  #########init = tf.argmax(adaptive_softmax(outputs, 'softmax')[0, -1]).numpy()
  #########primer_seq_ids = tf.constant([[init]])

  #primer_seq_ids = tf.constant([text[50]], dtype='int32') 

  #out = model.predict(primer_seq_ids, new_mems, scoring_fn)
 
  inferencer = TransformerXLModelInferencer(model, adaptive_softmax, 50)

  out = inferencer.infer(inputs) 
