"""Defines TransformerXL model in tf.keras.API."""
import tensorflow as tf
import numpy as np

import utils

from model import TransformerXLModel
#from model import AdaptiveSoftmaxV2
#from model import AdaptiveSoftmaxV1
from model_runners import TransformerXLModelInferencer

if __name__ == '__main__':
  stack_size = 9#6
  num_heads = 8
  hidden_size = 512
  filter_size = 2048
  training = False
  vocab_size = 267735
  m_seq_len = 224 * 4#4096
  q_seq_len = 224#4096
  cutoffs = [20000, 40000, 200000]

  np.random.seed(0)

  model = TransformerXLModel(vocab_size, cutoffs + [vocab_size], stack_size, hidden_size, num_heads, filter_size, 0.1)

  ckpt = tf.train.Checkpoint(model=model)
  latest_ckpt = tf.train.latest_checkpoint('.')
  ckpt.restore(latest_ckpt).expect_partial()
  print('\n\n\n\n', latest_ckpt)



  primer = ''
  fn = '/home/chaoji/Desktop/transformer-xl/tf/data/wikitext-103/test.txt'

  with open(fn) as f:
    for l in f:
      primer += l[:-1]
      if len(primer.split()) >= 128 * 60:
        break

 
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

  primer = primer.split()
  text = encode(primer, rev_vocab)

  primer_token_ids = tf.constant([text[:m_seq_len + 1]])

   
  inferencer = TransformerXLModelInferencer(model, m_seq_len, 1, 'nucleus')

  l = inferencer.infer(primer_token_ids)

  a = decode(text[:m_seq_len], vocab)
  c = decode(text[m_seq_len:], vocab)
  b = decode(l, vocab)
 


 
