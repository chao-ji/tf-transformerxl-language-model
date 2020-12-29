"""Pipeline for evaluating a trained TransformerXL model for language modeling.
"""
import functools
import json
import os

import tensorflow as tf
from absl import app
from absl import flags

from model import TransformerXLModel
from model_runners import TransformerXLModelEvaluator
from commons.layers import AdaptiveInputSoftmax
from commons.layers import EmbeddingLayer
from commons.dataset import parse_fn_sequence_pair
from commons import tokenization


flags.DEFINE_string(
    'filename', None, 'Prefix to the name of the files containing the '
        'validation set (.tfrecord) and configuration file (.json).')
flags.DEFINE_string(
    'vocab_path', None, 'Path to the vocabulary file.')
flags.DEFINE_string(
    'model_dir', None, 'Path to the directory that checkpoint files will be '
        'restored from.')

flags.DEFINE_integer(
    'm_seq_len', 224, 'Memory sequence length.')
flags.DEFINE_list(
    'cutoffs', [20000, 40000, 200000], 'Boundaries of the token IDs in the '
        'vocabulary used to split tokens into head and multiple tails.')
flags.DEFINE_bool(
    'adaptive_embedding', True, 'Whether to use adaptive token embedding and '
        'softmax for large vocabulary.')

flags.DEFINE_integer(
    'stack_size', 9, 'Num of layers in the decoder stack.')
flags.DEFINE_integer(
    'hidden_size', 512, 'The dimensionality of the embedding vector.')
flags.DEFINE_integer(
    'num_heads', 8, 'Num of attention heads.')
flags.DEFINE_integer(
    'filter_size', 2048, 'The depth of the intermediate dense layer of the'
        'feed-forward sublayer.')

FLAGS = flags.FLAGS


def main(_):
  filename = FLAGS.filename
  vocab_path = FLAGS.vocab_path
  model_dir = FLAGS.model_dir

  m_seq_len = FLAGS.m_seq_len
  cutoffs = FLAGS.cutoffs
  adaptive_embedding = FLAGS.adaptive_embedding

  stack_size = FLAGS.stack_size
  hidden_size = FLAGS.hidden_size
  num_heads = FLAGS.num_heads
  filter_size = FLAGS.filter_size

  with tf.io.gfile.GFile(filename + '.json') as f:
    dataset_config = json.load(f)

  subword = dataset_config['subword']
  batch_size = dataset_config['batch_size']
  
  if subword:
    tokenizer = tokenization.restore_subtokenizer_from_vocab_files(vocab_path)
  else:
    tokenizer = tokenization.restore_tokenizer_from_vocab_files(vocab_path)
  vocab_size = tokenizer.vocab_size
  cutoffs = list(map(int, cutoffs)) + [vocab_size]

  if adaptive_embedding:
    embedding_layer = AdaptiveInputSoftmax(hidden_size, cutoffs)
  else:
    embedding_layer = EmbeddingLayer(vocab_size, hidden_size)

  model = TransformerXLModel(embedding_layer,
                             stack_size,
                             hidden_size,
                             num_heads,
                             filter_size)

  parse_fn = functools.partial(parse_fn_sequence_pair, 
                               keys=('inputs', 'labels'), 
                               dtype='int32')
  dataset = tf.data.TFRecordDataset(filename + '.tfrecord')
  dataset = dataset.map(parse_fn).batch(batch_size)

  ckpt = tf.train.Checkpoint(model=model)
  latest_ckpt = tf.train.latest_checkpoint(model_dir)
  if latest_ckpt is None:
    raise ValueError('No checkpoint is found in %s' % model_dir)
  print('Loaded latest checkpoint', latest_ckpt)
  ckpt.restore(latest_ckpt).expect_partial()

  evaluator = TransformerXLModelEvaluator(model, 
                                          m_seq_len, 
                                          batch_size, 
                                          vocab_size, 
                                          adaptive_embedding)
  print('Evaluating file %s...' % (filename + '.tfrecord'))
  perplexity = evaluator.evaluate(dataset)
  print('Perplexity:', perplexity)


if __name__ == '__main__':
  flags.mark_flag_as_required('filename')
  flags.mark_flag_as_required('vocab_path')
  flags.mark_flag_as_required('model_dir')
  app.run(main)
