"""Pipeline for training a TransformerXL model for language modeling."""
import functools
import json
import os

import tensorflow as tf
from absl import app
from absl import flags

from model import TransformerXLModel
from model_runners import TransformerXLModelTrainer
from commons.utils import CosineDecayLearningRateSchedule
from commons.layers import AdaptiveInputSoftmax
from commons.layers import EmbeddingLayer
from commons.dataset import parse_fn_sequence_pair
from commons import tokenization


flags.DEFINE_string(
    'filename', None, 'Prefix to the name of the files containing the training '
        'set (.tfrecord) and configuration file (.json).')
flags.DEFINE_string(
    'vocab_path', None, 'Path to the vocabulary file.')
flags.DEFINE_string(
    'model_dir', None, 'Path to the directory that checkpoint files will be'
        'written to.')

flags.DEFINE_integer(
    'm_seq_len', 224, 'Memory sequence length.')
flags.DEFINE_list(
    'cutoffs', [20000, 40000, 200000], 'Boundaries of the token IDs in the '
        'vocabulary used to split tokens in to head and multiple tails.') 
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
flags.DEFINE_float(
    'dropout_rate', 0.1, 'Dropout rate for the Dropout layers.')
flags.DEFINE_float(
    'dropout_rate_attention', 0.0, 'Dropout rate applied on the ' 
        'query-to-reference attention matrix.')

flags.DEFINE_float(
    'learning_rate', 2.5e-4, 'Base learning rate.')
flags.DEFINE_integer(
    'learning_rate_warmup_steps', 0, 'Number of warm-up steps.')
flags.DEFINE_float(
    'optimizer_adam_beta1', 0.9, '`beta1` of Adam optimizer.')
flags.DEFINE_float(
    'optimizer_adam_beta2', 0.999, '`beta2` of Adam optimizer.')
flags.DEFINE_float(
    'optimizer_adam_epsilon', 1e-8, '`epsilon` of Adam optimizer.')

flags.DEFINE_float(
    'warmup_lr', 0., 'Learning rate for warm-up steps.')
flags.DEFINE_float(
    'clip_norm', 0.25, 'The value that the norm of gradient will be '
        'clipped to.')
flags.DEFINE_float(
    'alpha', 0.004, 'Minimum learning rate value as a fraction of '
        'learning rate.')
flags.DEFINE_integer(
    'num_steps', 400000, 'Num of training iterations (minibatches).')
flags.DEFINE_integer(
    'save_ckpt_per_steps', 10000, 'Every this num of steps to save checkpoint.')

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
  dropout_rate = FLAGS.dropout_rate
  dropout_rate_attention = FLAGS.dropout_rate_attention

  learning_rate = FLAGS.learning_rate
  learning_rate_warmup_steps = FLAGS.learning_rate_warmup_steps
  optimizer_adam_beta1 = FLAGS.optimizer_adam_beta1
  optimizer_adam_beta2 = FLAGS.optimizer_adam_beta2
  optimizer_adam_epsilon = FLAGS.optimizer_adam_epsilon

  warmup_lr = FLAGS.warmup_lr
  clip_norm = FLAGS.clip_norm
  alpha = FLAGS.alpha
  num_steps = FLAGS.num_steps 
  save_ckpt_per_steps = FLAGS.save_ckpt_per_steps

  with tf.io.gfile.GFile(filename + '.json') as f:
    dataset_config = json.load(f) 

  subword = dataset_config['subword']
  batch_size = dataset_config['batch_size']
 
  # transformerxl model
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
                             filter_size,
                             dropout_rate=dropout_rate,
                             dropout_rate_attention=dropout_rate_attention)

  # training datset
  parse_fn = functools.partial(parse_fn_sequence_pair, 
                               keys=('inputs', 'labels'),
                               dtype='int32')
  dataset = tf.data.TFRecordDataset(filename + '.tfrecord')
  dataset = dataset.map(parse_fn).repeat().batch(batch_size)

  # learning rate and optimizer
  schedule = CosineDecayLearningRateSchedule(
      learning_rate=learning_rate,
      decay_steps=num_steps-learning_rate_warmup_steps,
      alpha=alpha,
      warmup_steps=learning_rate_warmup_steps,
      warmup_lr=warmup_lr)
  optimizer = tf.keras.optimizers.Adam(
      schedule,
      optimizer_adam_beta1,
      optimizer_adam_beta2,
      epsilon=optimizer_adam_epsilon)

  # checkpoint
  ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)

  # build trainer and start training
  trainer = TransformerXLModelTrainer(model, 
                                      m_seq_len, 
                                      batch_size, 
                                      vocab_size, 
                                      adaptive_embedding)
  trainer.train(dataset, 
                optimizer, 
                ckpt, 
                model_dir, 
                num_steps, 
                save_ckpt_per_steps, 
                clip_norm)

if __name__ == '__main__':
  flags.mark_flag_as_required('filename')
  flags.mark_flag_as_required('vocab_path')
  flags.mark_flag_as_required('model_dir')
  app.run(main)
