"""Pipeline for making inference on what comes next given a promted piece of 
text, using a trained TransformerXL model.
"""
import json

import tensorflow as tf
from absl import app
from absl import flags

from model import TransformerXLModel
from model_runners import TransformerXLModelInferencer
from commons import tokenization


flags.DEFINE_string(
    'prompt_filename', None, 'Name of the text file containing the prompt from '
        'which new text will be generated.')
flags.DEFINE_string(
    'filename', None, 'Prefix to the name of the files containing the '
        'configuration file (.json) of the training corpus.')
flags.DEFINE_string(
    'vocab_path', None, 'Path to the vocabulary file.')
flags.DEFINE_string(
    'model_dir', None, 'Path to the directory that checkpoint files will be '
        'restored from.')

flags.DEFINE_enum(
    'decoding_method', 'beam_search', ['topk', 'nucleus', 'beam_search'], 
        'Decoding method.')
flags.DEFINE_integer(
    'm_seq_len', 224, 'Memory sequence length.')
flags.DEFINE_list(
    'cutoffs', [20000, 40000, 200000], 'Boundaries of the token IDs in the '
        'vocabulary used to split tokens into head and multiple tails.')
flags.DEFINE_bool(
    'adaptive_embedding', True, 'Whether to use adaptive token embedding and '
        'softmax for large vocabulary.')
flags.DEFINE_integer(
    'num_tokens', 512, 'The number of tokens to be generated.')

flags.DEFINE_integer(
    'stack_size', 9, 'Num of layers in the decoder stack.')
flags.DEFINE_integer(
    'hidden_size', 512, 'The dimensionality of the embedding vector.')
flags.DEFINE_integer(
    'num_heads', 8, 'Num of attention heads.')
flags.DEFINE_integer(
    'filter_size', 2048, 'The depth of the intermediate dense layer of the'
        'feed-forward sublayer.')
flags.DEFINE_bool(
    'tie_biases', True, 'Whether to force all layers use the same content '
        'bias and position bias (True), or create the biases for each layer'
        ' (False).')
flags.DEFINE_bool(
    'batch_memory_processing', False, 'whether to compute the sequence '
        'embeddings in the memory segment batchwise, or one at a time.')


FLAGS = flags.FLAGS


def main(_):
  prompt_filename = FLAGS.prompt_filename
  filename = FLAGS.filename
  vocab_path = FLAGS.vocab_path
  model_dir = FLAGS.model_dir

  decoding_method = FLAGS.decoding_method
  m_seq_len = FLAGS.m_seq_len
  cutoffs = FLAGS.cutoffs
  adaptive_embedding = FLAGS.adaptive_embedding
  num_tokens = FLAGS.num_tokens

  stack_size = FLAGS.stack_size
  hidden_size = FLAGS.hidden_size
  num_heads = FLAGS.num_heads
  filter_size = FLAGS.filter_size
  tie_biases = FLAGS.tie_biases

  with tf.io.gfile.GFile(filename + '.json') as f:
    dataset_config = json.load(f)

  subword = dataset_config['subword']

  if subword:
    tokenizer = tokenization.restore_subtokenizer_from_vocab_files(vocab_path)
  else:
    tokenizer = tokenization.restore_tokenizer_from_vocab_files(vocab_path)

  vocab_size = tokenizer.vocab_size
  cutoffs = list(map(int, cutoffs))

  model = TransformerXLModel(adaptive_embedding,
                             vocab_size,
                             cutoffs,
                             stack_size, 
                             hidden_size, 
                             num_heads, 
                             filter_size,
                             tie_biases=tie_biases)

  ckpt = tf.train.Checkpoint(model=model)
  latest_ckpt = tf.train.latest_checkpoint(model_dir)
  if latest_ckpt is None:
    raise ValueError('No checkpoint is found in %s' % model_dir)
  print('Loaded latest checkpoint', latest_ckpt)
  ckpt.restore(latest_ckpt).expect_partial()

  inferencer = TransformerXLModelInferencer(model, 
                                            m_seq_len, 
                                            1, 
                                            adaptive_embedding, 
                                            decoding_method,
                                            num_tokens=num_tokens)

  with open(prompt_filename) as f:
    prompt = f.read()

  prompt_token_ids = tokenizer.encode(prompt, add_eos=False)
  token_id_list = inferencer.infer(tf.constant([prompt_token_ids]))
  if tokenization.EOS_ID in token_id_list:
    index = token_id_list.index(tokenization.EOS_ID)
    token_id_list = token_id_list[:index]
  text =tokenizer.decode(token_id_list)
  print('\nPrompted Sequence:\n')
  print(prompt, '\n\n')
  print('Predicted sequence:\n')
  print(text)

if __name__ == '__main__':
  flags.mark_flag_as_required('prompt_filename')
  flags.mark_flag_as_required('vocab_path')
  flags.mark_flag_as_required('model_dir')
  app.run(main)
