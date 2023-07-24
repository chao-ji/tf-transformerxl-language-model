"""Defines utility functions."""
import tensorflow as tf


def get_positional_encoding(batch_size, seq_len, hidden_size, reverse=False):
  """Creates a tensor that encodes positional information.

  Args:
    batch_size: int scalar tensor, batch size. 
    seq_len: int scalar tensor, sequence length.
    hidden_size: int scalar, the hidden size of continuous representation.
    reverse: bool, whether to reverse the sequence. Defaults to False.

  Returns:
    positional_encoding: float tensor of shape [batch_size, seq_len, hidden_size], the 
      tensor that encodes positional information.
  """
  distances = tf.cast(tf.range(seq_len), 'float32')
  if reverse:
    distances = distances[::-1]
  inverse_frequencies = 1 / (10000 ** (tf.range(0, hidden_size, 2.0) /
      hidden_size))
  positional_encoding = tf.einsum('i,j->ij', distances, inverse_frequencies)
  positional_encoding = tf.concat([tf.sin(positional_encoding),
                                   tf.cos(positional_encoding)], axis=1)
  positional_encoding = tf.tile(positional_encoding[tf.newaxis], [batch_size, 1, 1])
  return positional_encoding


def get_look_ahead_mask(q_seq_len, m_seq_len):
  """Creates a tensor to mask out future tokens that should not be attended to.

  Given query sequence of length `q_seq_len`, and memory sequence of length
  `m_seq_len`, the mask would be a `q_seq_len x (m_seq_len + q_seq_len)` matrix
  that looks like this:

  0, ... | 0, 1, 1, ..., 1  
  0, ... | 0, 0, 1, ..., 1

     ...   ...

  0, ... | 0, 0, 0, ..., 1
  0, ... | 0, 0, 0, ..., 0

  where the submatrix to the left of `|` corresponds to the memory sequence, 
  while the submatrix to the right corresponds to the query sequence.

  Args:
    q_seq_len: int scalar tensor, query sequence length.
    m_seq_len: int scalar tensor, memory sequence length.

  Returns:
    look_ahead_mask:  float tensor of shape [1, 1, q_seq_len, 
        m_seq_len + q_seq_len].
  """
  mask = tf.ones([q_seq_len, q_seq_len])
  mask_u = tf.linalg.band_part(mask, 0, -1)
  mask_dia = tf.linalg.band_part(mask, 0, 0)
  mask_pad = tf.zeros([q_seq_len, m_seq_len])
  look_ahead_mask = tf.concat([mask_pad, mask_u - mask_dia], 1)
  look_ahead_mask = look_ahead_mask[tf.newaxis, tf.newaxis, :, :]
  return look_ahead_mask


def cache_memory(memory, inputs, m_seq_len=None):
  """Cache the memory for the next segment.

  Args:
    memory: float tensor of shape [batch_size, m_seq_len, hidden_size], memory
      for the current segment.
    inputs: float tensor of shape [batch_size, q_seq_len, hidden_size], 
      input sequences.
    m_seq_len: int scalar, num of time steps to be cached.

  Returns:
    new_memory: float tensor of shape [batch_size, m_seq_len, hidden_size],
      memory cached for the next segment.
  """
  if m_seq_len is None:
    m_seq_len = tf.shape(memory)[1]
  new_memory = tf.stop_gradient(
      tf.concat([memory, inputs], axis=1)[:, -m_seq_len:])
  return new_memory
