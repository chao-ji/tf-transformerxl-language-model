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


def rel_shift(inputs):
  """Shift the matrix in the input tensor, so that the query position matches 
  correctly with the key position for computing attention scores.

  Given input tensor `x` of shape [batch_size, num_heads, q_seq_len, r_seq_len],
  each slice `x[i, j]` is a matrix of shape [q_seq_len, r_seq_len] (Note that 
  generally `r_seq_len` >= `q_seq_len`

  the matrix `x[i, j]` in the output will be a left-shifted version of the input
  , where the 0th, 1st, ..., and `q_seq_len - 1`-th row will be left-shifted by 
  `q_seq_len - 1`, `q_seq_len - 2`, ..., and 0 positions.


  Args:
    inputs: float tensor of shape [batch_size, num_heads, q_seq_len, r_seq_len],
      the input tensor.

  Returns:
    outputs: float tensor of shape [batch_size, num_heads, q_seq_len, r_seq_len]
      , the shifted tensor.
  """
  shape = tf.shape(inputs)
  padded = tf.pad(inputs, [[0, 0], [0, 0], [0, 0], [1, 0]])
  reshaped = tf.reshape(padded, [shape[0], shape[1], shape[3] + 1, shape[2]])
  sliced = reshaped[:, :, 1:]
  outputs = tf.reshape(sliced, shape)
  return outputs

   
def cache_memory(memory, inputs, m_seq_len=None):
  """Cache the memory for the next segment.

  Args:
    memory: float tensor of shape [batch_size, m_seq_len, hidden_size], memory
      for the current segment.
    inputs: float tensor of shape [batch_size, q_seq_len, hidden_size], 
      input sequences.
    m_seq_len: int scalar, num of time steps to be cached.

  Returns:
    new_memory: float tensor of shape [batch_size, m_seq_len], memory cached
      for the next segment.
  """
  if m_seq_len is None:
    m_seq_len = tf.shape(memory)[1]
  new_memory = tf.stop_gradient(
      tf.concat([memory, inputs], axis=1)[:, -m_seq_len:])
  return new_memory
