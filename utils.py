"""
"""
import tensorflow as tf
import numpy as np


def get_positional_encoding(seq_len, hidden_size):
  """Creates a tensor that encodes (relative) positional information.

  Args:
    seq_len: int scalar tensor, sequence length.
    hidden_size: int scalar, the hidden size of continuous representation.

  Returns:
    positional_encoding: float tensor of shape [seq_len, hidden_size], the 
      tensor that encodes relative positional information.
  """
  distances = tf.range(seq_len - 1, -1, -1.0)
  inverse_frequencies = 1 / (
      10000 ** (tf.range(0, hidden_size, 2.0) / hidden_size))
  positional_encoding = tf.einsum('i,j->ij', distances, inverse_frequencies)
  positional_encoding = tf.concat([tf.sin(positional_encoding),
                                    tf.cos(positional_encoding)], axis=1)
  #positional_encoding = positional_encoding[tf.newaxis, :, :]
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
  """Shift the input tensor.

  Given input tensor `x` of shape [batch_size, num_heads, q_seq_len, r_seq_len],
  each slice `x[i, j]` is a matrix of shape [q_seq_len, r_seq_len], e.g.

  0,  1,  2
  3,  4,  5
  6,  7,  8
  9, 10, 11

  the shifted version of `x[i, j]` is 

  0,  3,  4
  5,  0,  6
  7,  8,  0
  9, 10, 11

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


class LearningRateSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, learning_rate, decay_steps, alpha, warmup_steps, warmup_lr):
    super(LearningRateSchedule, self).__init__()
    self._learning_rate = learning_rate
    self._decay_steps = decay_steps
    self._alpha = alpha
    self._warmup_steps = warmup_steps
    self._warmup_lr = warmup_lr

  def __call__(self, global_step):
    global_step = tf.cast(global_step, 'float32')

    cosine_decay = 0.5 * (1 + tf.cos(np.pi * tf.minimum(global_step 
        - self._warmup_steps, self._decay_steps) / self._decay_steps))
    decayed = (1 - self._alpha) * cosine_decay + self._alpha
    decayed_learning_rate = self._learning_rate * decayed

    decayed_learning_rate = tf.where(global_step < self._warmup_steps, 
                                     self._warmup_lr, 
                                     decayed_learning_rate)

    return decayed_learning_rate

   
def cache_memory(memory, embeddings, m_seq_len=None):
  """
  Args:

  Returns:
    
  """
  if m_seq_len is None:
    m_seq_len = tf.shape(memory)[1] #.shape[1]
  new_memory = tf.stop_gradient(
      tf.concat([memory, embeddings], axis=1)[:, -m_seq_len:])
  return new_memory
