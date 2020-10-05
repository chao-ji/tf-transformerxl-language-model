"""Defines TransformerXL model in tf.keras.API."""
import tensorflow as tf
import numpy as np

import utils


class AdaptiveSoftmax(tf.keras.layers.Layer):
  """Implements adaptive softmax for large vocabulary size."""
  def __init__(self, cutoffs, vocab_size, hidden_size):
    """Constructor.

    Args:
      vocab_size: int scalar, num of entries in the vocabulary.
      hidden_size: int scalar, the hidden size of continuous representation.
    """
    super(AdaptiveSoftmax, self).__init__()
    self._cutoffs = cutoffs
    self._vocab_size = vocab_size 
    self._hidden_size = hidden_size 

    self._cutoff_ends = [0] + cutoffs + [vocab_size]

  def _gather_logprob(self, logprob, labels):
    lp_size = tf.shape(logprob)
    r = tf.range(lp_size[0])
    index = tf.stack([r, labels], 1)
    return tf.gather_nd(logprob, index)  

  def build(self, inputs_shape):
    for i in range(len(self._cutoff_ends) - 1):
      l_index, r_index = self._cutoff_ends[i], self._cutoff_ends[i + 1]
      self.add_weight(name='b%s' % i, shape=(r_index - l_index), 
          initializer='zeros', dtype='float32', trainable=True)
       
    self.add_weight(name='cluster_weight', 
                    shape=[len(self._cutoffs), self._hidden_size], 
                    initializer='zeros')
    self.add_weight(name='cluster_bias', 
                    shape=[len(self._cutoffs)], 
                    initializer='zeros')
 
    super(AdaptiveSoftmax, self).build(inputs_shape)

  def call(self, inputs, labels, embedding_weights):
    """

    Args:
      inputs: float tensor of shape [batch_size, q_seq_len, hidden_size], the 
        embeddings of input tokens.
      labels: int tensor of shape [batch_size, q_sesq_len], the ids of 
        groundtruth next tokens.
      embedding_weights: float tensor of shape [vocab_size, hidden_size], the
        matrix storing the embeddings of each token from the vocabulary.
      
    Returns:
      loss: float tensor of shape [batch_size, q_seq_len], the per token 
        cross-entropy loss.
    """
    neg_log_likelihood = [] #tf.zeros_like(labels, dtype='float32')
    output_shape = labels.shape
    print('inputs', inputs.shape)
    for i in range(len(self._cutoff_ends) - 1):
      l_index, r_index = self._cutoff_ends[i], self._cutoff_ends[i + 1]

      # [batch_size, q_seq_len] 32, 50
      mask = tf.logical_and(labels >= l_index, labels < r_index)
      # [batch_size * q_seq_len, 2] 1600 2
      mask_index = tf.where(mask)

      # [batch_size * q_seq_len] 1600 
      cur_labels = tf.boolean_mask(labels, mask) - l_index

      # [seg_size, hidden_size] 
      cur_W = embedding_weights[l_index: r_index]
      # [seg_size]     
      cur_b = self.trainable_variables[i] 

      if i == 0:
        # [num_segs, hidden_size] 3, 410
        # [num_segs] 3
        cluster_weight, cluster_bias = self.trainable_variables[-2:]
        cur_W = tf.concat([cur_W, cluster_weight], 0)
        cur_b = tf.concat([cur_b, cluster_bias], 0)

        head_logit = tf.matmul(inputs, cur_W, transpose_b=True) + cur_b #self._logit(inputs, cur_W, cur_b)

        head_logprob = tf.nn.log_softmax(head_logit)

        cur_head_logprob = tf.boolean_mask(head_logprob, mask)

        cur_logprob = self._gather_logprob(cur_head_logprob, cur_labels)
      else:
        cur_head_logprob=  tf.boolean_mask(head_logprob, mask)

        cur_hidden = tf.boolean_mask(inputs, mask)

        #tail_logit = tf.squeeze(self._logit(cur_hidden[None], cur_W, cur_b), 0)
        tail_logit = tf.matmul(cur_hidden, cur_W, transpose_b=True) + cur_b

        tail_logprob = tf.nn.log_softmax(tail_logit)

        
        cur_logprob = (cur_head_logprob[:, self._cutoff_ends[1] + i - 1] + 
                    self._gather_logprob(tail_logprob, cur_labels))
      print('logprob', cur_logprob.shape)
      neg_log_likelihood.append(tf.scatter_nd(mask_index, -cur_logprob, labels.shape))
      
    return tf.reduce_sum(neg_log_likelihood, axis=0)

class Projection(tf.keras.layers.Layer):
  def __init__(self,
               num_heads,
               size_per_head,
               kernel_initializer='glorot_uniform',
               mode="split"):
    super(Projection, self).__init__()
    if mode not in ('split', 'merge'):
      raise ValueError('"mode" must be either "split" or "merge".')
    self._num_heads = num_heads
    self._size_per_head = size_per_head
    self._hidden_size = num_heads * size_per_head
    self._kernel_initializer = kernel_initializer
    self._mode = mode

  def build(self, inputs_shape):
    """Creates weights of this layer.

    Args:
      inputs_shape: tuple of ints or 1-D int tensor, the last element 
        corresponds to the depth.
    """
    depth = inputs_shape[-1]
    if depth is None:
      raise ValueError('The depth of inputs must not be None.')

    if self._mode == 'merge':
      kernel_shape = self._num_heads, self._size_per_head, self._hidden_size
    else:
      kernel_shape = self._hidden_size, self._num_heads, self._size_per_head

    self.add_weight(name='kernel',
                    shape=kernel_shape,
                    initializer=self._kernel_initializer,
                    dtype='float32',
                    trainable=True)
    super(Projection, self).build(inputs_shape)

  def call(self, inputs):
    """Performs the projection.

    Args:
      inputs: float tensor of shape [batch_size, seq_len, num_heads, 
        size_per_head] in Merge mode, or float tensor of shape [batch_size, 
        seq_len, hidden_size] in Split mode.

    Returns:
      outputs: float tensor of shape [batch_size, seq_len, hidden_size] in 
        Merge mode, or float tensor of shape [batch_size, seq_len, num_heads, 
        size_per_head] int Split mode.
    """
    kernel = self.trainable_variables[0]
    if self._mode == 'merge':
      outputs = tf.einsum('NTHS,HSD->NTD', inputs, kernel)
    else:
      outputs = tf.einsum('NTD,DHS->NTHS', inputs, kernel)
    return outputs


class FeedForwardNetwork(tf.keras.layers.Layer):

  def __init__(self, hidden_size, filter_size, dropout_rate):
    super(FeedForwardNetwork, self).__init__()
    self._hidden_size = hidden_size 
    self._filter_size = filter_size
    self._dropout_rate = dropout_rate

    self._dense_layer_filter = tf.keras.layers.Dense(
        filter_size, use_bias=True, activation=tf.nn.relu)
    self._dense_layer_output = tf.keras.layers.Dense(hidden_size, use_bias=True)
    self._dropout_layer = tf.keras.layers.Dropout(dropout_rate)

  def call(self, inputs, training):
    outputs = self._dense_layer_filter(inputs)
    outputs = self._dropout_layer(outputs, training=training)
    outputs = self._dense_layer_output(outputs)
    return outputs



class Attention(tf.keras.layers.Layer): 
  """Multi-headed attention layer used in TransformerXL model.""" 
  def __init__(self, hidden_size, num_heads, dropout_rate_attention):
    """Constructor.

    Args:
      hidden_size: int scalar, the hidden size of continuous representation.
      num_heads: int scalar, num of attention heads.
      dropout_rate_attention: float scalar, dropout rate applied on the 
        attention weights.
    """
    super(Attention, self).__init__()
    self._hidden_size = hidden_size
    self._num_heads = num_heads
    self._dropout_rate_attention = dropout_rate_attention
    self._size_per_head = hidden_size // num_heads

    self._dense_layer_query = Projection(
        num_heads, self._size_per_head, mode='split')
    self._dense_layer_key = Projection(
        num_heads, self._size_per_head, mode='split')
    self._dense_layer_value = Projection(
        num_heads, self._size_per_head, mode='split')
    self._dense_layer_r = Projection(
        num_heads, self._size_per_head, mode='split')
    self._dense_layer_output = Projection(
        num_heads, self._size_per_head, mode='merge')
    self._attention_dropout_layer = tf.keras.layers.Dropout(
        dropout_rate_attention)

  def build(self, inputs_shape):
    self.add_weight(name='content_bias',
                    shape=[self._num_heads, self._size_per_head],
                    initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
                    dtype='float32',
                    trainable=True)
    self.add_weight(name='position_bias',
                    shape=[self._num_heads, self._size_per_head],
                    initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
                    dtype='float32',
                    trainable=True)
    super(Attention, self).build(inputs_shape)

  def call(self, 
           query_seqs, 
           positional_encoding, 
           token_mask, 
           memory_seqs, 
           training):
    """Computes new representations of query sequences.

    Args:
      query_seqs: float tensor of shape [batch_size, q_seq_len, hidden_size],
        query_sequences.
      positional_encoding: float tensor of shape [r_seq_len, hidden_size], the
        tensor that encodes positional information (relative to the current 
        position).
      token_mask: float tensor of shape [1, 1, q_seq_len, r_seq_len], populated
        with either 0 (for tokens to keep) or 1 (for tokens to be masked).
      memory_seqs: float tensor of shape [batch_size, m_seq_len, hidden_size],
        memory sequences from the previous segment.  
      training: bool scalar, True if in training mode. 

    Returns:
      outputs: float tensor of shape [batch_size, q_seq_len, hidden_size], the
        new representation of `query_seqs`. 
    """
    # [num_heads, size_per_head]
    content_bias, position_bias = self.weights[:2]

    # [batch_size, r_seq_len, hidden_size]
    reference_seqs = tf.concat([memory_seqs, query_seqs], axis=1)

    # [batch_size, q_seq_len, num_heads, size_per_head] 
    query = self._dense_layer_query(query_seqs)

    # [batch_size, r_seq_len, num_heads, size_per_head]
    key = self._dense_layer_key(reference_seqs)

    # [batch_size, r_seq_len, num_heads, size_per_head] 
    value = self._dense_layer_value(reference_seqs)

    # [1, r_seq_len, hidden_size]
    positional_encoding = positional_encoding[tf.newaxis] 

    # [batch_size, num_heads, q_seq_len, r_seq_len]
    content = tf.einsum('NQHS,NRHS->NHQR', 
                        query + content_bias, 
                        key)
    positions = tf.einsum('NQHS,RHS->NHQR', 
                          query + position_bias, 
                          self._dense_layer_r(positional_encoding)[0])
    positions = utils.rel_shift(positions)

    # [batch_size, num_heads, q_seq_len, r_seq_len]
    attention_weights = (content + positions) / (self._size_per_head ** 0.5)
    attention_weights = attention_weights * (1 - token_mask) -1e30 * token_mask
    attention_weights = tf.nn.softmax(attention_weights, 3)
    attention_weights = self._attention_dropout_layer(
        attention_weights, training=training)

    # [batch_size, q_seq_len, num_heads, size_per_head]
    outputs = tf.einsum('NHQR,NRHS->NQHS', attention_weights, value)

    # [batch_size, q_seq_len, hidden_size]
    outputs = self._dense_layer_output(outputs)
    return outputs 


class DecoderLayer(tf.keras.layers.Layer):
  """The building block that makes the decoder stack of layers, consisting of a 
  self-attention sublayer and a feed-forward sublayer.
  """
  def __init__(self, 
               hidden_size, 
               num_heads, 
               filter_size, 
               dropout_rate, 
               dropout_rate_attention):
    """Constructor.

    Args:
      hidden_size: int scalar, the hidden size of continuous representation.
      num_heads: int scalar, num of attention heads.
      filter_size: int scalar, the depth of the intermediate dense layer of the
        feed-forward sublayer.
      dropout_rate: float scalar, dropout rate for the Dropout layers.   
      dropout_rate_attention: float scalar, dropout rate applied on the 
        attention weights.
    """
    super(DecoderLayer, self).__init__()
    self._hidden_size = hidden_size
    self._num_heads = num_heads
    self._filter_size = filter_size
    self._dropout_rate = dropout_rate
    self._dropout_rate_attention = dropout_rate_attention

    self._mha = Attention(
        hidden_size, num_heads, dropout_rate_attention)
    self._layernorm_mha = tf.keras.layers.LayerNormalization(epsilon=1e-12)
    self._dropout_mha = tf.keras.layers.Dropout(dropout_rate)

    self._ffn = FeedForwardNetwork(hidden_size, filter_size, dropout_rate) 
    self._layernorm_ffn = tf.keras.layers.LayerNormalization(epsilon=1e-12)
    self._dropout_ffn = tf.keras.layers.Dropout(dropout_rate)

  def call(
      self, inputs, positional_encoding, look_ahead_mask, memories, training):
    """Computes the output of the decoder layer.

    Args:
      inputs: float tensor of shape [batch_size, q_seq_len, hidden_size], query
        sequences.
      positoinal_encoding: float tensor of shape [r_seq_len, hidden_size], the
        tensor that encodes positional information (relative to the current 
        position).
      look_ahead_mask: float tensor of shape [1, 1, q_seq_len, r_seq_len], 
        populated with either 0 (for tokens to keep) or 1 (for tokens to be 
        masked).
      memories: float tensor of shape [batch_size, m_seq_len, hidden_size],
        memory sequences from the previous segment.
      training: bool scalar, True if in training mode.

    Returns:
      outputs: float tensor of shape [batch_size, q_seq_len, hidden_size], the
        new representation of `inputs`.
    """
    outputs = self._mha(
        inputs, positional_encoding, look_ahead_mask, memories, training)
    outputs = self._dropout_mha(outputs, training=training)
    ffn_inputs = self._layernorm_mha(outputs + inputs)

    outputs = self._ffn(ffn_inputs, training)
    outputs = self._dropout_ffn(outputs, training=training) 
    outputs = self._layernorm_ffn(outputs + ffn_inputs) 
    return outputs


class TransformerXLModel(tf.keras.Model):
  """TransformerXL neural architecture for language modeling as described in 
  https://arxiv.org/abs/1706.03762
  """
  def __init__(self,
               vocab_size,
               stack_size=6,
               hidden_size=512,
               num_heads=8,
               filter_size=2048,
               dropout_rate=0.1,
               dropout_rate_attention=0.0):
    """Constructor.

    Args:
      vocab_size: int scalar, num of entries in the vocabulary.
      stack_size: int scalar, num of layers in the decoder stack.
      hidden_size: int scalar, the hidden size of continuous representation.
      num_heads: int scalar, num of attention heads.
      filter_size: int scalar, the depth of the intermediate dense layer of the
        feed-forward sublayer.
      dropout_rate: float scalar, dropout rate for the Dropout layers.   
      dropout_rate_attention: float scalar, dropout rate applied on the 
        attention weights.       
    """
    super(TransformerXLModel, self).__init__()
    self._vocab_size = vocab_size
    self._stack_size = stack_size
    self._hidden_size = hidden_size
    self._num_heads = num_heads
    self._filter_size = filter_size
    self._dropout_rate = dropout_rate
    self._dropout_rate_attention = dropout_rate_attention

    self._embedding_layer = tf.keras.layers.Embedding(
        self._vocab_size,
        self._hidden_size,
        embeddings_initializer=tf.keras.initializers.RandomNormal(stddev=0.02))
    self._embeddings_dropout_layer = tf.keras.layers.Dropout(dropout_rate)
    self._positional_encoding_dropout_layer = tf.keras.layers.Dropout(
        dropout_rate)
    self._dropout_layer = tf.keras.layers.Dropout(dropout_rate)
    self._stack = [DecoderLayer(hidden_size, 
                                num_heads, 
                                filter_size, 
                                dropout_rate, 
                                dropout_rate_attention) 
                                    for _ in range(self._stack_size)]

  def call(self, inputs, memories, training=True):
    """Takes as input the token ids of a batch of sequence segments as well
    as the embeddings of tokens from the previous sequence segment, and 
    computes the estimated logits of the immediate "next" tokens of each token
    in inputs.

    Args:
      inputs: int tensor of shape [batch_size, q_seq_len], token ids of the 
        input sequence segments.
      memories: float tensor of shape [num_layers, batch_size, m_seq_len, 
        hidden_size], embeddings of the tokens from the previous sequence 
        segment for each layer of the decoder stack.
      training: bool scalar, True if in training mode. 

    Returns:
      outputs: float tensor of shape [batch_size, q_seq_len, hidden_size],  
    """
    m_seq_len = memories.shape[2]
    q_seq_len = inputs.shape[1]
    r_seq_len = m_seq_len + q_seq_len
    new_memories = []

    # [32, 50, 410]
    embeddings = self._embedding_layer(inputs) * self._hidden_size ** 0.5
    print('embedddings', embeddings.shape)

    # [1, 1, 50, 100]
    attn_mask = utils.get_look_ahead_mask(q_seq_len, m_seq_len)
    print('attn_mask', attn_mask.shape)

    # [100, 410] 
    positional_encoding = utils.get_positional_encoding(
        r_seq_len, self._hidden_size) 
    print('positional_encoding', positional_encoding.shape) 
    embeddings = self._embeddings_dropout_layer(
        embeddings, training=training)
    positional_encoding = self._positional_encoding_dropout_layer(
        positional_encoding, training=training)

    for i in range(self._stack_size): 
      new_memories.append(tf.stop_gradient(tf.identity(embeddings)))
      embeddings = self._stack[i](
          embeddings, positional_encoding, attn_mask, memories[i], training)

    outputs = self._dropout_layer(embeddings, training=training)

    new_memories = tf.stack(new_memories, axis=0)
    print('inputs', inputs.shape)
    print('memories', memories.shape)
    print('outputs', outputs.shape)
    print('new_memories', new_memories.shape)
    return outputs, new_memories

  def step(self, decoder_input, memories):
    m_seq_len = memories.shape[2]
    q_seq_len = 1
    r_seq_len = m_seq_len + q_seq_len

    new_memories = []

    embeddings = self._embedding_layer(inputs) * self._hidden_size ** 0.5

    positional_encoding = utils.get_positional_encoding(
        r_seq_len, self._hidden_size)

    for i in range(self._stack_size):
      new_memories
      embeddings = self._stack[i](
          embeddings, positional_encoding,  )
