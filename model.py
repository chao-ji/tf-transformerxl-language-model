"""Defines TransformerXL model in tf.keras.API."""
import tensorflow as tf

import utils
from commons.tokenization import SOS_ID
from commons.tokenization import EOS_ID
from commons.beam_search import NEG_INF
from commons import beam_search


class AdaptiveEmbedding(tf.keras.layers.Layer):
  def __init__(self,
               hidden_size,
               cutoffs,
               project_factor=4,
               weight_initializer='glorot_uniform'):

    super(AdaptiveEmbedding, self).__init__()
    self._hidden_size = hidden_size
    self._cutoffs = cutoffs
    self._project_factor = project_factor
    self._weight_initializer = 'glorot_uniform'

    self._num_tails = len(self._cutoffs) - 1

  def build(self, inputs_shape):
    """Creates weights of this layer.

    Args:
      inputs_shape: tuple of ints or 1-D int tensor.
    """
    self.add_weight(name='head_weight_proj',
                    shape=(self._hidden_size, self._hidden_size),
                    initializer=self._weight_initializer,
                    dtype='float32',
                    trainable=True)

    self.add_weight(name='head_weight',
                    shape=(self._hidden_size,
                           self._cutoffs[0] + self._num_tails),
                    initializer=self._weight_initializer,
                    dtype='float32',
                    trainable=True)

    current_project_factor = self._project_factor
    for i in range(self._num_tails):
      project_size = max(1, self._hidden_size // current_project_factor)
      current_project_factor *= self._project_factor
      self.add_weight(name='tail_weight_proj_%d' % i,
                      shape=(self._hidden_size, project_size),
                      initializer=self._weight_initializer,
                      dtype='float32',
                      trainable=True)

      tail_size = self._cutoffs[i + 1] - self._cutoffs[i]
      self.add_weight(name='tail_weight_%d' % i,
                      shape=(project_size, tail_size),
                      initializer=self._weight_initializer,
                      dtype='float32',
                      trainable=True)
    super(AdaptiveEmbedding, self).build(inputs_shape)

  def call(self, inputs, labels=None, mode='softmax'):
    if mode == 'softmax':
      return self.compute_softmax(inputs)
    elif mode == 'loss':
      return self.compute_loss(inputs, labels)
    elif mode == 'embeddings':
      return self.compute_embeddings(inputs)

  def compute_loss(self, inputs, labels):
    """Compute the loss corresponding to adaptive softmax.

    Args:
      inputs: float tensor of shape [batch_size, seq_len, hidden_size], the 
        tensor holding input word embeddings computed from the laste layer of 
        TransformerXL model.
      labels: int tensor of shape [batch_size, seq_len], the tensor holding 
        groundtruth token ids.

    Returns:
      losses: float tensor of shape [head_size + tail1_size + tail2_size + ...],
        the per-token loss.
    """
    head_weight = self.trainable_variables[1]

    training_losses = []
    head_labels = labels
    for i in range(self._num_tails):
      tail_weight_proj = self.trainable_variables[i*2+2]
      tail_weight = self.trainable_variables[i*2+3]

      mask = tf.logical_and(tf.greater_equal(labels, self._cutoffs[i]),
                            tf.less(labels, self._cutoffs[i + 1]))

      head_labels = tf.where(mask, self._cutoffs[0] + i, head_labels)

      tail_inputs = tf.boolean_mask(inputs, mask)
      tail_logits = tf.matmul(tf.matmul(
          tail_inputs, tail_weight_proj), tail_weight)
      tail_labels = tf.boolean_mask(labels - self._cutoffs[i], mask)

      tail_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=tail_labels, logits=tail_logits)
      training_losses.append(tail_loss)

    head_logits = tf.matmul(inputs, head_weight)

    head_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=head_labels, logits=head_logits)
    head_loss = tf.reshape(head_loss, [-1])
    training_losses.append(head_loss)

    losses = tf.concat(training_losses, axis=0)
    return losses

  def compute_softmax(self, inputs):
    """Computes adaptive softmax.

    Args:
      inputs: float tensor of shape [batch_size, seq_len, hidden_size], the 
        tensor holding input word embeddings computed from the laste layer of 
        TransformerXL model.

    Returns:
      softmax: float tensor of shape [batch_size, seq_len, vocab_size], the 
        per-token softmax 
    """
    head_weight = self.trainable_variables[1]

    head_logits = tf.matmul(inputs, head_weight)
    head_softmax = tf.nn.softmax(head_logits)

    softmax_list = [head_softmax[:, :, :self._cutoffs[0]]]
    for i in range(self._num_tails):
      tail_weight_proj = self.trainable_variables[i*2+2]
      tail_weight = self.trainable_variables[i*2+3]

      tail_logits = tf.matmul(tf.matmul(inputs, tail_weight_proj), tail_weight)

      tail_softmax = tf.nn.softmax(tail_logits)
      index = self._cutoffs[0] + i
      softmax_list.append(tail_softmax * head_softmax[:, :, index:index+1])

    softmax = tf.concat(softmax_list, axis=-1)
    return softmax


  def compute_embeddings(self, x):
    emb_scale = self._hidden_size ** 0.5
    cutoff_ends = [0] + self._cutoffs
    x_size = tf.shape(x)
    y = tf.zeros([x_size[0], x_size[1], self._hidden_size])
   
    tables = [tf.transpose(w) for w in self.trainable_variables[1::2]]
    tables[0] = tables[0][:self._cutoffs[0]]
    projs = [tf.transpose(w) for w in self.trainable_variables[::2]]

    for i in range(len(cutoff_ends) - 1):

      l_idx, r_idx = cutoff_ends[i], cutoff_ends[i + 1]
      mask = (x >= l_idx) & (x < r_idx)
      cur_x = tf.boolean_mask(x, mask) - l_idx
      cur_d_embed = self._hidden_size // (self._project_factor ** i)  
      
      cur_y = tf.gather(tables[i], cur_x)

      proj_W = projs[i]

      cur_y = tf.matmul(cur_y, proj_W)

      mask_idx = tf.cast(tf.where(mask), 'int64')
      y += tf.scatter_nd(mask_idx, cur_y, tf.cast(tf.shape(y), 'int64'))
      
    y *= emb_scale

    return y




class AdaptiveSoftmaxV1(tf.keras.layers.Layer):
  """Computes the adaptive softmax or the corresponding loss for language models
  with very large vocabulary, according to https://arxiv.org/abs/1609.04309,
  "Efficient softmax approximation for GPUs, Grave et al. 2016".
  """
  def __init__(self, 
               hidden_size, 
               cutoffs, 
               project_factor=4, 
               weight_initializer='glorot_uniform'):
    """Constructor.

    Args:
      hidden_size: int scalar, the hidden size of continuous representation.
      cutoffs: list of ints, the boundaries of word indices (words are sorted in 
        descending order of frequencies in the vocabulary) with which to split 
        the set of words into a head group and potentially multiple tail groups. 
        For example, for `cutoff` = [b0, b1, ..., V], where `V` is the 
        vocabulary size, the head contains words with indices `< b0`, and the 
        first tail contains words with indices `>= b0` and `< b1`, and so on.
      project_factor: int scalar, the factor by which to decrease the hidden 
        size of word embeddings for different tails. For example, words in the
        head has hidden size `hidden_size`, where words in the first tail has
        reduced hidden size `hidden_size // project_factor`, and so on.
      weight_initializer: string scalar, the weight initializer.
    """
    super(AdaptiveSoftmaxV1, self).__init__()
    self._hidden_size = hidden_size
    self._cutoffs = cutoffs
    self._project_factor = project_factor
    self._weight_initializer = 'glorot_uniform'

    self._num_tails = len(self._cutoffs) - 1

  def build(self, inputs_shape):
    """Creates weights of this layer.

    Args:
      inputs_shape: tuple of ints or 1-D int tensor.
    """
    self.add_weight(name='head_weight',
                    shape=(self._hidden_size, 
                           self._cutoffs[0] + self._num_tails),
                    initializer=self._weight_initializer,
                    dtype='float32',
                    trainable=True)  

    current_project_factor = self._project_factor
    for i in range(self._num_tails):
      project_size = max(1, self._hidden_size // current_project_factor)
      current_project_factor *= self._project_factor
      self.add_weight(name='tail_weight_proj_%d' % i, 
                      shape=(self._hidden_size, project_size),
                      initializer=self._weight_initializer,
                      dtype='float32',
                      trainable=True)

      tail_size = self._cutoffs[i + 1] - self._cutoffs[i]
      self.add_weight(name='tail_weight_%d' % i,
                      shape=(project_size, tail_size),
                      initializer=self._weight_initializer,
                      dtype='float32',
                      trainable=True)
    super(AdaptiveSoftmaxV1, self).build(inputs_shape)

  def call(self, inputs, labels=None, mode='softmax'):
    """Computes the forward pass of Adaptive Softmax.

    It operates in either "softmax" mode or "loss" mode:
    
      - In softmax mode, it computes the per-token softmax of shape [batch_size, 
        seq_len, vocab_size], given only input word embeddings.
      - In loss mode, it computes the per-token loss of shape [head_size + 
        tail1_size + tail2_size + ...], given both input word embeddings as well 
        as groundtruth token ids, where `head_size = batch_size * seq_len` and 
        `tail1_size`, `tail2_size`, ..., are the num of words whose indices fall 
        within the range of tail1, tail2, ..., and they may be zero. 

    Args:
      inputs: float tensor of shape [batch_size, seq_len, hidden_size], the 
        tensor holding input word embeddings computed from the laste layer of 
        TransformerXL model.
      labels: (Optional) int tensor of shape [batch_size, seq_len], the tensor 
        holding groundtruth token ids. Must be provided in "loss" mode.
      mode: string scalar, "softmax" or "loss".

    Returns:
      outputs: float tensor of shape [batch_size, seq_len, vocab_size], the 
        per-token softmax, if in "softmax" mode; Or float tensor of shape 
        [head_size + tail1_size + tail2_size + ...], the per-token loss, if in
        "loss" mode. 
    """
    if mode == 'softmax':
      return self.compute_softmax(inputs)
    elif mode == 'loss':
      return self.compute_loss(inputs, labels)
    else:
      raise ValueError('mode must be "softmax" or "loss", got %s' % mode)

  def compute_loss(self, inputs, labels):
    """Compute the loss corresponding to adaptive softmax.

    Args:
      inputs: float tensor of shape [batch_size, seq_len, hidden_size], the 
        tensor holding input word embeddings computed from the laste layer of 
        TransformerXL model.
      labels: int tensor of shape [batch_size, seq_len], the tensor holding 
        groundtruth token ids.

    Returns:
      losses: float tensor of shape [head_size + tail1_size + tail2_size + ...],
        the per-token loss.
    """
    head_weight = self.trainable_variables[0]

    training_losses = []
    head_labels = labels

    for i in range(self._num_tails):
      tail_weight_proj = self.trainable_variables[i*2+1]
      tail_weight = self.trainable_variables[i*2+2]

      mask = tf.logical_and(tf.greater_equal(labels, self._cutoffs[i]), 
                            tf.less(labels, self._cutoffs[i + 1]))

      head_labels = tf.where(mask, self._cutoffs[0] + i, head_labels)

      tail_inputs = tf.boolean_mask(inputs, mask)
      tail_logits = tf.matmul(tf.matmul(
          tail_inputs, tail_weight_proj), tail_weight)
      tail_labels = tf.boolean_mask(labels - self._cutoffs[i], mask)
      
      tail_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=tail_labels, logits=tail_logits)
      training_losses.append(tail_loss) 

    head_logits = tf.matmul(inputs, head_weight)

    head_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=head_labels, logits=head_logits)
    head_loss = tf.reshape(head_loss, [-1])
    training_losses.append(head_loss)
     
    losses = tf.concat(training_losses, axis=0) 
    return losses

  def compute_softmax(self, inputs):
    """Computes adaptive softmax.

    Args:
      inputs: float tensor of shape [batch_size, seq_len, hidden_size], the 
        tensor holding input word embeddings computed from the laste layer of 
        TransformerXL model.

    Returns:
      softmax: float tensor of shape [batch_size, seq_len, vocab_size], the 
        per-token softmax 
    """
    head_weight = self.trainable_variables[0]

    head_logits = tf.matmul(inputs, head_weight)
    head_softmax = tf.nn.softmax(head_logits)

    softmax_list = [head_softmax[:, :, :self._cutoffs[0]]]
    for i in range(self._num_tails):
      tail_weight_proj = self.trainable_variables[i*2+1]
      tail_weight = self.trainable_variables[i*2+2]

      tail_logits = tf.matmul(tf.matmul(inputs, tail_weight_proj), tail_weight)

      tail_softmax = tf.nn.softmax(tail_logits)
      index = self._cutoffs[0] + i
      softmax_list.append(tail_softmax * head_softmax[:, :, index:index+1])

    softmax = tf.concat(softmax_list, axis=-1)
    return softmax








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
    attention_weights += token_mask * NEG_INF
    attention_weights = tf.nn.softmax(attention_weights, axis=3)
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

    #self._embedding_layer = tf.keras.layers.Embedding(
    #    self._vocab_size,
    #    self._hidden_size,
    #    embeddings_initializer=tf.keras.initializers.RandomNormal(stddev=0.02))
    cutoffs = [20000, 40000, 200000]
    self._embedding_layer = AdaptiveEmbedding(hidden_size, cutoffs + [vocab_size])

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
      memories: float tensor of shape [batch_size, num_layers, m_seq_len, 
        hidden_size], embeddings of the tokens from the previous sequence 
        segment for each layer of the decoder stack.
      training: bool scalar, True if in training mode. 

    Returns:
      outputs: float tensor of shape [batch_size, q_seq_len, hidden_size],  

      new_memories: float tensor of shape [batch_size, num_layers, m_seq_len, 
        hidden_size] 
    """
    m_seq_len = tf.shape(memories)[2]
    q_seq_len = tf.shape(inputs)[1]

    r_seq_len = m_seq_len + q_seq_len
    new_memories = []

    # [32, 50, 410]
    #embeddings = self._embedding_layer(inputs) * self._hidden_size ** 0.5
    embeddings = self._embedding_layer(inputs, mode='embeddings')

    # [1, 1, 50, 100]
    attn_mask = utils.get_look_ahead_mask(q_seq_len, m_seq_len)

    # [100, 410] 
    positional_encoding = utils.get_positional_encoding(
        r_seq_len, self._hidden_size) 
    embeddings = self._embeddings_dropout_layer(
        embeddings, training=training)
    positional_encoding = self._positional_encoding_dropout_layer(
        positional_encoding, training=training)

    for i in range(self._stack_size): 
      new_memories.append(utils.cache_memory(memories[:, i], embeddings))

      embeddings = self._stack[i](
          embeddings, positional_encoding, attn_mask, memories[:, i], training)

    outputs = self._dropout_layer(embeddings, training=training)

    new_memories = tf.stack(new_memories, axis=1)
    return outputs, new_memories

  def predict(self, initial_ids, mems, scoring_fn):
    decoding_fn = self._build_decoding_fn(scoring_fn) 

    batch_size = initial_ids.shape[0]
    max_length = self._max_length = 512


    decoding_cache = {'memories': mems}

    self._beam_width = 4 
    self._alpha = 0.6

    bs = beam_search.BeamSearch(decoding_fn,
                                self._vocab_size,
                                batch_size, 
                                self._beam_width,
                                self._alpha,
                                max_length,
                                12312434343) #self._vocab_size)    

    
    print('initial_ids', initial_ids.shape)
    out = bs.search(initial_ids, decoding_cache)
    return out

  def _build_decoding_fn(self, scoring_fn):
    def decoding_fn(decoder_input, cache, **kwargs):
      """
      decoder_input: [batch_size * beam_width, 1]
      memories: [batch_size * beam_width, num_layers, m_seq_len, hidden_size]
      Returns:
        softmax: [batch_size * beam_width, vocab_size]
      """
      memories = cache['memories']

      m_seq_len = tf.shape(memories)[2] #memories.shape[2]
      q_seq_len = tf.shape(decoder_input)[1] #decoder_input.shape[1]
      r_seq_len = m_seq_len + q_seq_len
      new_memories = []

      embeddings = self._embedding_layer(decoder_input, mode='embeddings')
      attn_mask = utils.get_look_ahead_mask(q_seq_len, m_seq_len)

      positional_encoding = utils.get_positional_encoding(
          r_seq_len, self._hidden_size)

      for i in range(self._stack_size):
        new_memories.append(utils.cache_memory(memories[:, i], embeddings))

        embeddings = self._stack[i](
            embeddings, positional_encoding, attn_mask, memories[:, i], training=False)

      scores = scoring_fn(embeddings)

      cache['memories'] = tf.stack(new_memories, axis=1)
      print(cache['memories'].numpy().mean())

      scores = tf.squeeze(scores, axis=1)
      return scores, cache 
    return decoding_fn
