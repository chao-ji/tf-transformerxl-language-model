"""Defines TransformerXL model in tf.keras.API."""
import tensorflow as tf

import utils
from commons.layers import Projection
from commons.layers import FeedForwardNetwork 
from commons.layers import AdaptiveInputSoftmax
from commons.layers import EmbeddingLayer
from commons.beam_search import NEG_INF


class Attention(tf.keras.layers.Layer): 
  """Multi-headed attention layer used in TransformerXL model.""" 
  def __init__(self, hidden_size, num_heads, dropout_rate_attention):
    """Constructor.

    Args:
      hidden_size: int scalar, the hidden size of continuous representation.
      num_heads: int scalar, num of attention heads.
      dropout_rate_attention: float scalar, dropout rate applied on the 
        query-to-reference attention matrix. 
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
    """Creates weights of this layer.

    Args:
      inputs_shape: tuple of ints or 1-D int tensor, the last element
        corresponds to the depth. 
    """
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
    """Computes new representation of query sequences.

    Args:
      query_seqs: float tensor of shape [batch_size, q_seq_len, hidden_size],
        query_sequences.
      positional_encoding: float tensor of shape [q_seq_len + m_seq_len, 
        hidden_size], the tensor that encodes positional information of 
        `query_seqs` and `memory_seqs` concatenated along the time step axis.
      token_mask: float tensor of shape [1, 1, q_seq_len, q_seq_len + m_seq_len]
        , populated with either 0 (for tokens to keep) or 1 (for tokens to be 
        masked).
      memory_seqs: float tensor of shape [batch_size, m_seq_len, hidden_size],
        memory sequences from the previous segment.  
      training: bool scalar, True if in training mode. 

    Returns:
      outputs: float tensor of shape [batch_size, q_seq_len, hidden_size], the
        new representation of `query_seqs`. 
    """
    # [num_heads, size_per_head]
    content_bias, position_bias = self.weights[:2]

    # [batch_size, q_seq_len + m_seq_len, hidden_size]
    reference_seqs = tf.concat([memory_seqs, query_seqs], axis=1)

    # [batch_size, q_seq_len, num_heads, size_per_head] 
    query = self._dense_layer_query(query_seqs)

    # [batch_size, q_seq_len + m_seq_len, num_heads, size_per_head]
    key = self._dense_layer_key(reference_seqs)

    # [batch_size, q_seq_len + m_seq_len, num_heads, size_per_head] 
    value = self._dense_layer_value(reference_seqs)

    # [1, q_seq_len + m_seq_len, hidden_size]
    positional_encoding = positional_encoding[tf.newaxis] 

    # [batch_size, num_heads, q_seq_len, q_seq_len + m_seq_len]
    content = tf.einsum('NQHS,NRHS->NHQR', 
                        query + content_bias, 
                        key)
    positions = tf.einsum('NQHS,RHS->NHQR', 
                          query + position_bias, 
                          self._dense_layer_r(positional_encoding)[0])
    positions = utils.rel_shift(positions)

    # [batch_size, num_heads, q_seq_len, q_seq_len + m_seq_len]
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
        query-to-reference attention matrix. 
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
      inputs: float tensor of shape [batch_size, q_seq_len, hidden_size], the 
        input sequences whose "next-token" sequences we need to predict.
      positional_encoding: float tensor of shape [q_seq_len + m_seq_len, 
        hidden_size], the tensor that encodes positional information of 
        `query_seqs` and `memory_seqs` concatenated along the time step axis.
      look_ahead_mask: float tensor of shape [1, 1, q_seq_len, q_seq_len + 
        m_seq_len], populated with either 0 (for tokens to keep) or 1 (for 
        tokens to be masked).
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
               adaptive_embedding,
               vocab_size,
               cutoffs=None,
               stack_size=6,
               hidden_size=512,
               num_heads=8,
               filter_size=2048,
               dropout_rate=0.1,
               dropout_rate_attention=0.0):
    """Constructor.

    Args:
      adaptive_embedding: bool scalar, whether to use adaptive token embedding
        and softmax for large vocabulary.
      vocab_size: int scalar, vocabulary size.
      cutoffs: list of ints, boundaries of the token IDs in the vocabulary use
        to split tokens in to head and multiple tails.
      stack_size: int scalar, num of layers in the decoder stack.
      hidden_size: int scalar, the hidden size of continuous representation.
      num_heads: int scalar, num of attention heads.
      filter_size: int scalar, the depth of the intermediate dense layer of the
        feed-forward sublayer.
      dropout_rate: float scalar, dropout rate for the Dropout layers.   
      dropout_rate_attention: float scalar, dropout rate applied on the 
        query-to-reference attention matrix. 
    """
    super(TransformerXLModel, self).__init__()
    self._adaptive_embedding = adaptive_embedding
    self._vocab_size = vocab_size
    self._cutoffs = cutoffs
    self._stack_size = stack_size
    self._hidden_size = hidden_size
    self._num_heads = num_heads
    self._filter_size = filter_size
    self._dropout_rate = dropout_rate
    self._dropout_rate_attention = dropout_rate_attention

    if adaptive_embedding:
      if cutoffs is None:
        raise ValueError('`cutoffs` must be provided if using adaptive '
            'embedding.')
      cutoffs = cutoffs + [vocab_size]
      self._embedding_layer = AdaptiveInputSoftmax(hidden_size, cutoffs)
    else:
      self._embedding_layer = EmbeddingLayer(vocab_size, hidden_size)

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
    computes the embedding vectors of the immediate "next" tokens of each token
    in inputs.

    Args:
      inputs: int tensor of shape [batch_size, q_seq_len], token ids of the 
        input sequence segment.
      memories: float tensor of shape [batch_size, stach_size, m_seq_len, 
        hidden_size], embeddings of the tokens from the previous sequence 
        segment for each layer of the decoder stack.
      training: bool scalar, True if in training mode. 

    Returns:
      outputs: float tensor of shape [batch_size, q_seq_len, hidden_size], the
        final embedding vectors of the input sequence segment. 
      new_memories: float tensor of shape [batch_size, stack_size, m_seq_len, 
        hidden_size], the updated embedding vectors of the memory sequence 
        segment. 
    """
    embeddings, new_memories = self._get_final_embeddings(
        inputs, memories, training)
    outputs = self._dropout_layer(embeddings, training=training)
    return outputs, new_memories

  def _build_decoding_fn(self, scoring_fn):
    """Builds a callback function needed for beam search.

    Args:
      scoring_fn: a callable that converts the embedding vectors (float tensor 
        of shape [batch_size, seq_len, hidden_size]) to scores (float tensor of 
        shape [batch_size, seq_len, vocab_size], i.e. logits or softmax'ed 
        probabilities)

    Returns:
      decoding_fn: a callable that outputs the scores of the next decoded token
        ids.
    """
    def decoding_fn(decoder_input, cache, **kwargs):
      """Computes the scores of the next decoded token ids.

      Args:
        decoder_input: int tensor of shape [batch_size * beam_width, 1], the 
          decoded tokens at index `i`.
        cache: dict of entries:
          'memories': float tensor of shape [batch_size, stack_size, m_seq_len, 
            hidden_size], embeddings of the tokens from the previous sequence 
            segment for each layer of the decoder stack. 

      Returns:
        scores: float tensor of shape [batch_size * beam_width, vocab_size].
        cache: a dict with the same structure as the input `cache`.
      """
      embeddings, new_memories = self._get_final_embeddings(
          decoder_input, cache['memories'], training=False)
      cache['memories'] = new_memories 
      scores = tf.squeeze(scoring_fn(embeddings), axis=1)
      return scores, cache

    return decoding_fn

  def _get_final_embeddings(self, inputs, memories, training):
    """Computes the final embedding vectors coming off the top layer of 
    TransformerXL model.

    Args:
      inputs: int tensor of shape [batch_size, q_seq_len], token ids of the 
        input sequence segment.
      memories: float tensor of shape [batch_size, stack_size, m_seq_len, 
        hidden_size], embeddings of the tokens from the previous sequence 
        segment for each layer of the decoder stack.
      training: bool scalar, True if in training mode. 

    Returns:
      embeddings: float tensor of shape [batch_size, q_seq_len, hidden_size],
        the final embeddings of inputs.
      new_memories: float tensor of shape [batch_size, stack_size, m_seq_len, 
        hidden_size], the updated embedding vectors of the memory sequence 
        segment.
    """
    m_seq_len = tf.shape(memories)[2]
    q_seq_len = tf.shape(inputs)[1]
    new_memories = []

    # [batch_size, q_seq_len, hidden_size]
    embeddings = self._embedding_layer(inputs, mode='embedding')

    # [1, 1, q_seq_len, q_seq_len + m_seq_len]
    attention_mask = utils.get_look_ahead_mask(q_seq_len, m_seq_len)

    # [q_seq_len + m_seq_len, hidden_size] 
    positional_encoding = utils.get_positional_encoding(
          m_seq_len + q_seq_len, self._hidden_size)
    
    embeddings = self._embeddings_dropout_layer(
        embeddings, training=training)
    positional_encoding = self._positional_encoding_dropout_layer(
        positional_encoding, training=training)

    for i in range(self._stack_size):
      new_memories.append(utils.cache_memory(memories[:, i], embeddings))
      embeddings = self._stack[i](embeddings, 
                                  positional_encoding, 
                                  attention_mask, 
                                  memories[:, i], 
                                  training=training)
    new_memories = tf.stack(new_memories, axis=1)
    return embeddings, new_memories
