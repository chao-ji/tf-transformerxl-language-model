"""Defines Trainer, Evaluator and Inferencer class that wraps a TransformerXL 
model and performs training, evaluation and inference, respectively.
"""
import functools
import os

import numpy as np
import tensorflow as tf
from commons import beam_search
from commons import utils
from commons.tokenization import EOS_ID


class TransformerXLModelTrainer(object):
  """Trains a TransformerXL model."""
  def __init__(self, 
               model, 
               m_seq_len,
               batch_size, 
               adaptive_embedding):
    """Constructor.

    Args:
      model: an instance of TransformerXL model.
      m_seq_len: int scalar, length of the memory sequence.
      batch_size: int scalar, batch size.
      adaptive_embedding: bool scalar, whether to use adaptive embedding (and 
        softmax) layer.
    """
    self._model = model
    self._m_seq_len = m_seq_len
    self._batch_size = batch_size
    self._adaptive_embedding = adaptive_embedding

  def train(self,
            dataset,
            optimizer,
            ckpt,
            ckpt_path,
            num_iterations,
            persist_per_iterations,
            clip_norm=None,
            log_per_iterations=100,
            logdir='log'):
    """Run training iterations.

    Args:
      dataset: a tf.data.Dataset instance, the input data generator.
      optimizer: a tf.keras.optimizer.Optimizer instance, applies gradient 
        updates.
      ckpt: a tf.train.Checkpoint instance, saves or load weights to/from 
        checkpoint file.
      ckpt_path: string scalar, the path to the directory that the checkpoint 
        files will be written to or loaded from.
      num_iterations: int scalar, num of iterations to train the model.
      persist_per_iterations: int scalar, saves weights to checkpoint files
        every `persist_per_iterations` iterations.
      clip_norm: float scalar, the max absolute value of the norm the gradient 
        tensors. 
      log_per_iterations: int scalar, prints log info every `log_per_iterations`
        iterations.
      logdir: string scalar, the directory that the tensorboard log data will
        be written to.
    """
    batch_size = self._batch_size
    stack_size = self._model._stack_size
    m_seq_len = self._m_seq_len
    hidden_size = self._model._hidden_size

    train_step_signature = [
        tf.TensorSpec(shape=(batch_size, None), dtype='int32'),
        tf.TensorSpec(shape=(batch_size, stack_size, m_seq_len, hidden_size), 
            dtype='float32'),
        tf.TensorSpec(shape=(batch_size, None), dtype='int32')]
    
    @tf.function(input_signature=train_step_signature)
    def train_step(inputs, memories, labels):
      with tf.GradientTape() as tape:
        outputs, new_memories = self._model(inputs, memories, training=True)
        if self._adaptive_embedding:
          losses = self._model._embedding_layer(outputs, labels, mode='loss')
        else:
          logits = self._model._embedding_layer(outputs, mode='logits')
          losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
              labels=labels, logits=logits)

        loss = tf.reduce_mean(losses)

      trainable_variables = self._model.trainable_variables
      gradients = tape.gradient(loss, trainable_variables)
      if clip_norm is not None:
        gradients, norm = tf.clip_by_global_norm(gradients, clip_norm)
      optimizer.apply_gradients(
          zip(gradients, trainable_variables))

      step = optimizer.iterations
      lr = optimizer.learning_rate
      return loss, new_memories, step - 1, lr

    summary_writer = tf.summary.create_file_writer(logdir)

    latest_ckpt = tf.train.latest_checkpoint(ckpt_path)
    if latest_ckpt:
      print('Restoring from checkpoint: %s ...' % latest_ckpt)
      ckpt.restore(latest_ckpt)
    else:
      print('Training from scratch...')

    memories = tf.zeros((batch_size, stack_size, m_seq_len, hidden_size))

    for inputs, labels in dataset:
      loss, memories, step, lr = train_step(inputs, memories, labels)

      with summary_writer.as_default():
        tf.summary.scalar('train_loss', loss, step=step)
        tf.summary.scalar('learning_rate', lr, step=step)

      if step.numpy() % log_per_iterations == 0:
        print('global step: %d, loss: %f, learning rate:' %
            (step.numpy(), loss.numpy()), lr.numpy())
      if step.numpy() % persist_per_iterations == 0:
        print('Saving checkpoint at global step %d ...' % step.numpy())
        ckpt.save(os.path.join(ckpt_path, 'transformerxl'))

      if step.numpy() == num_iterations:
        break


class TransformerXLModelEvaluator(object):
  """Evaluates a trained TransformerXL model in terms of per-token perplexity.
  """
  def __init__(self, model, m_seq_len, batch_size, vocab_size, adaptive_embedding):
    """Constructor.

    Args:
      model: an instance of TransformerXL model.
      m_seq_len: int scalar, length of the memory sequence.
      batch_size: int scalar, batch size.    
      adaptive_embedding: bool scalar, whether to use adaptive embedding (and 
        softmax) layer.
    """
    self._model = model
    self._m_seq_len = m_seq_len
    self._batch_size = batch_size
    self._adaptive_embedding = adaptive_embedding

  def evaluate(self, dataset):
    """Iterate through the validation dataset and compute the perplexity.

    Args:
      dataset: a tf.data.Dataset instance, the input data generator.

    Returns:
      perplexity: float scalar, the average per-token perplexity.
    """
    batch_size = self._batch_size
    stack_size = self._model._stack_size
    m_seq_len = self._m_seq_len
    hidden_size = self._model._hidden_size

    memories = tf.zeros((batch_size, stack_size, m_seq_len, hidden_size))
    
    loss_list = []
    def eval_step(inputs, memories, labels):
      outputs, memories = self._model(inputs, memories, training=False)
      if self._adaptive_embedding:
        losses = self._model._embedding_layer(outputs, labels, mode='loss')
      else:
        logits = self._model._embedding_layer(outputs, mode='logits')
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits)

      loss = tf.reduce_mean(losses)

      return loss, memories

    for inputs, labels in dataset:
      loss, memories = eval_step(inputs, memories, labels)
      loss_list.append(loss.numpy())

    perplexity = np.exp(np.mean(loss_list))    
    return perplexity 


class TransformerXLModelInferencer(object):
  """Make inference on the most likely (-ish) sequence of text that logically
  and coherently follows a prompt (i.e. a piece of text that gives a "context") 
  based on a trained TransformerXL model.
  """
  def __init__(self, 
               model, 
               m_seq_len, 
               batch_size,
               adaptive_embedding,
               decoding_method, 
               num_tokens=512,
               beam_width=4,
               alpha=0.6,
               batch_memory_processing=False):
    """Constructor.

    Args:
      model: an instance of TransformerXL model.
      m_seq_len: int scalar, length of the memory sequence.
      batch_size: int scalar, batch_size.
      adaptive_embedding: bool scalar, whether to use adaptive embedding (and 
        softmax) layer.
      decoding_method: string scalar, decoding method. Must be "nucleus", 'topk'
        or "beam_search".
      num_tokens: int scalar, num of tokens to be generated.
      beam_width: int scalar, number of beams for beam search. Ignored if 
        decoding method is not beam search.
      alpha: float scalar, defining the strength of length normalization. 
        Ignored if decoding method is not beam search.
      batch_memory_processing: bool scalar, whether to compute the sequence
        embeddings in the memory segment batchwise, or one at a time.
    """
    if decoding_method not in ('nucleus', 'topk', 'beam_search'):
      raise ValueError('`decoding_method` must be either nucleus, topk or '
          'beam_search, got %s' % decoding_method)
    self._model = model
    self._m_seq_len = m_seq_len
    self._batch_size = batch_size
    self._adaptive_embedding = adaptive_embedding
    self._decoding_method = decoding_method
    self._num_tokens = num_tokens
    self._beam_width = beam_width
    self._alpha = alpha
    self._batch_memory_processing = batch_memory_processing

  def infer(self, prompt_token_ids):
    """Generate text based on the prompted text.

    Args:
      prompt_token_ids: int tensor of shape [1, seq_len], token ids of the 
        prompted text.

    Returns:
      token_id_list: a list of integers, the token ids of the generated text. 
    """
    batch_size = self._batch_size
    stack_size = self._model._stack_size
    m_seq_len = self._m_seq_len
    hidden_size = self._model._hidden_size

    memories = tf.zeros((batch_size, stack_size, m_seq_len, hidden_size))

    if self._batch_memory_processing:
      _, memories = self._model(
        prompt_token_ids[:, :-1], memories, training=False)
    else:
      for pos in prompt_token_ids[0, :-1]:
        _, memories = self._model(pos[tf.newaxis, tf.newaxis], memories,
          training=False)

    if self._decoding_method != 'beam_search':
      if self._decoding_method == 'nucleus':
        sampling_fn = utils.nucleus_sampling
      else:
        sampling_fn = utils.topk_sampling
      
      token_id_list = []
      for i in range(self._num_tokens):
        if i == 0:
          init_ids = prompt_token_ids[:, -1:]

        outputs, memories = self._model(init_ids, memories, training=False)
        if self._adaptive_embedding:
          scores = self._model._embedding_layer(outputs, mode='softmax')
        else:
          scores = self._model._embedding_layer(outputs, mode='logits')
          scores = tf.nn.softmax(scores, axis=-1)

        next_token_id = sampling_fn(scores.numpy()[0, 0])
        token_id_list.append(next_token_id)
        init_ids = tf.constant([[next_token_id]])
        if next_token_id == EOS_ID:
          break
    else:
      if self._adaptive_embedding:
        scoring_fn = functools.partial(
            self._model._embedding_layer, mode='softmax')
      else:
        def scoring_fn(inputs):
          logits = self._model._embedding_layer(inputs, mode='logits')
          return tf.nn.softmax(logits, axis=-1)

      initial_ids = prompt_token_ids[:, -1]

      decoding_fn = self._model._build_decoding_fn(scoring_fn)
      decoding_cache = {'memories': memories}
      bs = beam_search.BeamSearch(decoding_fn,
                                  self._model._vocab_size,
                                  batch_size,
                                  self._beam_width,
                                  self._alpha,
                                  self._num_tokens,
                                  EOS_ID,
                                  logits_as_scores=False)

      outputs, _, _ = bs.search(initial_ids, decoding_cache)
      token_id_list = outputs[0, 0].numpy().tolist()

    return token_id_list
