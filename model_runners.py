"""Defines Trainer, Evaluator and Inferencer class that wraps a TransformerXL 
model and performs training, evaluation and inference, respectively.
"""
import functools
import os

import numpy as np
import tensorflow as tf


class TransformerXLModelTrainer(object):
  """Trains a TransformerXL model."""
  def __init__(self, model, m_seq_len, batch_size):
    """Constructor.

    Args:
      model: an instance of TransformerXL model.
      m_seq_len: int scalar, length of the memory sequence.
      batch_size: int scalar, batch_size.
    """
    self._model = model
    self._m_seq_len = m_seq_len
    self._batch_size = batch_size

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
        outputs, new_memories = self._model(inputs, memories)
        losses = self._model._embedding_layer(outputs, labels, mode='loss')
        loss = tf.reduce_mean(losses)

      trainable_variables = self._model.trainable_variables
      gradients = tape.gradient(loss, trainable_variables)
      if clip_norm is not None:
        gradients, norm = tf.clip_by_global_norm(gradients, clip_norm)
      optimizer.apply_gradients(
          zip(gradients, trainable_variables))

      step = optimizer.iterations
      lr = optimizer.learning_rate(step)
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
  def __init__(self, model, m_seq_len, batch_size):
    """Constructor.

    Args:
      model: an instance of TransformerXL model.
      m_seq_len: int scalar, length of the memory sequence.
      batch_size: int scalar, batch_size.    
    """
    self._model = model
    self._m_seq_len = m_seq_len
    self._batch_size = batch_size

  def evaluate(self, dataset):
    """Iterate through the validation dataset and compute the perplexity.

    Args:
      dataset: a tf.data.Dataset instance, the input data generator.

    Returns:
      ppl: float scalar, the average per-token perplexity.
    """
    batch_size = self._batch_size
    stack_size = self._model._stack_size
    m_seq_len = self._m_seq_len
    hidden_size = self._model._hidden_size

    memories = tf.zeros((batch_size, stack_size, m_seq_len, hidden_size))
    
    loss_list = []
    def eval_step(inputs, memories, labels):
      outputs, memories = self._model(inputs, memories, training=False)
      losses = self._model._embedding_layer(outputs, labels, mode='loss')
      loss = tf.reduce_mean(losses)
      return loss, memories

    for inputs, labels in dataset:
      loss, memories = eval_step(inputs, memories, labels)
      loss_list.append(loss.numpy())

    ppl = np.exp(np.mean(loss_list))    
    return ppl


def nucleus_sampling(scores, threshold=0.95):
  """Sample from the head of the probability distribution that contains the 
  vast majority of probability mass. See https://arxiv.org/abs/1904.09751 
  for details. 

  Args:
    scores: numpy array of shape [vocab_size], the probability distribution (
      sum to one) of all possible next-tokens over the vocabulary.
    threshold: float scalar, the minimum value of the sum of probability mass
      that the head of the distribution must exceed. 
  """
  ids = np.argsort(-scores)
  cumsum = [0.] + np.cumsum(scores[ids]).tolist()
  low, high = 0, len(cumsum) - 2

  while low <= high:
    mid = (low + high) // 2
    sum1 = cumsum[mid]
    sum2 = cumsum[mid + 1]
    if sum1 < threshold and sum2 >= threshold:
      break
    elif sum2 < threshold: # rule out indices <= `mid` 
      low = mid + 1
    elif sum1 >= threshold: # rule out indices >= `mid`
      high = mid - 1

  probs = scores[ids[:mid + 1]] / sum2
  return np.random.choice(ids[:mid + 1], p=probs) 

def topk_sampling(scores, k=40):
  """
  """
  pass 


class TransformerXLModelInferencer(object):
  """Make inference on the most likely (-ish) sequence of text that logically
  and coherently follows a primer sequence based on a trained TransformerXL 
  model.
  """
  def __init__(self, model, m_seq_len, batch_size, num_tokens=500):
    """Constructor.

    Args:
      model: an instance of TransformerXL model.
      m_seq_len: int scalar, length of the memory sequence.
      q_
      batch_size: int scalar, batch_size.
      num_tokens: int scalar, num of tokens to be generated.
    """
    self._model = model
    self._m_seq_len = m_seq_len
    self._batch_size = batch_size
    self._num_tokens = num_tokens

  def infer(self, primer_token_ids):
    batch_size = self._batch_size
    stack_size = self._model._stack_size
    m_seq_len = self._m_seq_len
    hidden_size = self._model._hidden_size

    memories = tf.zeros([batch_size, stack_size, m_seq_len, hidden_size], dtype='float32')

    _, memories = self._model(primer_token_ids[:, :-1], memories, training=False)
    l = []
    for i in range(self._num_tokens):
      if i == 0:
        init_ids = primer_token_ids[:, -1:]

      outputs, memories = self._model(init_ids, memories, training=False)
      softmax = self._model._embedding_layer(outputs, mode='softmax')
      #ids = np.argsort(-softmax.numpy()[0, 0]) 
      #next_token_id = np.random.choice(ids[:5], 1)[0] 
      next_token_id = nucleus_sampling(softmax.numpy()[0, 0])
      l.append(next_token_id)
      init_ids = tf.constant([[next_token_id]])

    return l







  '''

  def infer(self, primer_token_ids):
    """Generates text based on a primer sequence. 

    Args:
      primer_token_ids: numpy array of shape [batch_size, seq_len], storing the
        token ids of a batch of primer sequences.
    """ 
    batch_size = primer_token_ids.shape[0]
    stack_size = self._model._stack_size
    m_seq_len = self._m_seq_len
    hidden_size = self._model._hidden_size

    memories = tf.zeros([batch_size, stack_size, m_seq_len, hidden_size], dtype='float32')

    print(primer_token_ids[:, :-1].shape, primer_token_ids[:, :-1].numpy().mean(), memories.shape)
    outputs, memories = self._model(primer_token_ids[:, :-1], memories, False)
    print(memories.numpy().mean()) 
    scoring_fn = functools.partial(self._model._embedding_layer, mode='softmax')
    initial_ids = primer_token_ids[:, -1]

    out = self._model.predict(initial_ids, memories, scoring_fn)

    return out

  '''

