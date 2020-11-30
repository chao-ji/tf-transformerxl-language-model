import tensorflow as tf
import numpy as np
import functools
import os


class TransformerXLModelTrainer(object):
  """"""

  def __init__(self, model, m_seq_len, batch_size):
    """Constructor.

    Args:
      model:
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
    """"""
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
        #losses = self._adaptive_softmax(outputs, labels, 'loss')
        losses = self._model._embedding_layer(outputs, labels, mode='loss')
        loss = tf.reduce_mean(losses)

      trainable_variables = self._model.trainable_variables #+ 
                             #self._adaptive_softmax.trainable_variables)
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
  """"""
  def __init__(self, model, m_seq_len, batch_size):
    """"""
    self._model = model
    self._m_seq_len = m_seq_len
    self._batch_size = batch_size

  def evaluate(self, dataset):
    """"""
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

    return np.exp(np.mean(loss_list))    


class TransformerXLModelInferencer(object):
  """"""
  def __init__(self, model, m_seq_len):
    """"""
    self._model = model
    self._m_seq_len = m_seq_len

  def infer(self, primer_token_ids):
    """
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
