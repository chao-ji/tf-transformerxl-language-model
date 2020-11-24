import os
import tensorflow as tf
import numpy as np

from model import TransformerXLModel
from model import AdaptiveSoftmaxV1
from utils import LearningRateSchedule

#tf.random.set_seed(0)

dataset = tf.data.TFRecordDataset('/home/chaoji/Desktop/transformer-xl/tf/data/wikitext-103/tfrecords/train.bsz-32.tlen-96.tfrecords')

def parse_fn(serialized_example):
   parse_dict = {'inputs': tf.io.VarLenFeature(tf.int64),
                 'labels': tf.io.VarLenFeature(tf.int64)}
   parsed = tf.io.parse_single_example(serialized_example, parse_dict)
   inputs = tf.sparse.to_dense(parsed['inputs'])
   labels = tf.sparse.to_dense(parsed['labels'])
   return inputs, labels


batch_size = 32

dataset = dataset.map(parse_fn).repeat().batch(batch_size)


vocab_size = 267735
stack_size = 6
num_heads = 8 
filter_size = 2048
dropout_rate = 0.1

m_seq_len = 96 
hidden_size = 512

cutoffs = [20000, 40000, 200000]

model = TransformerXLModel(vocab_size, 
                           stack_size, 
                           hidden_size, 
                           num_heads, 
                           filter_size, 
                           dropout_rate=dropout_rate)

# [6, 32, 50, 410]
memories = tf.zeros((batch_size, stack_size, m_seq_len, hidden_size))


adaptive_softmax = AdaptiveSoftmaxV1(hidden_size, cutoffs + [vocab_size])

clip = 0.25
min_lr_ratio = 0.004

train_steps = 400000
alpha = 0.004
learning_rate = 2.5e-4
warmup_steps = 0 

warmup_lr = 0.0

schedule = LearningRateSchedule(learning_rate=learning_rate, 
                                decay_steps=train_steps-warmup_steps, 
                                alpha=alpha, 
                                warmup_steps=warmup_steps, 
                                warmup_lr=warmup_lr)


optimizer_adam_beta1 = 0.9
optimizer_adam_beta2 = 0.999
optimizer_adam_epsilon = 1e-8


optimizer = tf.keras.optimizers.Adam(
    schedule, 
    optimizer_adam_beta1,
    optimizer_adam_beta2,
    epsilon=optimizer_adam_epsilon)


@tf.function
def train_step(inputs, memories, labels):
  with tf.GradientTape() as tape:
    outputs, new_memories = model(inputs, memories)
    losses = adaptive_softmax(outputs, labels, 'loss') 
    loss = tf.reduce_mean(losses)

  all_vars = model.trainable_variables + adaptive_softmax.trainable_variables

  grads = tape.gradient(loss, all_vars)
  clipped, gnorm = tf.clip_by_global_norm(grads, clip)

  optimizer.apply_gradients(
      zip(clipped, all_vars))

  step = optimizer.iterations
  lr = optimizer.learning_rate(step)
  return loss, new_memories, step - 1, lr

  
ckpt = tf.train.Checkpoint(model=model, adaptive_softmax=adaptive_softmax, optimizer=optimizer)
ckpt_path = '.'
latest_ckpt = tf.train.latest_checkpoint(ckpt_path)
if latest_ckpt:
  print('Restoring from checkpoint: %s ...' % latest_ckpt)
  ckpt.restore(latest_ckpt)
else:
  print('Training from scratch...')
 
for inputs, labels in dataset:
  inputs = tf.cast(inputs, 'int32')
  labels = tf.cast(labels, 'int32')

  loss, memories, step, lr = train_step(inputs, memories, labels)


  if step.numpy() % 100 == 0:
    print(step.numpy(), 'loss', loss.numpy(), 'lr', lr.numpy())

  if step.numpy() % 10000 == 0:
    ckpt.save(os.path.join(ckpt_path, 'transformerxl'))

  if step.numpy() == train_steps:
    break




