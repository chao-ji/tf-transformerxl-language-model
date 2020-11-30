import os
import tensorflow as tf
import numpy as np

from model import TransformerXLModel
from model import AdaptiveSoftmaxV1
from utils import LearningRateSchedule
from model_runners import TransformerXLModelTrainer

dataset = tf.data.TFRecordDataset('/home/chaoji/Desktop/transformer-xl/tf/data/wikitext-103/tfrecords/train.bsz-32.tlen-128.tfrecords')

def parse_fn(serialized_example):
  parse_dict = {'inputs': tf.io.VarLenFeature(tf.int64),
                 'labels': tf.io.VarLenFeature(tf.int64)}
  parsed = tf.io.parse_single_example(serialized_example, parse_dict)
  inputs = tf.sparse.to_dense(parsed['inputs'])
  labels = tf.sparse.to_dense(parsed['labels'])
  inputs = tf.cast(inputs, 'int32')
  labels = tf.cast(labels, 'int32')
  return inputs, labels


batch_size = 32


dataset = dataset.map(parse_fn).repeat().batch(batch_size)


vocab_size = 267735
stack_size = 8#6
num_heads = 8
filter_size = 2048
dropout_rate = 0.1

m_seq_len = 128 #96
hidden_size = 512

cutoffs = [20000, 40000, 200000]

model = TransformerXLModel(vocab_size,
                           stack_size,
                           hidden_size,
                           num_heads,
                           filter_size,
                           dropout_rate=dropout_rate)


#adaptive_softmax = AdaptiveSoftmaxV1(hidden_size, cutoffs + [vocab_size])

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


ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)

ckpt_path = '.'
clip_norm = 0.25
save_ckpt_per_step = 10000


trainer = TransformerXLModelTrainer(model, m_seq_len, batch_size)

trainer.train(dataset, optimizer, ckpt, ckpt_path, train_steps, save_ckpt_per_step, clip_norm)

 
