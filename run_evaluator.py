import os
import tensorflow as tf
import numpy as np

from model import TransformerXLModel
from model import AdaptiveSoftmaxV1
from model_runners import TransformerXLModelEvaluator

dataset = tf.data.TFRecordDataset('/home/chaoji/Desktop/transformer-xl/tf/data/wikitext-103/tfrecords/valid.bsz-32.tlen-128.tfrecords')

def parse_fn(serialized_example):
   parse_dict = {'inputs': tf.io.VarLenFeature(tf.int64),
                 'labels': tf.io.VarLenFeature(tf.int64)}
   parsed = tf.io.parse_single_example(serialized_example, parse_dict)
   inputs = tf.sparse.to_dense(parsed['inputs'])
   labels = tf.sparse.to_dense(parsed['labels'])
   return inputs, labels

batch_size = 32 

dataset = dataset.map(parse_fn).batch(batch_size)



vocab_size = 267735
stack_size = 8#6
num_heads = 8
filter_size = 2048
dropout_rate = 0.1

m_seq_len = 128
hidden_size = 512

cutoffs = [20000, 40000, 200000]

model = TransformerXLModel(vocab_size,
                           stack_size,
                           hidden_size,
                           num_heads,
                           filter_size,
                           dropout_rate=dropout_rate)


memories = tf.zeros((batch_size, stack_size, m_seq_len, hidden_size))

#adaptive_softmax = AdaptiveSoftmaxV1(hidden_size, cutoffs + [vocab_size])

ckpt = tf.train.Checkpoint(model=model)#, adaptive_softmax=adaptive_softmax)

ckpt_path = '.'
latest_ckpt = tf.train.latest_checkpoint(ckpt_path)
print('latest_ckpt', latest_ckpt)
ckpt.restore(latest_ckpt).expect_partial()


evaluator = TransformerXLModelEvaluator(model, 128, batch_size)

ppl = evaluator.evaluate(dataset)


"""
l = []
for inputs, labels in dataset:
  outputs, memories = model(inputs, memories)
  losses = adaptive_softmax(outputs, labels, 'loss')
  loss = tf.reduce_mean(losses)
  l.append(loss.numpy())
"""

