import sys
from pathlib import Path
import time
import re
import os
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
from tensorflow.compat.v1.nn.rnn_cell import BasicLSTMCell
import random
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

CURRENT_DIR = Path('.')
UTILS_DIR = CURRENT_DIR / '../'
sys.path.append(UTILS_DIR.absolute().as_posix())



def batched_gather1(tensor, indices):
    """Gather in batch from a tensor of arbitrary size.

    In pseduocode this module will produce the following:
    output[i] = tf.gather(tensor[i], indices[i])

    Args:
      tensor: Tensor of arbitrary size.
      indices: Vector of indices.
    Returns:
      output: A tensor of gathered values.
    """
    shape = (tensor.get_shape().as_list())
    flat_first = tf.reshape(tensor, [shape[0] * shape[1]] + shape[2:])
    indices = tf.convert_to_tensor(indices)
    offset_shape = [shape[0]] + [1] * (indices.shape.ndims - 1)
    offset = tf.reshape(tf.range(shape[0]) * shape[1], offset_shape)
    output = tf.gather(flat_first, indices + offset)
    return output 

class ABLSTM(object):
    def __init__(self, config):
        self.max_len = config["max_len"]
        self.hidden_size = config["hidden_size"]
        self.vocab_size = config["vocab_size"]
        self.embedding_size = config["embedding_size"]
        self.n_class = config["n_class"]
        self.learning_rate = config["learning_rate"]
        self.label_neg1 = tf.placeholder(tf.int32, [config["batch_size"]], name='input_y1')
        self.label_neg2 = tf.placeholder(tf.int32, [config["batch_size"]], name='input_y2')
        self.label_neg3 = tf.placeholder(tf.int32, [config["batch_size"]], name='input_y3')
        self.label_neg4 = tf.placeholder(tf.int32, [config["batch_size"]], name='input_y4')
        self.label_neg5 = tf.placeholder(tf.int32, [config["batch_size"]], name='input_y5')
        self.label_neg6 = tf.placeholder(tf.int32, [config["batch_size"]], name='input_y6')
        # placeholder
        self.x = tf.placeholder(tf.int32, [config["batch_size"], self.max_len])
        self.label = tf.placeholder(tf.float32, [config["batch_size"], self.n_class], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32)

    def build_graph(self):
        print("building graph")
        
        # Word embedding
        self.embeddings_var = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0),
                                     trainable=True)
        
        self.W_concept = tf.Variable(tf.random_uniform([self.n_class, self.embedding_size], -1.0, 1.0), name="LDAV",trainable=True)

        self.batch_embedded = tf.nn.embedding_lookup(self.embeddings_var, self.x)

        # AVERAGE EMBEDDING
        self.em = tf.reduce_mean(self.batch_embedded,1, name="mean")

        rnn_outputs, _ = bi_rnn(BasicLSTMCell(self.hidden_size),
                                BasicLSTMCell(self.hidden_size),
                                inputs=self.batch_embedded, dtype=tf.float32)

        self.fw_outputs, self.bw_outputs = rnn_outputs


        W = tf.Variable(tf.random_normal([self.hidden_size], stddev=0.1))
        self.H = self.fw_outputs + self.bw_outputs # (batch_size, seq_len, HIDDEN_SIZE)
        self.avg_hidden = tf.nn.tanh(tf.reduce_sum(self.H, 1))
        M = tf.tanh(self.H)  # M = tanh(H)  (batch_size, seq_len, HIDDEN_SIZE)

        self.alpha = tf.nn.softmax(tf.reshape(tf.matmul(tf.reshape(M, [-1, self.hidden_size]),
                                                        tf.reshape(W, [-1, 1])),
                                              (-1, self.max_len)))  # batch_size x seq_len
        self.r = tf.matmul(tf.transpose(self.H, [0, 2, 1]),
                      tf.reshape(self.alpha, [-1, self.max_len, 1]))
        self.tr = tf.transpose(self.H, [0, 2, 1])
        r = tf.squeeze(self.r)
        h_star = tf.tanh(r)  # (batch , HIDDEN_SIZE

        self.h_drop = h_star
        # Fully connected layer（dense layer)
        FC_W = tf.Variable(tf.truncated_normal([self.hidden_size, self.n_class], stddev=0.1))
        FC_b = tf.Variable(tf.constant(0., shape=[self.n_class]))
        self.logits = tf.nn.xw_plus_b(self.h_drop, FC_W, FC_b)

        self.loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.label))
        self.prediction = tf.argmax(tf.nn.softmax(self.logits), 1)
        correct_predictions = tf.equal(self.prediction, tf.argmax(self.label, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
        
 
        # PREDICT CONCEPT VECTOR 
        self.concept_vect =tf.nn.embedding_lookup(self.W_concept,self.prediction )
        self.targets = (self.em)
        normalize_b = tf.nn.l2_normalize(self.concept_vect,1)        
        normalize_a = tf.nn.l2_normalize(self.targets,1)
        self.cos_similarity=tf.reduce_mean(((1-tf.reduce_sum(tf.multiply(normalize_a,normalize_b),1))))
        
        

        q = self.W_concept
        # PAIRWISE DISTANCE SIMILAIRTY/DISTANCE
        r = tf.reduce_sum(q*q, 1)

        # turn r into column vector
        r = tf.reshape(r, [-1, 1])
        self.pairwice_concept = r - 2*tf.matmul(q, tf.transpose(q)) + tf.transpose(r)
        
        loss_to_minimize = self.loss+ 16201*self.cos_similarity - 182020*tf.reduce_sum((self.pairwice_concept))
        self.target_loss = loss_to_minimize
        tvars = tf.trainable_variables()
        gradients = tf.gradients(loss_to_minimize, tvars, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)
        grads, global_norm = tf.clip_by_global_norm(gradients, 1.0)

        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = self.optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step,
                                                       name='train_step')
        print("graph built successfully!")




BATCH_SIZE = 1024
config = {
    "max_len": 20,
    "hidden_size": 128,
    "vocab_size": 59706,
    "embedding_size": 128,
    "n_class": 4,
    "learning_rate": 1e-3,
    "batch_size": BATCH_SIZE,
    "train_epoch": 20
}

classifier = ABLSTM(config)
classifier.build_graph()





