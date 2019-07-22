"""
    python
"""
import argparse
import logging
import sys
import os
import tensorflow as tf
from tensorflow.python.ops import summary_ops_v2
import time
import numpy as np


def get_batches():
    data = list()
    """
    :param int_text:
    :param batch_size:
    :param seq_length:
    :return:x, y
    """
    return np.array(data)


class LSTMRNNNetwork(object):

    def __init__(self, batch_size, rnn_size, seq_length, cell_size):

        self.batch_size = batch_size
        self.cell_size = cell_size
        self.rnn_size = rnn_size

        self.train_batches = get_batches()
        self.losses = {'train': [], 'test': []}

        self.model = tf.keras.Sequential([
            # tf.keras.layers.Embedding(self.vocab_size, self.embed_dim, batch_input_shape=[self.batch_size, None]),
            tf.keras.layers.LSTM(cell_size, return_sequences=True, stateful=True,
                                 recurrent_initializer='glorot_uniform'),
            tf.keras.layers.Dense(units=1, activation='linear'),
        ])
        self.model.summary()

        self.optimizer = tf.keras.optimizers.Adam()
        self.computeLoss = tf.keras.losses.sparse_categorical_crossentropy(from_logits=True)

        if tf.io.gfile.exists(MODEL_DIR):
            logger.info('Removing existing model dir: {}'.format(MODEL_DIR))
            tf.io.gfile.rmtree(MODEL_DIR)
        else:
            tf.io.gfile.makedirs(MODEL_DIR)

        train_dir = os.path.join(MODEL_DIR, 'summaries', 'train')

        self.train_summary_writer = summary_ops_v2.create_file_writer(train_dir, flush_millis=10000)

        checkpoint_dir = os.path.join(MODEL_DIR, 'checkpoints')
        self.checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
        self.checkpoint = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer)

        self.checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    @tf.function
    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            logits = self.model(x, training=True)
            loss = self.computeLoss(y, logits)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss, logits

    def train(self, epochs=1, log_freq=50):
        for i in range(epochs):
            with self.train_summary_writer.as_default():
                start = time.time()
                avg_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)
                for batch_i, (x, y) in enumerate(self.train_batches):
                    loss, logits = self.train_step(x, y)
                    avg_loss(loss)
                    self.losses['train'].append(loss)

                    if tf.equal(self.optimizer.iterations % log_freq, 0):
                        summary_ops_v2.scalar('loss', avg_loss.result(), step=self.optimizer.iterations)

                        rate = log_freq / (time.time() - start)
                        logger.info('Step #{}\tLoss: {:0.6f} ({} steps/sec)'.format(
                            self.optimizer.iterations.numpy(), loss, rate))

                        avg_loss.reset_states()
                        start = time.time()

            self.checkpoint.save(self.checkpoint_prefix)
            logger.info("save model\n")


if __name__ == '__main__':

    logging.basicConfig(
        format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s',
        level=logging.INFO
    )

    program = os.path.basename(sys.argv[0])
    if len(sys.argv) < 2:
        print(globals()['__doc__'] % locals())
        sys.exit(1)

    parser = argparse.ArgumentParser()
    parser.add_argument('-batch_size', help='批次数', default=400, type=int)
    parser.add_argument('-cell_size', help='隐藏层维度', default=1000, type=int)
    parser.add_argument('-seq_length', help='序列长度', default=15, type=int)
    parser.add_argument('-embed_dim', help='向量维度', default=256, type=int)
    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    MODEL_DIR = './model'

    # LSTMRNNNetwork(args.batch_size, args.cell_size, args.seq_length, args.embed_dim)

    logger.info("finished running %s", program)
