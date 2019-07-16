import numpy as np
import tensorflow as tf
import time,os
from utilities.utils import data_iterator

class BaseModel(object):
    """Generic class for general methods that are not specific to NER"""

    def __init__(self, config):
        self.config = config
        self.sess   = None
        self.loss = None
        self.saver  = None
        self.train_summary_list = []
        self.valid_summary_list = []
        self.model = None
        self.vocab = None
        self.builder = None
        self.tag_weights = []


        self.timestamp = str(int(time.time()))
        self.out_dir = os.path.abspath(os.path.join(config.train_path, "runs-" + config.output_prefix, self.timestamp))
        self.checkpoint_dir = os.path.abspath(os.path.join(self.out_dir, "checkpoints"))
        self.checkpoint_path= os.path.join(self.checkpoint_dir, "model")
        self.final_prefix = os.path.join(self.checkpoint_dir, "final")

        self.best_f_dev_cls = 0
        self.best_f_dev_tag = 0

        self.best_f_dev_cls = 0
        self.best_p_dev_cls = 0
        self.best_r_dev_cls = 0
        self.best_f_dev_tag = 0
        self.best_p_dev_tag = 0
        self.best_r_dev_tag = 0

    def addVocab(self,vocab):
        self.vocab = vocab

    def initialize_session(self):
        session_conf = tf.ConfigProto(
            allow_soft_placement=self.config.allow_soft_placement,
          log_device_placement=self.config.log_device_placement)
        session_conf.gpu_options.allow_growth=True
        if self.sess is None:
            self.sess = tf.Session(config=session_conf)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.saver = tf.train.Saver()
        print("Writing to {}\n".format(self.out_dir))
        self.sess.run(tf.global_variables_initializer())
    def end_session(self):
        self.sess.close()
    def add_optimizer(self):
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.optimizer = tf.train.AdadeltaOptimizer(learning_rate=0.1, rho=0.95, epsilon=1e-6)
        self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
        self.train_op = self.optimizer.apply_gradients(self.grads_and_vars, global_step=self.global_step)

    def train(self,trainData,devData):
        self.run_epoch(trainData,devData)

    def train_step(self,data,e):
        pass
    def eval_step(self,data):
        pass
    def saveModel(selfs,export_path):
        pass

    def add_summary(self):
        self.grad_summaries = []
        for g, v in self.grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                self.grad_summaries.append(grad_hist_summary)
                self.grad_summaries.append(sparsity_summary)
        self.grad_summaries_merged = tf.summary.merge(self.grad_summaries)
        self.loss_summary = tf.summary.scalar("loss", self.loss)

        # Train Summaries
        self.train_summary_op = tf.summary.merge([self.loss_summary, self.grad_summaries_merged])
        self.train_summary_dir = os.path.join(self.out_dir, "summaries", "train")
        self.train_summary_writer = tf.summary.FileWriter(self.train_summary_dir, self.sess.graph)

        # Dev summaries
        self.dev_summary_op = tf.summary.merge([self.loss_summary])
        self.dev_summary_dir = os.path.join(self.out_dir, "summaries", "dev")
        self.dev_summary_writer = tf.summary.FileWriter(self.dev_summary_dir, self.sess.graph)



    def run_epoch(self, trainData, devData):
        for e in np.arange(self.config.num_epochs):
            sum_loss = 0
            for step, trainBatch in enumerate(data_iterator(
                    trainData, batch_size=self.config.batch_size, shuffle=True, alpha=self.config.alpha)):
                sum_loss += self.train_step(trainBatch, e)

                current_step = tf.train.global_step(self.sess, self.global_step)
            if e % self.config.evaluate_every == 0:
                print("\nDevolope:")
                p_dev_cls, r_dev_cls, f_dev_cls, p_dev_tag, r_dev_tag, f_dev_tag = self.eval_step(devData)

                if self.best_f_dev_cls + self.best_f_dev_tag < f_dev_cls + f_dev_tag:
                    self.best_f_dev_cls = f_dev_cls
                    self.best_p_dev_cls = p_dev_cls
                    self.best_r_dev_cls = r_dev_cls
                    self.best_f_dev_tag = f_dev_tag
                    self.best_p_dev_tag = p_dev_tag
                    self.best_r_dev_tag = r_dev_tag
                    self.saveModel(self.checkpoint_path)

        print("\nDevelope:")
        p_dev_cls, r_dev_cls, f_dev_cls, p_dev_tag, r_dev_tag, f_dev_tag = self.eval_step(devData)

        if self.best_f_dev_cls + self.best_f_dev_tag < f_dev_cls + f_dev_tag:
            self.best_f_dev_cls = f_dev_cls
            self.best_p_dev_cls = p_dev_cls
            self.best_r_dev_cls = r_dev_cls
            self.best_f_dev_tag = f_dev_tag
            self.best_p_dev_tag = p_dev_tag
            self.best_r_dev_tag = r_dev_tag
            self.saveModel(self.checkpoint_path)




