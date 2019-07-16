import tensorflow as tf
from .attention import attention
class Model(object):
    def __init__(self, config,vectors):
        self.doc_max_length = config.doc_max_length
        self.sequence_length = config.window_size * 2 +1
        self.num_classes = config.num_classes
        self.num_tags = config.num_tags
        self.tag_value_map = config.tag_value_map
        self.task = config.task

        self.vectors = vectors
        self.word_embedding_size = config.word_embedding_size
        self.position_embedding_size = config.position_embedding_size
        self.filter_sizes = config.filter_sizes
        self.num_filters = config.num_filters
        self.num_rnn_units = config.num_rnn_units
        self.tag_loss_weight = config.tag_loss_weight
        self.l2_reg_lambda = config.l2_reg_lambda
        self.input_doc = tf.placeholder(tf.int32, [None, self.doc_max_length], name='input_doc')
        self.doc_actual_length = tf.placeholder(tf.int32, [None], name='doc_actual_length')
        self.attention_size = config.attention_size
        self.attention_keep_prob = tf.placeholder(tf.float32, name='attention_keep_prob')
        self.input_tag = tf.placeholder(tf.int32, [None, self.doc_max_length], name='input_tag')
        self.cls_names = config.cls_names
        self.input_cls ={}
        for cls_name in self.cls_names:
            self.input_cls[cls_name] = tf.placeholder(tf.int32, [None], name='input_'+cls_name)
        self.rnn_output_keep_prob = tf.placeholder(tf.float32, name='rnn_output_keep_prob')
        # Keeping track of l2 regularization loss (optional)
        self.attention_loss = 0
        self.l2_loss = tf.constant(0.0)

    def add_embedding_layer(self):
        # Embedding layer
        initial = tf.constant(self.vectors, dtype=tf.float32)
        with tf.name_scope('embedding'):
            wordVectors = tf.get_variable('word_vectors', initializer=initial,trainable=True)
            self.embedded_words = tf.nn.embedding_lookup(wordVectors, self.input_doc)

    def add_bilstm_layer(self):
        lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(self.num_rnn_units, forget_bias=1.0)
        lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_fw_cell, output_keep_prob=self.rnn_output_keep_prob)
        lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(self.num_rnn_units, forget_bias=1.0)
        lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_bw_cell, output_keep_prob=self.rnn_output_keep_prob)

        self.bilstm_outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, self.embedded_words,
                                                     self.doc_actual_length, dtype=tf.float32)

        self.word_features = tf.concat(self.bilstm_outputs,2)
    def add_attention_layer(self,scope,tag_value):
        with tf.name_scope(scope):
            attention_output, alphas = attention(self.bilstm_outputs, self.attention_size, return_alphas=True)
            print(attention_output)
            print(alphas)
            tf.summary.histogram('alphas', alphas)
            input_tag_selected = tf.where(self.input_tag == tag_value, tf.ones_like(self.input_tag),tf.zeros_like(self.input_tag))
            self.attention_loss += tf.reduce_mean(tf.reduce_sum(tf.square(tf.subtract(tf.cast(input_tag_selected, tf.float32), alphas)), axis=1))

            attention_sent_output_drop = tf.nn.dropout(attention_output, self.attention_keep_prob)
        return attention_sent_output_drop


    def add_fc_layer(self,input,input_size,num_classes,scope):

        #self.feature = self.cnn_drop
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            W = tf.get_variable(
                'W',
                shape=[input_size, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable(name='b',initializer=tf.constant(0.1, shape=[num_classes]))

            scores = tf.nn.xw_plus_b(input, W, b, name='scores')
            predictions = tf.argmax(scores, 1, name='predictions')
        return predictions, scores, W,b

    def tag_prediction(self):
        self.word_tag_predictions =[]
        self.word_tag_scores =[]
        W = None
        b = None
        for i in range(self.doc_max_length):
            word_tag_prediction, word_tag_score,W,b = self.add_fc_layer(self.word_features[:,i,:]
                                                                        ,2 * self.num_rnn_units,self.num_tags,"tag_fc")
            self.word_tag_predictions.append(word_tag_prediction)
            self.word_tag_scores.append(word_tag_score)
        self.l2_loss += tf.nn.l2_loss(W)
        self.l2_loss += tf.nn.l2_loss(b)


        self.mask = tf.sequence_mask(self.doc_actual_length, self.doc_max_length)
        print('mask',self.mask)
        with tf.name_scope("tag_output"):
            self.word_tag_predictions = tf.stack(self.word_tag_predictions,1)

            self.word_tag_predictions=tf.boolean_mask(self.word_tag_predictions,self.mask,name = 'predictions',axis=0)

            print(self.word_tag_predictions)
            self.word_tag_scores = tf.stack(self.word_tag_scores,1)

            self.word_tag_scores=tf.boolean_mask(self.word_tag_scores,self.mask,axis=0)
            self.word_tag_scores =tf.identity(self.word_tag_scores,name='scores')
    def doc_prediction(self):
        self.doc_cls_predictions = {}
        self.doc_cls_scores = {}
        for cls_name in self.cls_names:
            tag_value = self.tag_value_map[cls_name]
            attention_sent_output_drop = self.add_attention_layer(cls_name + '_attention_layer',tag_value)
            print(attention_sent_output_drop)

            self.doc_cls_predictions[cls_name], self.doc_cls_scores[cls_name],W,b = self.add_fc_layer(attention_sent_output_drop,
                                                                                2 * self.num_rnn_units,
                                                                            self.num_classes[cls_name],cls_name+"_doc_output")
            self.l2_loss += tf.nn.l2_loss(W)
            self.l2_loss += tf.nn.l2_loss(b)
    def add_loss(self):
        with tf.name_scope('loss'):
            self.tag_losses = tf.constant(0.0)
            if 'extraction' in self.task:
                tags = tf.boolean_mask(self.input_tag,self.mask,axis=0)
                tags = tf.contrib.layers.one_hot_encoding(tags,num_classes=self.num_tags)
                tag_losses = tf.nn.softmax_cross_entropy_with_logits(labels=tags,
                                                                     logits=self.word_tag_scores)
                self.tag_losses = tf.reduce_mean(tag_losses)
            self.doc_losses = tf.constant(0.0)
            if 'cls' in  self.task:
                doc_losses = tf.constant(0.0)
                for cls_name in self.cls_names:
                    cls = tf.contrib.layers.one_hot_encoding(self.input_cls[cls_name],num_classes=self.num_classes[cls_name])
                    doc_losses += tf.nn.softmax_cross_entropy_with_logits(labels=cls,logits=self.doc_cls_scores[cls_name])
                self.doc_losses = tf.reduce_mean(doc_losses)

            self.loss = self.tag_losses * self.tag_loss_weight + self.doc_losses + self.l2_reg_lambda * self.l2_loss


    def build(self):
        self.add_embedding_layer()
        self.add_bilstm_layer()
        self.tag_prediction()
        self.doc_prediction()
        self.add_loss()
