import tensorflow as tf
from .attention import attention
class Model(object):
    def __init__(self, config, vectors):
        self.doc_max_length = config.doc_max_length
        self.sequence_length = config.window_size * 2 +1
        self.num_classes = config.num_classes
        self.task = config.task

        self.vectors = vectors
        self.word_embedding_size = config.word_embedding_size
        self.position_embedding_size = config.position_embedding_size
        self.filter_sizes = config.filter_sizes
        self.num_filters = config.num_filters
        self.attention_size = config.attention_size

        self.tag_loss_weight = config.tag_loss_weight
        self.l2_reg_lambda = config.l2_reg_lambda
        self.input_words = tf.placeholder(tf.int32, [None, self.doc_max_length, self.sequence_length], name='input_words')
        self.doc_actual_length = tf.placeholder(tf.int32, [None], name='doc_actual_length')

        self.input_relative_position = tf.placeholder(tf.int32, [None, self.doc_max_length, self.sequence_length], name='input_relative_position')
        self.input_tag = tf.placeholder(tf.int32, [None, self.doc_max_length], name='input_tag')
        self.cls_names = config.cls_names

        self.input_cls ={}
        self.included_tags = config.included_tags
        self.num_tags = len(self.included_tags) + 1

        for cls_name in self.cls_names:
            self.input_cls[cls_name] = tf.placeholder(tf.int32, [None], name='input_'+cls_name)

        self.cnn_dropout_keep_prob = tf.placeholder(tf.float32, name='cnn_dropout_keep_prob')

        self.l2_loss = tf.constant(0.0)

    def add_embedding_layer(self):


        # Embedding layer
        initial = tf.constant(self.vectors, dtype=tf.float32)
        with tf.name_scope('embedding'):
            wordVectors = tf.get_variable('word_vectors', initializer=initial,trainable=True)
            self.cnn_embedded_words = tf.nn.embedding_lookup(wordVectors, self.input_words)
            positionVectors = tf.get_variable(name='W', initializer=tf.random_uniform([self.sequence_length,
                                                             self.position_embedding_size], -1.0, 1.0))

            self.embedded_position = tf.nn.embedding_lookup(positionVectors, self.input_relative_position )
            print(self.cnn_embedded_words.get_shape())
            print(self.embedded_position.get_shape())
            self.concat_word_pos_embedded = tf.concat(
                [self.cnn_embedded_words, self.embedded_position], 3)
            self.cnn_concat_word_pos_embedded_expanded = tf.expand_dims(self.concat_word_pos_embedded, -1)

    def add_cnn_layer(self, input, length,filter_sizes,embedding_size,scope):
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.variable_scope('%s-conv-maxpool-%s' % (scope,filter_size), reuse=tf.AUTO_REUSE):
                filter_shape = [filter_size, embedding_size , 1,
                                self.num_filters]
                W = tf.get_variable(name='W',initializer = tf.truncated_normal(filter_shape, stddev=0.1))
                b = tf.get_variable(name='b',initializer = tf.constant(0.1, shape=[self.num_filters]))
                conv = tf.nn.conv2d(
                    input,
                    W,
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name='conv')

                h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')

                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name='pool')
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_features = self.num_filters * len(self.filter_sizes)
        h_pool = tf.concat(pooled_outputs, 3)
        h_pool_flat = tf.reshape(h_pool, [-1, num_features])

        # Add dropout
        with tf.variable_scope('cnn-dropout-%s'%scope):
            cnn_drop = tf.nn.dropout(h_pool_flat, self.cnn_dropout_keep_prob)
        return cnn_drop
    def add_cnn_attn_layer(self, input, length,filter_sizes,embedding_size,scope):
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.variable_scope('%s-conv-maxpool-%s' % (scope,filter_size), reuse=tf.AUTO_REUSE):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size , 1,
                                self.num_filters]
                W = tf.get_variable(name='W',initializer = tf.truncated_normal(filter_shape, stddev=0.1))
                b = tf.get_variable(name='b',initializer = tf.constant(0.1, shape=[self.num_filters]))
                conv = tf.nn.conv2d(
                    input,
                    W,
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name='conv')

                h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
                print(h, h.get_shape())
                h = tf.reshape(h,[-1,length - filter_size + 1,self.num_filters])

                pooled,alphas=attention(h, self.attention_size, return_alphas=True)

                pooled_outputs.append(pooled)

        # Combine all the pooled features
        h_pool = tf.concat(pooled_outputs, 1)

        # Add dropout
        with tf.variable_scope('cnn-dropout-%s'%scope):
            cnn_drop = tf.nn.dropout(h_pool, self.cnn_dropout_keep_prob)
            print(cnn_drop)
        return cnn_drop
    def tag_cnn_features(self):
        self.word_features = []
        for i in range(self.doc_max_length):
            input = self.cnn_concat_word_pos_embedded_expanded[:,i,:,:,:]#tf.slice(self.cnn_concat_word_pos_embedded_expanded, [0,i,0,0,0],[0,1,self.sequence_length,self.word_embedding_size,1])
            self.tag_input_shape = tf.shape(self.cnn_concat_word_pos_embedded_expanded[:,i,:,:,:])
            cnn_drop = self.add_cnn_layer(input,self.sequence_length,self.filter_sizes,self.word_embedding_size + self.position_embedding_size,'tag' )
            self.cnn_drop_shape=tf.shape(cnn_drop)
            self.word_features.append(cnn_drop)


    def add_fc_layer(self,input,input_size,num_classes,scope):
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
            word_tag_prediction, word_tag_score,W,b = self.add_fc_layer(self.word_features[i],self.num_filters * len(self.filter_sizes),self.num_tags,"tag_fc")
            self.word_tag_predictions.append(word_tag_prediction)
            self.word_tag_scores.append(word_tag_score)
        self.l2_loss += tf.nn.l2_loss(W)
        self.l2_loss += tf.nn.l2_loss(b)


        self.mask = tf.sequence_mask(self.doc_actual_length, self.doc_max_length)
        print('mask',self.mask)
        with tf.name_scope("tag_output"):
            self.word_tag_predictions = tf.stack(self.word_tag_predictions,1)
            self.word_tag_predictions=tf.boolean_mask(self.word_tag_predictions,self.mask,name = 'predictions',axis=0)
            self.word_tag_scores = tf.stack(self.word_tag_scores,1)
            self.word_tag_scores=tf.boolean_mask(self.word_tag_scores,self.mask,axis=0)
            self.word_tag_scores =tf.identity(self.word_tag_scores,name='scores')
    def doc_prediction(self):
        self.doc_input = tf.expand_dims(tf.stack(self.word_features, axis=1), -1)
        self.doc_cls_predictions = {}
        self.doc_cls_scores = {}
        for cls_name in self.cls_names:

            doc_features = self.add_cnn_attn_layer(self.doc_input, self.doc_max_length, self.filter_sizes,
                                              self.num_filters * len(self.filter_sizes), cls_name)

            self.doc_cls_predictions[cls_name], self.doc_cls_scores[cls_name],W,b = self.add_fc_layer(doc_features,
                                                                              self.num_filters * len(self.filter_sizes),
                                                                              self.num_classes[cls_name],cls_name+"_doc_output")
            self.l2_loss += tf.nn.l2_loss(W)
            self.l2_loss += tf.nn.l2_loss(b)

    def add_loss(self):
        with tf.name_scope('loss'):
            self.tag_losses = tf.constant(0.0)
            if 'extraction' in self.task:
                if len(self.included_tags) == 1:
                    tags = tf.where(self.input_tag == self.included_tags[0], tf.ones_like(self.input_tag, dtype=tf.int32),
                         tf.zeros_like(self.input_tag, dtype=tf.int32))
                else:
                    tags = self.input_tag
                tags = tf.boolean_mask(tags,self.mask,axis=0)
                tags = tf.contrib.layers.one_hot_encoding(tags,num_classes=self.num_tags)
                if len(self.included_tags) == 1:
                    weights = tf.constant([0.2, 0.8])

                    tags = tags * weights
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
        self.tag_cnn_features()
        self.tag_prediction()
        self.doc_prediction()
        self.add_loss()
