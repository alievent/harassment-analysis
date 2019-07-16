import tensorflow as tf
class Model(object):
    def __init__(self, config, vectors):
        self.doc_max_length = config.doc_max_length
        #self.sequence_length = config.window_size * 2 +1
        self.num_classes = config.num_classes
        self.num_tags = config.num_tags
        self.task = config.task

        self.vectors = vectors
        self.word_embedding_size = config.word_embedding_size
        self.position_embedding_size = config.position_embedding_size
        self.filter_sizes = config.filter_sizes
        self.num_filters = config.num_filters
        self.attention_size = config.attention_size

        self.tag_loss_weight = config.tag_loss_weight
        self.l2_reg_lambda = config.l2_reg_lambda
        self.input_doc = tf.placeholder(tf.int32, [None, self.doc_max_length], name='input_doc')

        #self.input_relative_position = tf.placeholder(tf.int32, [None, self.doc_max_length, self.sequence_length], name='input_relative_position')
        self.cls_names = config.cls_names
        self.input_cls ={}
        for cls_name in self.cls_names:
            self.input_cls[cls_name] = tf.placeholder(tf.int32, [None], name='input_'+cls_name)

        self.cnn_dropout_keep_prob = tf.placeholder(tf.float32, name='cnn_dropout_keep_prob')

        # Keeping track of l2 regularization loss (optional)
        self.l2_loss = tf.constant(0.0)

    def add_embedding_layer(self):


        # Embedding layer
        initial = tf.constant(self.vectors, dtype=tf.float32)
        with tf.name_scope('embedding'):
            wordVectors = tf.get_variable('word_vectors', initializer=initial,trainable=True)
            self.embedded_words = tf.nn.embedding_lookup(wordVectors, self.input_doc)
            print(self.embedded_words)

            self.cnn_embedded_expanded = tf.expand_dims(self.embedded_words, -1)
            #self.cnn_concat_word_pos_embedded_expanded = tf.expand_dims(self.cnn_embedded_words,-1)

    def add_cnn_layer(self, input, length,filter_sizes,embedding_size,scope):
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.variable_scope('%s-conv-maxpool-%s' % (scope,filter_size), reuse=tf.AUTO_REUSE):
                # Convolution Layer
                #filter_shape = [filter_size, self.word_embedding_size + self.position_embedding_size, 1, self.num_filters]
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
                #print(W)
                #print(b)
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')

                # Maxpooling over the outputs
                print(h)
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name='pool')
                pooled_outputs.append(pooled)
                print(pooled)
        # Combine all the pooled features
        num_features = self.num_filters * len(self.filter_sizes)
        h_pool = tf.concat(pooled_outputs, 3)
        h_pool_flat = tf.reshape(h_pool, [-1, num_features])

        # Add dropout
        with tf.variable_scope('cnn-dropout-%s'%scope):
            cnn_drop = tf.nn.dropout(h_pool_flat, self.cnn_dropout_keep_prob)
        return cnn_drop



    def add_fc_layer(self,input,input_size,num_classes,scope):

        #self.feature = self.cnn_drop
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            W = tf.get_variable(
                'W',
                shape=[input_size, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable(name='b',initializer=tf.constant(0.1, shape=[num_classes]))
            #print(W)
            #print(b)
            scores = tf.nn.xw_plus_b(input, W, b, name='scores')
            predictions = tf.argmax(scores, 1, name='predictions')
        return predictions, scores, W,b


    def doc_prediction(self):
        self.doc_input = self.cnn_embedded_expanded
        self.doc_cls_predictions = {}
        self.doc_cls_scores = {}
        for cls_name in self.cls_names:

            doc_features = self.add_cnn_layer(self.doc_input, self.doc_max_length, self.filter_sizes,
                                              self.word_embedding_size, cls_name)

            self.doc_cls_predictions[cls_name], self.doc_cls_scores[cls_name],W,b = self.add_fc_layer(doc_features,
                                                                              self.num_filters * len(self.filter_sizes),
                                                                              self.num_classes[cls_name],cls_name+"_doc_output")
            self.l2_loss += tf.nn.l2_loss(W)
            self.l2_loss += tf.nn.l2_loss(b)

        #print(self.doc_cls_scores)
        #print(self.doc_cls_scores)
    def add_loss(self):
        with tf.name_scope('loss'):

            doc_losses = tf.constant(0.0)
            for cls_name in self.cls_names:
                cls = tf.contrib.layers.one_hot_encoding(self.input_cls[cls_name],num_classes=self.num_classes[cls_name])
                doc_losses += tf.nn.softmax_cross_entropy_with_logits(labels=cls,logits=self.doc_cls_scores[cls_name])
            self.doc_losses = tf.reduce_mean(doc_losses)

            self.loss =self.doc_losses + self.l2_reg_lambda * self.l2_loss

    def build(self):
        self.add_embedding_layer()
        self.doc_prediction()
        self.add_loss()
