import tensorflow as tf
import numpy as np
from .baseModel import BaseModel
from .jacnn import Model
from sklearn.metrics import precision_recall_fscore_support
from utilities.utils import evalReport
import datetime
import random
import sys


class JACNNModel(BaseModel):
    def __init__(self,config):
        BaseModel.__init__(self,config)

    def buildModel(self):
        self.model = Model(self.config,self.vocab.wordVectors)
        self.model.build()
        self.loss = self.model.loss
        self.add_optimizer()
        self.initialize_session()
        self.add_summary()

    def train_step(self,batch,e):
        """
        A single training step
        """
        position = [[[i for i in range(batch['input_words'].shape[2])] for _ in range(batch['input_words'].shape[1])]
                    for _ in range(len(batch['input_words']))]
        position = np.array(position)
        batch_words = batch['input_words']
        #print(batch_x[0])
        batch_tag = batch['input_tag']
        #print(batch_tag[0])

        batch_doc_actual_length=batch['input_doc_actual_length']

        feed_dict = {
            self.model.input_words: batch_words,
            self.model.input_tag: batch_tag,
            self.model.doc_actual_length:batch_doc_actual_length,
            self.model.input_relative_position: position,
            self.model.cnn_dropout_keep_prob: self.config.cnn_dropout_keep_prob
        }
        for cls_name in self.config.cls_names:
            feed_dict[self.model.input_cls[cls_name]] = batch[cls_name]

        #print(tf.shape(self.model.doc_input).eval(feed_dict,self.sess))
        #print(tf.shape(self.model.word_features[1]).eval(feed_dict,self.sess))
        #print(self.model.cnn_drop_shape.eval(feed_dict,self.sess))
        #print(tf.shape(self.model.doc_features).eval(feed_dict,self.sess))

        _, step, summaries, loss = self.sess.run(
            [self.train_op, self.global_step, self.train_summary_op, self.loss],
            feed_dict)
        time_str = datetime.datetime.now().isoformat()
        print("\rtraining:{}:epoch {} step {}, loss {:g}".format(time_str, e, step, loss))
        self.train_summary_writer.add_summary(summaries, step)
        return loss

    def eval_step(self,data):
        """
        Evaluates model on a eval set
        """

        doc_cls_predictions = {}
        for cls_name in self.config.cls_names:
            doc_cls_predictions[cls_name] = []
        tag_prediction=[]
        # y_prediction = np.array([])
        num_of_batches = int((len(data['input_words']) - 1) / self.config.batch_size) + 1
        for batch_num in range(num_of_batches):
            start_index = batch_num * self.config.batch_size
            end_index = min((batch_num + 1) * self.config.batch_size, len(data['input_words']))
            batch_words = data['input_words'][start_index:end_index]
            batch_tag = data['input_tag'][start_index:end_index]
            batch_doc_actual_length = data['input_doc_actual_length'][start_index:end_index]
            position = [[[i for i in range(data['input_words'].shape[2])] for _ in range(data['input_words'].shape[1])]
                        for _ in range(len(batch_words))]
            position = np.array(position)


            feed_dict = {
                self.model.input_words: batch_words,
                self.model.input_tag: batch_tag,
                self.model.doc_actual_length: batch_doc_actual_length,
                self.model.input_relative_position: position,
                self.model.cnn_dropout_keep_prob: 1,
            }
            for cls_name in self.config.cls_names:
                feed_dict[self.model.input_cls[cls_name]] =  data[cls_name][start_index:end_index]


            step, summaries, doc_cls_pred,tag_pred = self.sess.run(
                [self.global_step, self.dev_summary_op, self.model.doc_cls_predictions, self.model.word_tag_predictions],
                feed_dict)
            # print type(y_pred)
            for cls_name in self.config.cls_names:
                doc_cls_predictions[cls_name].extend(doc_cls_pred[cls_name])            #print('doc',doc_cls_pred)

            tag_prediction.extend(tag_pred)
            #print('tag',tag_pred)
            #print(len(tag_prediction))
            #print(type(tag_prediction[0]))
        doc_precision, doc_recall, doc_f1_score = 0,0,0
        for cls_name in self.config.cls_names:
            #print(cls_name)
            #print(len(data[cls_name]),len(data['input_doc']))
            p, r, f, status = precision_recall_fscore_support(data[cls_name], np.array(doc_cls_predictions[cls_name]),
                                                                                labels=range(0, self.config.num_classes[cls_name]),
                                                                                pos_label=None,
                                                                                average='micro')
            doc_precision += p
            doc_recall += r
            doc_f1_score += f

        tags = []
        for i, t in enumerate(data['input_tag']):
            for j in range(data['input_doc_actual_length'][i]):
                tags.append(t[j])
        pred_tags = []
        for tag in tag_prediction:
            pred_tags.append(tag)

        #print(len(pred_tags))
        tag_precision, tag_recall, tag_f1_score, status = precision_recall_fscore_support(tags,np.array(pred_tags),
                                                                              labels=range(0, self.config.num_tags),
                                                                              pos_label=None,
                                                                              average='micro')
        print('tag 0',tag_precision, tag_recall, tag_f1_score)

        tag_precision, tag_recall, tag_f1_score, status = precision_recall_fscore_support(tags,np.array(pred_tags),
                                                                              labels=range(1, self.config.num_tags),
                                                                              pos_label=None,
                                                                              average='micro')

        print('tag 1',tag_precision, tag_recall, tag_f1_score)

        if len(self.config.cls_names) > 0:
            tmp = len(self.config.cls_names)
        else:
            tmp = 1
        print('doc',doc_precision / tmp , doc_recall / tmp, doc_f1_score /tmp)
        return doc_precision, doc_recall, doc_f1_score,tag_precision, tag_recall, tag_f1_score

    def saveModel(self, export_path):
        print('Exporting trained model to', export_path)
        if self.builder == None:
            savedModel_path = export_path + datetime.datetime.now().isoformat()
            self.config.model_path = savedModel_path
            self.builder = tf.saved_model.builder.SavedModelBuilder(savedModel_path)
            '''
            input_tensor_infos = {}
            for tensor in input_tensors:
                input_tensor_infos[tensor.name](tf.saved_model.utils.build_tensor_info(tensor))
            output_tensor_infos = {}
            for tensor in output_tensors:
                output_tensor_infos.append(tf.saved_model.utils.build_tensor_info(tensor))
            '''
            input_words_tensor_info = tf.saved_model.utils.build_tensor_info(self.model.input_words)
            input_relative_position_tensor_info = tf.saved_model.utils.build_tensor_info(self.model.input_relative_position)
            input_cnn_dropout_keep_prob_tensor_info = tf.saved_model.utils.build_tensor_info(self.model.cnn_dropout_keep_prob)

            word_tag_prediction_tensor_info = tf.saved_model.utils.build_tensor_info(self.model.word_tag_predictions)
            outputs = {
                             'word_tag_predictions':word_tag_prediction_tensor_info
                             }
            for cls_name in self.config.cls_names:
                outputs[cls_name + '_prediction'] = tf.saved_model.utils.build_tensor_info(self.model.doc_cls_predictions[cls_name])

            prediction_signature = (
                tf.saved_model.signature_def_utils.build_signature_def(
                    inputs={'input_words': input_words_tensor_info,'input_relative_position': input_relative_position_tensor_info,
                            'cnn_dropout_keep_prob' : input_cnn_dropout_keep_prob_tensor_info},
                    outputs=outputs,
                    method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

            #legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
            self.builder.add_meta_graph_and_variables(
                self.sess, [tf.saved_model.tag_constants.SERVING],
                signature_def_map={
                    'predict_event':
                        prediction_signature,
                })
                #legacy_init_op=legacy_init_op)

        self.builder.save()
        self.builder = None

        print('Done exporting!')

    def eval(self,test_data, config):
        doc_cls_predictions,tag_prediction=self.predict(test_data,config,config.model_path)

        doc_cls_reports = {}
        for cls_name in doc_cls_predictions:

            doc_cls_reports[cls_name] = evalReport(test_data[cls_name], doc_cls_predictions[cls_name],
                                                   config, config.num_classes[cls_name])
        tags = []
        for i, t in enumerate(test_data['input_tag']):
            for j in range(test_data['input_doc_actual_length'][i]):
                tags.append(t[j])
        pred_tags = []

        for tag in tag_prediction:
            pred_tags.append(tag)

        tag_report = evalReport(tags, pred_tags, config, config.num_tags)

        return doc_cls_reports, tag_report
    @staticmethod
    def predict(test_data,config,model_path):
        graph = tf.Graph()
        with graph.as_default():
            session_conf = tf.ConfigProto(
                allow_soft_placement=config.allow_soft_placement,
                log_device_placement=config.log_device_placement)
            sess = tf.Session(config=session_conf)
            with sess.as_default():
                tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], model_path)
                input_words = graph.get_tensor_by_name("input_words:0")
                input_relative_position = graph.get_tensor_by_name("input_relative_position:0")
                doc_actual_length = graph.get_tensor_by_name("doc_actual_length:0")

                cnn_dropout_keep_prob = graph.get_tensor_by_name("cnn_dropout_keep_prob:0")

                doc_cls_predictions = {}
                for cls_name in config.cls_names:
                    doc_cls_predictions[cls_name] = graph.get_operation_by_name(cls_name+"_doc_output/predictions").outputs[0]

                word_tag_predictions = graph.get_operation_by_name("tag_output/predictions/GatherV2").outputs[0]

                batch_size = 100
                cls_predictions = {}
                for cls_name in config.cls_names :
                    cls_predictions[cls_name] = []
                tag_prediction=[]
                # y_prediction = np.array([])
                num_of_batches = int((len(test_data['input_words']) - 1) / batch_size) + 1
                for batch_num in range(num_of_batches):
                    start_index = batch_num * batch_size
                    end_index = min((batch_num + 1) * batch_size, len(test_data['input_words']))
                    batch_words = test_data['input_words'][start_index:end_index]
                    batch_doc_actual_length = test_data['input_doc_actual_length'][start_index:end_index]

                    position = [[[i for i in range(test_data['input_words'].shape[2])] for _ in range(test_data['input_words'].shape[1])]
                        for _ in range(len(batch_words))]
                    position = np.array(position)
                    doc_cls_pred,tag_pred= sess.run([doc_cls_predictions,word_tag_predictions], {input_words: batch_words,
                                                               doc_actual_length:batch_doc_actual_length,
                                                               input_relative_position: position,
                                                               cnn_dropout_keep_prob: 1.0})

                    # print type(y_pred)
                    for cls_name in config.cls_names:
                        cls_predictions[cls_name].extend(doc_cls_pred[cls_name])
                    tag_prediction.extend(tag_pred)
                sess.close()
        tf.reset_default_graph()
        return cls_predictions,tag_prediction

    @staticmethod
    def outputEmbeddings(test_data, config, model_path):
        graph = tf.Graph()
        with graph.as_default():
            session_conf = tf.ConfigProto(
                allow_soft_placement=config.allow_soft_placement,
                log_device_placement=config.log_device_placement)
            sess = tf.Session(config=session_conf)
            with sess.as_default():
                tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], model_path)
                input_words = graph.get_tensor_by_name("input_words:0")
                input_relative_position = graph.get_tensor_by_name("input_relative_position:0")
                doc_actual_length = graph.get_tensor_by_name("doc_actual_length:0")

                cnn_dropout_keep_prob = graph.get_tensor_by_name("cnn_dropout_keep_prob:0")

                word_representations = []
                for i in range(config.doc_max_length):
                    if i == 0:
                        word_representations.append(graph.get_tensor_by_name("cnn-dropout-tag/dropout/mul:0"))
                    else:
                        word_representations.append(graph.get_tensor_by_name("cnn-dropout-tag_%d/dropout/mul:0" % i))

                batch_size = 50

                doc_word_predictions = []
                # y_prediction = np.array([])
                num_of_batches = int((len(test_data['input_words']) - 1) / batch_size) + 1
                for batch_num in range(num_of_batches):
                    start_index = batch_num * batch_size
                    end_index = min((batch_num + 1) * batch_size, len(test_data['input_words']))
                    batch_words = test_data['input_words'][start_index:end_index]
                    batch_doc_actual_length = test_data['input_doc_actual_length'][start_index:end_index]

                    position = [[[i for i in range(test_data['input_words'].shape[2])] for _ in
                                 range(test_data['input_words'].shape[1])]
                                for _ in range(len(batch_words))]
                    position = np.array(position)
                    batch_word_representations = sess.run(word_representations,
                                                          {input_words: batch_words,
                                                           doc_actual_length: batch_doc_actual_length,
                                                           input_relative_position: position,
                                                           cnn_dropout_keep_prob: 1.0})

                    batch_word_representations = np.stack(batch_word_representations, axis=1)
                    print(batch_word_representations.shape)

                    doc_word_predictions.extend(batch_word_representations.tolist())
        return doc_word_predictions




