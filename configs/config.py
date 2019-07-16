class Config(object):
    def __init__(self):
        self.model ='JCNN'
        self.task = ['cls','extraction']
        self.l2_reg_lambda = 0.001
        self.included_tags = [1,2,3,4]
        self.num_tags = 5
        self.doc_max_length = 100
        self.num_epochs = 1
        self.min_epochs = 100
        self.batch_size = 50
        self.word_embedding_size = 100
        self.position_embedding_size=10
        self.tag_embedding_size = 5
        self.window_size = 5
        self.filter_sizes = [1,2,3,4]
        self.num_filters = 50
        self.num_rnn_units = 50
        self.attention_size = 50
        self.cnn_dropout_keep_prob = 0.5
        self.rnn_dropout_keep_prob = 0.75
        self.attention_keep_prob = 0.5
        self.tag_loss_weight = 1
        self.max_num_words = 0

        self.trainIdFile = ""
        self.devIdFile = ""
        self.testIdFile = ""
        self.non_annotated_id_file = ""
        self.multiLabelIdFile =""
        self.extraction_file = ""
        self.classification_file = ""
        self.forms_file = ""
        self.test_tags_path = ""
        self.predicted_tag_path = ""
        self.form_names=['commenting','ogling','touching']
        self.cls_names = ['harasser_age','harasser_num','harasser_type','location_type','time_of_day']
        self.num_classes = {'harasser_age':3,'harasser_num':3,'harasser_type':10,'location_type':14,'time_of_day':3,'commenting':2,'ogling':2,'touching':2}
        self.tag_value_map = {'harasser_age':1,'harasser_num':1,'harasser_type':1,'location_type':3,'time_of_day':2,'commenting':4,'ogling':4,'touching':4}
        self.categories =['harasser_type','location_type']
        self.output_prefix ='jcnn'
        self.exp_path= ""
        self.train_path =self.exp_path + self.output_prefix + '/'
        self.vocab_path = self.train_path + "vocabulary_" + self.output_prefix + '.bin'
        self.model_path = ""
        self.pre_trained_wordVectors_path = ""
        self.config_path = self.train_path + 'savedConfigs-' + self.output_prefix + '.bin'
        self.evaluate_every = 1
        self.allow_soft_placement = True
        self.log_device_placement = False
        self.delimiters = str('?!.\n')

    def hyperParamToStr(self):
        return "model name: " + str(self.model) + '\n' # add other parameters that need to be recorded
