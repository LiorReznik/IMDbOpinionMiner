import numpy as np
import  tensorflow as tf
import os,urllib,tarfile,re,logging,pickle,sys
from datetime import datetime
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from collections import Counter
import json
class Preprocess:

    def __init__(self, path='http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz', archive_name='imdbrev.tar.gz',folder_name='aclImdb'):
        self.prep = {'X_train': [], 'Y_train': [], 'X_test': [], "Y_test": [],"vocab_size": [], "max_seq_len": 0}
        self.__start_logger
        self.__path=path
        self.__vocab=[]
        self.__archive_name = archive_name
        self.__folder_name = folder_name
        self.__preprocess
        print(self.prep['X_train'].shape,self.prep['X_test'].shape, self.prep['Y_train'].shape, self.prep['Y_test'].shape )




    @property
    def __preprocess(self):
        self.__download_reviews
        self.__prepare_reviews
        self.sent_to_seq

        self.__logger.info('end of the preprocessing')

    @property
    def __start_logger(self):
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.__logger = logging.getLogger('Preprocess')
        self.__logger.setLevel(logging.DEBUG)
        # create file handler which logs even debug messages
        fh = logging.FileHandler(datetime.now().strftime('Preprocess_%H_%M_%d_%m_%Y.log'))
        fh.setLevel(logging.DEBUG)
        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.ERROR)
        # create formatter and add it to the handlers
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # add the handlers to the logger
        self.__logger.addHandler(fh)
        self.__logger.addHandler(ch)

    @property
    def __download_reviews(self):
        """""
        function to download the reviews form stanfords servers
        """""
        if not os.path.exists(self.__archive_name):
            try:
                self.__logger.info("Downloading the reviews")
                self.__reviews_tar, _ = urllib.request.urlretrieve(self.__path, self.__archive_name)
            except urllib.error as e:
                self.__logger.exception(e.to_str())

            self.__logger.info("Download has completed")
        else:
            self.__logger.warning("Skipping Download,The Archive already in the HD")

    @property
    def __prepare_reviews(self):
        """""
        function to extract the reviews and sends them into preprocessing
        """""

        def extract():
            self.__logger.info("starting the extraction")
            with tarfile.open(self.__archive_name) as file:
                file.extractall()
            self.__logger.info("done extracting")

        if not os.path.exists(self.__folder_name):
             extract()
        else:
            self.__logger.warning("the archive is already in the disk")
        self.__get_and_prep(path='aclImdb/train/pos/', polarity=True, type='train')
        self.__get_and_prep(path='aclImdb/train/neg/', polarity=False, type='train')
        print(self.prep['X_train'][0])
        self.__get_and_prep(path='aclImdb/test/pos/', polarity=True, type='test')
        self.__get_and_prep(path='aclImdb/test/neg/', polarity=False, type='test')

        self.prep['Y_train'] = np.array(self.prep['Y_train'])
        self.prep['Y_test'] = np.array(self.prep['Y_test'])

    def __get_and_prep(self, path, polarity, type):

        def normalize():
            self.__logger.info('Normalizing the review')
            nonlocal rev
            rev = re.sub("<br />", "", rev)
            rev = re.sub("'s", " is", rev)
            rev = re.sub("'ve", " have", rev)
            rev = re.sub("'t", " not", rev)
            rev = re.sub("cannot", " can not", rev)
            rev = re.sub("'re", " are", rev)
            rev = re.sub("'d", " would", rev)
            rev = re.sub("'ll", " will", rev)
            rev = re.sub("[^a-z ]+", '', rev.lower().replace('.',' ').replace(',',' '))

            # stemmer = SnowballStemmer('english')
            # rev = [stemmer.stem(word) for word in rev]
            from nltk.stem import WordNetLemmatizer
            import nltk
            wordnet_lemmatizer = WordNetLemmatizer()
            rev = list(map(lambda x: wordnet_lemmatizer.lemmatize(x), rev.split()))

        def remove_stop_words():
            self.__logger.info("tokenizing the review")
            nonlocal rev
            #rev = rev.split()
            stop_words = set(stopwords.words('english'))
            rev = [w for w in rev if w not in stop_words]
            rev = " ".join(rev)

        polar = self.__path.split('/')[-2]
        idx = 0
        for file in os.listdir(path):
            if file.endswith('.txt'):
                idx+=1
                self.__logger.info('working on the {},{}, out of the {} set'.format(idx, polar, type))
                with open(os.path.join(path,file), 'r', encoding="utf-8") as f:
                    self.__logger.info('reading the review')
                    rev = f.read()
                    normalize()
                    remove_stop_words()
                    self.__logger.info('saving the data')
                    self.prep['X_train'].append(rev)
                    self.prep['Y_train'].append(int(polarity))


    @property
    def sent_to_seq(self):
        def find_max_seq():
            def find_max_for_type(type):
                self.__logger.info('finding the max sequence len for the {} set'.format(type))
                #print(len(self.prep['X_train']))
               # return max([len(set(sentence)) for sentence in self.prep['X_{}'.format(type)]])
                return max([len(set(sentence)) for sentence in self.prep['X_train']])
            #self.prep['max_seq_len'] = max(find_max_for_type('train'), find_max_for_type('test'))

        def dataset_to_seq():
            tokenizer = Tokenizer(num_words=15000)
            tokenizer.fit_on_texts(self.prep['X_train'])
            self.prep['X_train'] = tokenizer.texts_to_sequences(self.prep['X_train'])
            self.prep['vocab'] = tokenizer.word_index


        def pad_seq():
            self.prep['X_{}'.format(type)] = pad_sequences(self.prep['X_train'], padding='post',
                                                           truncating='post', maxlen=140)


        type = 'train'
        dataset_to_seq()
        print(self.prep['X_train'][0])
        pad_seq()
        print(self.prep['X_train'][0])
        #type = 'test'
        #dataset_to_seq()
        #pad_seq()
        from sklearn.model_selection import train_test_split
        self.prep['X_train'],self.prep['X_test'],self.prep['Y_train'],self.prep['Y_test'] =\
            train_test_split(self.prep['X_train'], self.prep['Y_train'], test_size=0.33, random_state=42)


