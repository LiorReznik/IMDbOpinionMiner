import numpy as np
import  tensorflow as tf
import os,urllib,tarfile,re


class Preprocess:

    def __init__(self, path='http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz', archive_name='imdbrev.tar.gz',folder_name='imdbrev'):
        self.__prep = {'X_train': np.array([]), 'Y_train': [], 'X_test': np.array([]), "Y_test": []}
        self.__reviews = {'train': [], "test": []}
        self.__path=path
        self.__archive_name = archive_name
        self.__folder_name = folder_name
        self.__download_reviews
        self.__prepare_reviews
        self.__pad_and_truncate('train')
        self.__pad_and_truncate('test')


    @property
    def __download_reviews(self):
        """""
        function to download the reviews
        """""
        if not os.path.exists(self.__archive_name):
            try:
                self.__reviews_tar, _ = urllib.request.urlretrieve(self.__path, self.__archive_name)
            except urllib.error as e:
                print(e)

            print("done downloading")
        else:
             print ("File already exists")

    @property
    def __prepare_reviews(self):
        """""
        function to extract the reviews
        """""
        def extract():
            print("starting the extraction")
            with tarfile.open(self.__archive_name) as file:
                file.extractall()
            print("done extracting")

        if not os.path.exists(self.__folder_name):
             extract()
        self.__get_reviews(path='aclImdb/train/pos/',polarity=True,type='train')
        self.__get_reviews(path='aclImdb/train/neg/',polarity=False,type='train')
        self.__get_reviews(path='aclImdb/test/pos/', polarity=True, type='test')
        self.__get_reviews(path='aclImdb/test/neg/', polarity=False, type='test')
        self.__prep['Y_train'] = np.array(self.__prep['Y_train'])
        self.__prep['Y_test'] = np.array(self.__prep['Y_test'])

    def __get_reviews(self,path,polarity, type):

        def normalize():
            nonlocal rev
            rev = rev.lower().replace('<br />', ' ')
            rev = re.sub("[^a-z0-9 ]+", '', rev)

        for file in os.listdir(path):
            if file.endswith('.txt'):
                with open(os.path.join(path,file), 'r', encoding="utf-8") as f:
                    rev = f.read()
                    normalize()
                    self.__reviews[type].append(rev)
                    self.__prep['Y_{}'.format(type)].append(int(polarity))

    def __pad_and_truncate(self, type):

        def find_max_seq():
            return max([len(set(sentence.split())) for sentence in self.__reviews[type]])

        self.__vocab = tf.contrib.learn.preprocessing.VocabularyProcessor(find_max_seq())
        self.__prep['X_{}'.format(type)] = np.array(list(self.__vocab.fit_transform(self.__reviews[type])))




Preprocess()