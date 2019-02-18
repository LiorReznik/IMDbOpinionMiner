# -*- coding: utf-8 -*-
"""
@author: Lior Reznik
"""
import numpy as np
import os,urllib,tarfile,re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
import pickle,nltk


class Preprocess:
    def __init__(self, path='http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz', archive_name='imdbrev.tar.gz',folder_name='aclImdb'):
        nltk.download('wordnet')#if the wordnet is not uptodate or doesnot exsists in the computer then downloading it from nltk servers we will nedd this for lemma
        nltk.download('stopwords')#if the stopwords vocab is not uptodate or doesnot exsists in the computer downloading then it from nltk servers 
        self.prep = {'X': [], 'Y': []}
        self.path = path
        self.archive_name = archive_name
        self.folder_name = folder_name
        self.preprocess
        print(self.prep['X_train'].shape, self.prep['X_test'].shape, self.prep['Y_train'].shape, self.prep['Y_test'].shape )

    @property
    def preprocess(self):
        self.download_reviews
        self.prepare_reviews
        self.sent_to_seq

    @property
    def download_reviews(self):
        """""
        function to download the reviews form stanfords servers
        """""
        if not os.path.exists(self.archive_name):
            try:
                print("Downloading the reviews")
                self.__reviews_tar, _ = urllib.request.urlretrieve(self.path, self.archive_name)
            except urllib.error as e:
                print(e.to_str())

            print("Download has completed")
        else:
            print("Skipping Download,The Archive already in the HD")

    @property
    def prepare_reviews(self):
        """""
        function to extract the reviews and sends them into preprocessing
        """""
        def extract():
            print("starting the extraction")
            with tarfile.open(self.archive_name) as file:
                file.extractall()
            print("done extracting")

        if not os.path.exists(self.folder_name):
            extract()
        else:
            print("the archive is already in the disk")

        self.get_and_prep(path='aclImdb/train/pos/', polarity=True)
        self.get_and_prep(path='aclImdb/train/neg/', polarity=False)
        self.get_and_prep(path='aclImdb/test/pos/', polarity=True)
        self.get_and_prep(path='aclImdb/test/neg/', polarity=False)

    def get_and_prep(self, path, polarity):

        def normalize():
            print('Normalizing the review')
            nonlocal rev
            rev = re.sub("<br />", "", rev)
            rev = re.sub("'s", " is", rev)
            rev = re.sub("'ve", " have", rev)
            rev = re.sub("n't", " not", rev)
            rev = re.sub("cannot", " can not", rev)
            rev = re.sub("'re", " are", rev)
            rev = re.sub("'d", " would", rev)
            rev = re.sub("'ll", " will", rev)
            rev = re.sub("[^a-z ]+", '', rev.lower().replace('.',' ').replace(',',' '))

            #lemmatizing the review
            from nltk.stem import WordNetLemmatizer
            wordnet_lemmatizer = WordNetLemmatizer()
            rev = list(map(lambda x: wordnet_lemmatizer.lemmatize(x), rev.split()))

        def remove_stop_words():
            """""
            removing stop words except not and nor
            """""
            print("tokenizing the review")
            nonlocal rev
            stop_words = set(stopwords.words('english')) - {'not', 'nor'}
            rev = [w for w in rev if w not in stop_words]
            rev = " ".join(rev)

        for file in os.listdir(path):
            if file.endswith('.txt'):
                with open(os.path.join(path,file), 'r', encoding="utf-8") as f:
                    print('reading the review')
                    rev = f.read()
                    normalize()
                    remove_stop_words()
                    print('saving the data')
                    self.prep['X'].append(rev)
                    self.prep['Y'].append(int(polarity))

    @property
    def sent_to_seq(self):
        def to_numpy():
            self.prep['X'] = np.array(self.prep['X'])
            self.prep['Y'] = np.array(self.prep['Y'])

        def dataset_to_seq():
            # indicating to the model that we want only the top 25k words to save when transforming to seq
            tokenizer = Tokenizer(num_words=25000)
            # building  a vocab out of the data and saving all the words,
            # the numeric representation that will be given to the words will be the lowest index is the word with the higer freq
            tokenizer.fit_on_texts(self.prep['X'])
            # making a vector of sequences(matrix) were only the top 25k words is taken into consideration
            self.prep['X'] = tokenizer.texts_to_sequences(self.prep['X'])
            self.prep['vocab'] = tokenizer.word_index#saving the vocab so we can use it when loading the model

        def pad_seq():
            self.prep['X'] = pad_sequences(self.prep['X'], padding='post',
                                                           truncating='post', maxlen=180)

        def split():
            from sklearn.model_selection import train_test_split
            self.prep['X_train'], self.prep['X_test'], self.prep['Y_train'], self.prep['Y_test'] = \
                train_test_split(self.prep['X'], self.prep['Y'], test_size=0.33, random_state=42)

        def to_pickle():
            with open('vocab.pickle', 'wb') as f:
                pickle.dump(self.prep['vocab'], f,pickle.HIGHEST_PROTOCOL)

        to_numpy()
        dataset_to_seq()
        pad_seq()
        split()
        to_pickle()
