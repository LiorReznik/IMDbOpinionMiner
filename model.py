from preprocess import Preprocess
import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense , Conv1D,MaxPooling1D , LSTM , Embedding, Dropout , Activation, GRU, Flatten
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model, Sequential
from sklearn.metrics import classification_report
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
import time

class Trainer:

    def __init__(self,preprocess):
        self.__preprocess = preprocess.prep
        self.train()
        self.predict()

    def train(self):
        tensorboard = TensorBoard(log_dir='./logs/test_25000_32_80{}'.format(int(time.time())), histogram_freq=2,
                                  write_graph=True, write_images=True)
        self.model = Sequential()
        self.model.add(Embedding(15000, 32, input_length=140))
        self.model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
        self. model.add(MaxPooling1D(pool_size=2))
        self.model.add(LSTM(200, dropout=0.2))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(loss='binary_crossentropy', optimizer = Adam(lr=0.0001), metrics=['accuracy'])
        print(self.model.summary())
        earlystopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')
        self.model.fit(self.__preprocess['X_train'], self.__preprocess['Y_train'], batch_size=32, epochs=50
        ,verbose=2, callbacks=[tensorboard], shuffle=True, validation_split=0.2)

    def predict(self):
        scores = self.model.evaluate(self.__preprocess['X_test'], self.__preprocess['Y_test'],verbose=1)
        print("Test accuracy:" , scores[1]*100)

        bad = "this movie was terrible and bad"
        good ="the new movie was not so bad"




Trainer(preprocess=Preprocess())