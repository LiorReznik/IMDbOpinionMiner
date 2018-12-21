from preprocess import Preprocess
import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense , Conv1D,MaxPooling1D , LSTM , Embedding, Dropout , Activation, GRU, Flatten
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Sequential
from keras.callbacks import TensorBoard


class Trainer:

    def __init__(self,preprocess):
        self.__preprocess = preprocess.prep
        self.train
        self.evaluate

    # def train(self):
    #     tensorboard = TensorBoard(log_dir='./logs/test_25000_32_80{}'.format(int(time.time())), histogram_freq=2,
    #                               write_graph=True, write_images=True)
    #     self.model = Sequential()
    #     self.model.add(Embedding(15000, 32, input_length=140))
    #     self.model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    #     self. model.add(MaxPooling1D(pool_size=2))
    #     self.model.add(LSTM(200, dropout=0.2))
    #     self.model.add(Dense(1, activation='sigmoid'))
    #     self.model.compile(loss='binary_crossentropy', optimizer = Adam(lr=0.0001), metrics=['accuracy'])
    #     print(self.model.summary())
    #     earlystopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')
    #     self.model.fit(self.__preprocess['X_train'], self.__preprocess['Y_train'], batch_size=32, epochs=50
    #     ,verbose=2, callbacks=[tensorboard], shuffle=True, validation_split=0.2)

    @property
    def train(self):
        max_features = 25000
        # cut texts after this number of words (among top max_features most common words)
        maxlen = 60
        batch_size = 32
        tensorboard = TensorBoard(log_dir='./logs/BLSTMCNN__poo200_4p_10EPC_adam1', histogram_freq=0,write_graph=True, write_images=False)
#tensorboard --logdir=C:\sentimentReviews-master\logs\ --host=localhost --port=80

        print('Build model...')
        self.model = Sequential()
        self.model.add(Embedding(max_features, 96, input_length=180))
        #self.model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
        #self. model.add(MaxPooling1D(pool_size=2))
        self.model.add(Bidirectional(LSTM(200,recurrent_dropout=0.2, dropout=0.2,return_sequences=True)))
        self.model.add(Conv1D(filters=200, kernel_size=3, padding='same', activation='relu'))
        self. model.add(MaxPooling1D(pool_size=4))
        self.model.add(Flatten())
        self.model.add(Dense(1, activation='sigmoid'))
        early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')
        checkpiont=ModelCheckpoint('Checkpoints/BLSTMCNN_p200_4p_adam.h5', monitor='val_loss', verbose=1, save_best_only=True,
                                   save_weights_only=False, mode='auto', period=1)
        self.model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        print(self.model.summary())
        print('Train...')
        self.model.fit(self.__preprocess['X_train'], self.__preprocess['Y_train'],
                  batch_size=batch_size,
                  epochs=10,
                  validation_split=0.2,verbose=2, callbacks=[tensorboard,checkpiont,early_stop])

    @property
    def evaluate(self):
        scores = self.model.evaluate(self.__preprocess['X_test'], self.__preprocess['Y_test'],verbose=1)
        print("Test accuracy:" , scores[1]*100)


Trainer(preprocess=Preprocess())