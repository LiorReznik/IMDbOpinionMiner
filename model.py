from preprocess import Preprocess
import numpy as np
from keras.layers import Dense , Input , LSTM , Embedding, Dropout , Activation, GRU, Flatten
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model, Sequential
from sklearn.metrics import classification_report
from keras.callbacks import TensorBoard

class Trainer:

    def __init__(self,preprocess):
        self.__preprocess = preprocess.prep
        self.train()
        self.predict()

    def train(self):
        tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0,
                                  write_graph=True, write_images=False)
        embed_size = 100
        self.model = Sequential([Embedding(7000, 128),
                                 Bidirectional(LSTM(128, return_sequences=True)),
                                 GlobalMaxPool1D(),
                                 Dense(128, activation="relu"),
                                 Dropout(0.05),
                                 Dense(1, activation="sigmoid")])

        self.model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        print(self.model.summary())
        self.batch_size = 15
        epochs = 4
        self.model.fit(self.__preprocess['X_train'], self.__preprocess['Y_train'], batch_size=self.batch_size, epochs=epochs,
                       validation_split=0.2, callbacks=[tensorboard])

    def predict(self):
        predicts = np.array([i[0] for i in self.model.predict_classes(self.__preprocess['X_test'])])
        print(classification_report(self.__preprocess['Y_test'], predicts))
        score, acc = self.model.evaluate(self.__preprocess['X_test'], self.__preprocess['Y_test'],
                                    batch_size=self.batch_size)
        print('Test score:', score)
        print('Test accuracy:', acc)
    def predict_on_news(self):pass



Trainer(preprocess=Preprocess())