from preprocess import Preprocess
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense , Conv1D,MaxPooling1D , LSTM , Embedding, Dropout, Flatten
from keras.layers import Bidirectional
from keras.models import Sequential
from keras.callbacks import TensorBoard
from keras.optimizers import rmsprop
from keras.models import load_model

class Trainer:
    """""
    class to train or evalute model
    """""

    def __init__(self,preprocess=Preprocess(), load=None):
        """""
        :param load->a pertained model to evaluate
        :param preprocess-> preprocessed data to train on or if load is not none then a data to test on
        """""
        self.preprocess = preprocess.prep
        if load:
            self.model = load_model(load)
        else:
            self.train
        self.evaluate

    @property
    def train(self):
        """""
        train the model on preprocessed data
        """""
        max_features = 25000 #the most higer feature to feed the network with
        batch_size = 32
        tensorboard = TensorBoard(log_dir='./logs/FINAL25', histogram_freq=0,write_graph=True, write_images=False)

        print('Build model...')
        #model configuration
        self.model = Sequential()
        self.model.add(Embedding(max_features, 96, input_length=180))
        self.model.add(Bidirectional(LSTM(200,recurrent_dropout=0.2, dropout=0.2,return_sequences=True)))
        self.model.add(Conv1D(filters=200, kernel_size=3, padding='same', activation='relu'))
        self. model.add(MaxPooling1D(pool_size=2))
        self.model.add(Flatten())
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1, activation='sigmoid'))
        # if the model does not improves after 3  sequal epochs then stop
        early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')
        #save the best model
        checkpiont=ModelCheckpoint('FINAL25.h5', monitor='val_loss', verbose=1, save_best_only=True,
                                   save_weights_only=False, mode='auto', period=1)
        self.model.compile(loss='binary_crossentropy',
                      optimizer=rmsprop(lr=0.0001),
                      metrics=['accuracy'])
        print(self.model.summary())
        print('Train...')
        self.model.fit(self.preprocess['X_train'], self.preprocess['Y_train'],
                       batch_size=batch_size, validation_split=0.2,
                       epochs=50, verbose=2, callbacks=[tensorboard,checkpiont,early_stop])

    @property
    def evaluate(self):
        """""
        evaluate the model on test data
        """""
        scores = self.model.evaluate(self.preprocess['X_test'], self.preprocess['Y_test'], verbose=1)
        print("Test accuracy:" , scores[1]*100)


Trainer(load='FINAL25.h5')