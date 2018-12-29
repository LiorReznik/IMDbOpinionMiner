from tkinter import *
import re
import numpy as np
from tkinter import messagebox
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
import pickle

class View:
    def __init__(self, main):
        """
        :param main: the main frame
        """
        self.main = main#saving the main frame
        self.model=load_model('FINAL25.h5')#loading the model
        #opening the vocab
        with open('vocab.pickle', 'rb') as f:
            self.vocab = pickle.load(f)
        self.__build

    @property
    def __build(self):
        """""
        A function that builds the GUI
        """""
        self.main.geometry("561x341+689+150")
        self.main.title("Opinion miner")
        self.main.configure(background="#d9d9d9")
        self.rev_label = Label(self.main)
        self.rev_label.place(relx=0.267, rely=0.059, height=11, width=244)
        self.rev_label.configure(background="#d9d9d9")
        self.rev_label.configure(disabledforeground="#a3a3a3")
        self.rev_label.configure(foreground="#000000")
        self.rev_label.configure(text='''הכנס ביקורת''')
        self.rev_label.configure(width=244)
        self.rev = Text(self.main)
        self.rev.place(relx=0.143, rely=0.117, relheight=0.452, relwidth=0.756)
        self.rev.configure(background="white")
        self.rev.configure(font="TkTextFont")
        self.rev.configure(foreground="black")
        self.rev.configure(highlightbackground="#d9d9d9")
        self.rev.configure(highlightcolor="black")
        self.rev.configure(insertbackground="black")
        self.rev.configure(selectbackground="#c4c4c4")
        self.rev.configure(selectforeground="black")
        self.rev.configure(width=424)
        self.rev.configure(wrap='word')
        self.Message1 = Message(self.main)
        self.Message1.place(relx=0.16, rely=0.616, relheight=0.097, relwidth=0.731)
        self.Message1.configure(background="#d9d9d9")
        self.Message1.configure(foreground="#000000")
        self.Message1.configure(highlightbackground="#d9d9d9")
        self.Message1.configure(highlightcolor="black")
        self.Message1.configure(width=410)
        self.classify_button = Button(self.main,command=self.classify)
        self.classify_button.place(relx=0.784, rely=0.792, height=24, width=51)
        self.classify_button.configure(activebackground="#d9d9d9")
        self.classify_button.configure(activeforeground="#000000")
        self.classify_button.configure(background="#d9d9d9")
        self.classify_button.configure(disabledforeground="#a3a3a3")
        self.classify_button.configure(foreground="#000000")
        self.classify_button.configure(highlightbackground="#d9d9d9")
        self.classify_button.configure(highlightcolor="black")
        self.classify_button.configure(pady="0")
        self.classify_button.configure(text='''סווג''')

    def file_error_handling(self):
        messagebox.showerror("Error", "You need to enter a sentence")
        ans = messagebox.askyesno(message='Do you want to try again?')
        if not ans:
            messagebox.showinfo('', 'Exiting...')
            self.main.destroy()


    def classify(self):
        """""
        function to classift the review that the user has entered
        """""

        def normalize():
              nonlocal rev
              rev = re.sub("<br />", "", rev)
              rev = re.sub("'s", " is", rev)
              rev = re.sub("'ve", " have", rev)
              rev = re.sub("n't", " not", rev)
              rev = re.sub("cannot", " can not", rev)
              rev = re.sub("'re", " are", rev)
              rev = re.sub("'d", " would", rev)
              rev = re.sub("'ll", " will", rev)
              rev = re.sub("[^a-z ]+", '', rev.lower().replace('.', ' ').replace(',', ' '))

        def stem():
                nonlocal rev
                from nltk.stem import WordNetLemmatizer
                wordnet_lemmatizer = WordNetLemmatizer()
                rev = list(map(lambda x: wordnet_lemmatizer.lemmatize(x), rev.split()))

        def remove_stop_words():
            nonlocal rev
            stop_words = set(stopwords.words('english'))-{'not','nor'}
            rev = [w for w in rev if w not in stop_words]

        rev = self.rev.get("1.0", 'end-1c')
        normalize()
        stem()
        remove_stop_words()
        rev = " ".join(rev)
        #saving only the words that in the top 25k freq inf the vocab
        rev = [self.vocab[word]for word in rev.split() if word in self.vocab.keys() and self.vocab[word] <= 25000]

        padded = pad_sequences([rev],  padding='post', truncating='post', maxlen=180)
        self.prec = self.model.predict(np.array([padded][0]))[0][0]
        self.pred = 'good' if self.prec >= 0.70 else 'bad'
        print(self.prec)
        messagebox.showinfo("", "the review is {} with {} out of 1 score".format(self.pred,self.prec))


root = Tk()
View(root)

root.mainloop()

