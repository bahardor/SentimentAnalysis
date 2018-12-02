# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 12:30:31 2018
@author: Bahar Dorri
"""

from string import punctuation
import csv
import itertools
import numpy as np
from sklearn.metrics import confusion_matrix
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import LSTM#, Dropout
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, precision_score, recall_score
from matplotlib import pyplot as plt

def load_doc(filename):
    global recommend
    text=[]
    recommend=[]
    with open(filename, 'r') as f:
        data = csv.reader(f)
        data=list(data)
        for i in range(1,len(data)):
            temp=data[i][4]
            if temp!='':
                
                text.append(temp)
                temp=''
                recommend.append(data[i][6])
    #print(len(recommend),len(text))  #22641  
    f.close()
    return(text)
 
# turn a doc into clean tokens
def clean_doc(doc, vocab):
    documents = list()
    table = str.maketrans('', '', punctuation)
    for i in range(len(doc)):
        tokens = doc[i].lower().split()
    	# remove punctuation from each token
        tokens = [w.translate(table) for w in tokens]

    	# filter out tokens not in vocab
        tokens = [w for w in tokens if w in vocab]
        tokens = ' '.join(tokens)
        documents.append(tokens)

    return documents

def process_docs(path, vocab):
    doc= load_doc(path)
    documents = clean_doc(doc, vocab)
    return documents
 
fileV=open('vocab.txt',"r")
vocab = fileV.read()
fileV.close()
vocab = vocab.split()
vocab = set(vocab)

# load all training reviews
reviews=process_docs('WomensClothing-Reviews.csv',vocab)

train_docs = reviews[1:16001]#[1:1000]
test_docs = reviews[16001:19001] #[1000:1100]

ytrain = recommend[1:16001]#[1:1000]
ytest =  recommend[16001:19001] #[1000:1100]'''

'''train_docs = reviews[1:1000]
test_docs = reviews[1000:1100]

ytrain = recommend[1:1000]
ytest =  recommend[1000:1100]'''

tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_docs) # fit the tokenizer on the documents

max_length = max([len(s.split()) for s in train_docs]) 

# sequence encode
encodedtrain_docs = tokenizer.texts_to_sequences(train_docs)
Xtrain = pad_sequences(encodedtrain_docs, maxlen=max_length, padding='post') # pad sequences

encodedtest_docs = tokenizer.texts_to_sequences(test_docs)
Xtest = pad_sequences(encodedtest_docs, maxlen=max_length, padding='post')

vocab_size = len(tokenizer.word_index) + 1

model = Sequential()
model.add(Embedding(vocab_size,150, input_length=max_length))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
#model.add(Dropout(0.2))
model.add(LSTM(100))
#model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

kfold = KFold(10, True, 1)
Xtrain.astype(int)
ytrain=np.array(ytrain)

for train, valid in kfold.split(Xtrain, ytrain):
    model.fit(Xtrain[train], ytrain[train],validation_data=(Xtrain[valid], ytrain[valid]), epochs=5)#, epochs=10, batch_size=64)

y_pred=model.predict_classes(Xtest)
y_pred=y_pred.ravel()

yt=[]
for i in range(len(Xtest)):
    yt.append(int(ytest[i]))
yt=np.asarray(yt)
    
scores = model.evaluate(Xtest, ytest)#, verbose=0)
print("Test Accuracy: %.2f%%" % (scores[1]*100))
print("Test Loss: %.2f%%" % (scores[0]*100))

F1=f1_score(yt, y_pred ,average='binary')
print('F1=',F1)  

Precision=precision_score(yt, y_pred ,average='binary')
print('Precision=',Precision)

Recall=recall_score(yt, y_pred ,average='binary')
print('Recall=',Recall)

def plot_confusion_matrix(cm, classes, title='Confusion Matrix',cmap=plt.cm.Blues):
  
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("Confusion matrix")
    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' 
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

class_names={'Recommend','Not Recommend'}
cnf_matrix = confusion_matrix(yt, y_pred)
np.set_printoptions(precision=2)

plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,title='Confusion Matrix')

plt.show()