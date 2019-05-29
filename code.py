# -*- coding: utf-8 -*-
"""
Created on Tue May 28 20:21:21 2019

@author: Garry
"""

import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import pandas as pd
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import LSTM, Dropout,add,Embedding, TimeDistributed, Dense, RepeatVector, Activation, Flatten,BatchNormalization
from keras.optimizers import Adam, RMSprop
from keras.layers.wrappers import Bidirectional
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
import nltk



file='Flickr8k/Flickr8k_text/Flickr8k.token.txt'
caption=open(file,'r').read().strip().split('\n')


data = {}
for row in caption:
  row = row.split('\t')
  row[0] = row[0][:len(row[0])-2]
  if row[0] in data:
    data[row[0]].append(row[1])
  else:
    data[row[0]] = [row[1]]



import glob
images = 'Flickr8k/Flicker8k_Dataset/'
img = glob.glob(images+'*.jpg')


train= 'Flickr8k/Flickr8k_text/Flickr_8k.trainImages.txt'
train = set(open(train, 'r').read().strip().split('\n'))

test= 'Flickr8k/Flickr8k_text/Flickr_8k.testImages.txt'
test = set(open(test, 'r').read().strip().split('\n'))

val= 'Flickr8k/Flickr8k_text/Flickr_8k.devImages.txt'
val = set(open(val, 'r').read().strip().split('\n'))





def get_data(d):
  temp=[]
  for i in img:
    if i[27:] in d:
      temp.append(i)
  return temp


train_img=get_data(train)
test_img=get_data(test)
val_img=get_data(val)


def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x
def preprocess(image_path):
    img = image.load_img(image_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x)
    return x



model = InceptionV3(weights='imagenet')

from keras.models import Model
new_input = model.input
hidden_layer = model.layers[-2].output
model_new = Model(new_input, hidden_layer)


def encode(image):
    image = preprocess(image)
    temp_enc = model_new.predict(image)
    temp_enc = np.reshape(temp_enc, temp_enc.shape[1])
    return temp_enc

'''
encoding_train = {}
for img in tqdm(train_img):
    encoding_train[img[len(images):]] = encode(img)


encoding_test = {}
for img in tqdm(test_img):
    encoding_test[img[len(images):]] = encode(img)


with open("encoded_images_inceptionV3.p", "wb") as encoded_pickle:
    pickle.dump(encoding_train, encoded_pickle)


with open("encoded_images_test_inceptionV3.p", "wb") as encoded_pickle:
    pickle.dump(encoding_test, encoded_pickle)

'''
encoding_train = pickle.load(open('encoded_images_inceptionV3.p', 'rb'))
encoding_test = pickle.load(open('encoded_images_test_inceptionV3.p', 'rb'))

train_d = {}
for i in train_img:
    if i[len(images):] in data:
        train_d[i] = data[i[len(images):]]

val_d = {}
for i in val_img:
    if i[len(images):] in data:
        val_d[i] = data[i[len(images):]]

test_d = {}
for i in test_img:
    if i[len(images):] in data:
        test_d[i] = data[i[len(images):]]

######################################################
caps = []
for key, val in train_d.items():
    for i in val:
        caps.append('<start> ' + i + ' <end>')

words = [i.split() for i in caps]


unique = []
for i in words:
    unique.extend(i)
unique = list(set(unique))



'''
with open("unique.p", "wb") as pickle_d:
  pickle.dump(unique, pickle_d)
'''

unique = pickle.load(open('unique.p', 'rb'))


word2idx = {val:index for index, val in enumerate(unique)}
idx2word = {index:val for index, val in enumerate(unique)}



f = open('flickr8k_training_dataset.txt', 'w')
f.write("image_id\tcaptions\n")

for key, val in train_d.items():
  for i in val:
    f.write(key[len(images):] + "\t" + "<start> " + i +" <end>" + "\n")
f.close()

df = pd.read_csv('flickr8k_training_dataset.txt', delimiter='\t')
c = [i for i in df['captions']]
imgs = [i for i in df['image_id']]



max_len = 0
for c in caps:
    c = c.split()
    if len(c) > max_len:
        max_len = len(c)

#########################################

def data_generator(batch_size = 32):
        partial_caps = []
        next_words = []
        images = []
        
        df = pd.read_csv('flickr8k_training_dataset.txt', delimiter='\t')
        df = df.sample(frac=1)
        iter = df.iterrows()
        c = []
        imgs = []
        for i in range(df.shape[0]):
            x = next(iter)
            c.append(x[1][1])
            imgs.append(x[1][0])


        count = 0
        while True:
            for j, text in enumerate(c):
                current_image = encoding_train[imgs[j]]
                for i in range(len(text.split())-1):
                    count+=1
                    
                    partial = [word2idx[txt] for txt in text.split()[:i+1]]
                    partial_caps.append(partial)
                    
                    # Initializing with zeros to create a one-hot encoding matrix
                    # This is what we have to predict
                    # Hence initializing it with vocab_size length
                    n = np.zeros(vocab_size)
                    # Setting the next word to 1 in the one-hot encoded matrix
                    n[word2idx[text.split()[i+1]]] = 1
                    next_words.append(n)
                    
                    images.append(current_image)

                    if count>=batch_size:
                        next_words = np.asarray(next_words)
                        images = np.asarray(images)
                        partial_caps = sequence.pad_sequences(partial_caps, maxlen=max_len, padding='post')
                        yield [[images, partial_caps], next_words]
                        partial_caps = []
                        next_words = []
                        images = []
                        count = 0


###########################



samples_per_epoch = 0
for ca in caps:
    samples_per_epoch += len(ca.split())-1


vocab_size = len(unique)
embedding_size = 300
EMBEDDING_DIM = 300
from keras.layers import *

inputs1 = Input(shape=(2048,))
fe1 = Dropout(0.3)(inputs1)
fe2 = Dense(EMBEDDING_DIM, activation='relu')(fe1)

# partial caption sequence model
inputs2 = Input(shape=(max_len,))
se1 = Embedding(vocab_size,EMBEDDING_DIM, mask_zero=True)(inputs2)
se2 = Dropout(0.3)(se1)
se3 = LSTM(EMBEDDING_DIM)(se2)

# decoder (feed forward) model
decoder1 = add([fe2, se3])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)

# merge the two input models
model = Model(inputs=[inputs1, inputs2], outputs=outputs)

model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])



epoch = 3
batch_size = 128
model.fit_generator(data_generator(batch_size=batch_size), steps_per_epoch=samples_per_epoch/batch_size, epochs=epoch, verbose=1, callbacks=None)

model.save("Weights_1.h")





#####TESTIN#########################
fin_model=load_model('Weights_1.h')


def predict_captions(image_file):
    start_word = ["<start>"]
    while 1:
        now_caps = [word2idx[i] for i in start_word]
        now_caps = sequence.pad_sequences([now_caps], maxlen=max_len, padding='post')
        e = encoding_test[image_file]
        preds = fin_model.predict([np.array([e]), np.array(now_caps)])
        word_pred = idx2word[np.argmax(preds[0])]
        start_word.append(word_pred)
        
        if word_pred == "<end>" or len(start_word) > max_len: 
    #keep on predicting next word unitil word predicted is <end> or caption lenghts is greater than max_lenght(40)
            break
            
    return ' '.join(start_word[1:-1])





def beam_search_predictions(image_file, beam_index = 3):
    start = [word2idx["<start>"]]
    
    start_word = [[start, 0.0]]
    
    while len(start_word[0][0]) < max_len:
        temp = []
        for s in start_word:
            now_caps = sequence.pad_sequences([s[0]], maxlen=max_len, padding='post')
            e = encoding_test[image_file]
            preds = fin_model.predict([np.array([e]), np.array(now_caps)])
            
            word_preds = np.argsort(preds[0])[-beam_index:]
            
            # Getting the top Beam index = 3  predictions and creating a 
            # new list so as to put them via the model again
            for w in word_preds:
                next_cap, prob = s[0][:], s[1]
                next_cap.append(w)
                prob += preds[0][w]
                temp.append([next_cap, prob])
                    
        start_word = temp
        # Sorting according to the probabilities
        start_word = sorted(start_word, reverse=False, key=lambda l: l[1])
        # Getting the top words
        start_word = start_word[-beam_index:]
    
    start_word = start_word[-1][0]
    intermediate_caption = [idx2word[i] for i in start_word]

    final_caption = []
    
    for i in intermediate_caption:
        if i != '<end>':
            final_caption.append(i)
        else:
            break
    
    final_caption = ' '.join(final_caption[1:])
    return final_caption
image_file='3218480482_66af7587c8.jpg'
test_image =  'Flickr8k/Flicker8k_Dataset/3218480482_66af7587c8.jpg'
Image.open(test_image)

print ('Greedy search:', predict_captions(image_file))
print ('Beam Search, k=3:', beam_search_predictions(image_file, beam_index=3))
print ('Beam Search, k=5:', beam_search_predictions(image_file, beam_index=5))
print ('Beam Search, k=7:', beam_search_predictions(image_file, beam_index=7))









