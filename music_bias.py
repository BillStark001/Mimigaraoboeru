# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 20:41:15 2018

@author: Zhao
"""

import os
import load_data
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
load_music = load_data.load_data_from_music

from keras.models import Model
from keras.layers import Conv2D, Flatten, Dense, Input, Dropout, MaxPooling2D
from keras.optimizers import SGD#, Momentum
from keras.callbacks import ReduceLROnPlateau

seq_shape = (256, 300, 1)
batch_train = 16
batch_val = 8
batch_iter = 64
batch_iter_val = 64
w_path='model.h5'

gen_train = load_data.data_generator(batch_train, r=(0, 6500))
gen_val = load_data.data_generator(batch_val, r=(6500, 8000))

def cnn_model(feature=False):
    im_in = Input(shape=seq_shape)
    '''
    x = Conv2D(4, 8, padding='same', activation='tanh', name='conv_1')(im_in)
    x = Conv2D(8, 8, padding='same', activation='tanh', strides=(2, 2), name='conv_2')(x)
    x = Conv2D(16, 4, padding='same', activation='tanh', strides=(2, 2), name='conv_3')(x)
    x = Conv2D(16, 4, padding='same', activation='tanh', strides=(2, 2), name='conv_4')(x)
    x = Conv2D(32, 4, padding='same', activation='tanh', strides=(2, 2), name='conv_5')(x)
    x = Conv2D(32, 4, padding='same', activation='tanh', strides=(2, 2), name='conv_6')(x)
    '''
    x = Conv2D(96, 11, padding='valid', activation='relu', strides=(4, 4), name='conv_1')(im_in)
    x = MaxPooling2D()(x)
    x = Conv2D(256, 5, padding='valid', activation='relu', name='conv_2')(x)
    x = MaxPooling2D()(x)
    x = Conv2D(256, 5, padding='valid', activation='relu', name='conv_3')(x)
    x = Conv2D(128, 3, padding='valid', activation='relu', name='conv_4')(x)
    x = Conv2D(128, 3, padding='valid', activation='relu', name='conv_5')(x)
    x = MaxPooling2D()(x)
    x = Flatten(name='flatten')(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='tanh', name='dense_1')(x)
    x = Dense(128, activation='tanh', name='dense_2')(x)
    sm_out = Dense(8, activation='softmax', name='dense_category')(x)
    if feature: model = Model(inputs=im_in, outputs=x)
    else: model = Model(inputs=im_in, outputs=sm_out)
    return model

def train_model(model, epoch=3, path=None):
    opt = SGD(lr=0.01, momentum=0.9, decay=0.0005)
    rlr = ReduceLROnPlateau(factor=np.sqrt(0.1))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    try:
        model.fit_generator(gen_train, epochs=epoch, steps_per_epoch=batch_iter, verbose=1, 
                            validation_data=gen_val, validation_steps=batch_iter_val, 
                            callbacks=[rlr])
    except:
        pass
    finally:
        model.save_weights(path)
    
    return model

def load_model(path, feature=False):
    model = cnn_model(feature=feature)
    model.load_weights(path, by_name=True)
    return model

music_path = './musics_rendered/'

def quick_dump(x, path):
    with open(path, 'wb') as f:
        pickle.dump(x, f)
        
def quick_load(path, default=None):
    try:
        with open(path, 'rb') as f:
            x = pickle.load(f)
        return x
    except:
        return default

def cosine_sim(x, y):
    a = np.sqrt(np.dot(x, x))
    b = np.sqrt(np.dot(y, y))
    return np.dot(x, y) / (a * b)

if __name__ == '__main__':
    pass
    #model = cnn_model()
    #model = load_model('./model_.h5')
    #model.summary()
    #train_model(model, epoch=16000, path=w_path)
    model = load_model(w_path, feature=True)
    #model.summary()
    
    fav_ids = os.listdir('./lyrics/0_fav/')
    for i in range(len(fav_ids)):
        fav_ids[i] = (int(fav_ids[i].split('_')[0]), int(fav_ids[i].split('_')[1]))
    
    music_dict = {}
    musics = os.listdir(music_path)#[:200]
    for i in range(len(musics)):
        musics[i] = int(musics[i][:-4])
    for i in tqdm(musics, ncols=32, desc='MUPD'):
        t = model.predict(load_music(i))
        music_dict[i] = np.average(t, axis=0)
    
    fav_vs = []
    for i in fav_ids:
        if i[0] in music_dict: fav_vs.append(music_dict[i[0]] * i[1])
    fav_vs = np.array(fav_vs)
    #plt.imshow(fav_vs)
    #plt.show()
    fav_vs = np.average(fav_vs, axis=0)
    
    music_bias = []
    for i in music_dict:
        music_bias.append((i, cosine_sim(fav_vs, music_dict[i])))
    music_bias.sort(key=lambda x: x[1], reverse=True)
    #plt.plot(x)
    quick_dump(music_bias, 'music_bias.pkl')
    
    
