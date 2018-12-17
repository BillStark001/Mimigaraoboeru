# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 20:29:54 2018

@author: Zhao
"""

import os

import pandas as pd
import cv2
import numpy as np

rendered_path = '../Datasets/fma_rendered'
data_dir = rendered_path + '/{:0>6}.png'
music_dir = './musics_rendered/{}.png'
info = pd.read_csv('fma_info.csv')

def to_one_hot(x, sort=16):
    ans = np.zeros(sort, dtype='float')
    ans[x] = 1.
    return ans

y_map = {6: 5, 8: 6, 14: 7}

def data_generator(batch_size=32, length=300, r=(0, 8000)):
    #global kang
    while True:
        x = []
        y = []
        r_ = np.random.randint(low=r[0], high=r[1], size=batch_size)
        for i in r_:
            #print(data_dir.format(info['id'][i]))
            img = cv2.imread(data_dir.format(info['id'][i]), cv2.IMREAD_GRAYSCALE) / 256
            img = img[:, :length]
            '''
            if img.size == 38400: 
                print(info['id'][i])
                if not info['id'][i] in kang: kang.append(info['id'][i])
                continue
            '''
            img = img.reshape((256, 300, 1))
            x.append(img)
            y_ = info['genre_id'][i]
            if y_ in y_map: y_ = y_map[y_]
            y.append(to_one_hot(y_, sort=8))
        x = np.array(x)
        y = np.array(y)
        yield x, y
        
def load_data_from_music(mid, length=300):
    img = cv2.imread(music_dir.format(mid), cv2.IMREAD_GRAYSCALE) / 256
    cut = int(np.floor(img.shape[1] / length))
    img = img[:, :cut * length]
    img = np.reshape(img, (256, cut, length, 1))
    img = img.transpose((1, 0, 2, 3))
    return img
        
if __name__ == '__main__':
    '''
    g = data_generator(400, r=(0, 6000))
    while True:
        x, y = next(g)
    '''
    m = load_data_from_music(22637814)
    
    