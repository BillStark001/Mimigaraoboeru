# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 14:35:14 2018

@author: zhaoj
"""

import pandas as pd
import numpy as np
import pickle

words = pd.read_csv('words.csv')

seichi = []
for i in words['seichi']:
    i = i.split('，')
    for j in i:
        if not j in seichi:
            seichi.append(j)
seichi.sort()

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

def kunyomi_process(frame):
    ans = {}
    k = frame['kunyomi']
    s = frame['sezzoku']
    for i in frame.index:
        if isinstance(k[i], float): continue
        if not k[i] in ans: ans[k[i]] = []
        if not isinstance(s[i], float): ans[k[i]].append(s[i])
    return ans

yomigata = pd.read_csv('yomigata.csv', encoding='gbk')
x = yomigata['kanji'][1]
onyomi, kunyomi = {}, {}
kanji = []
cur_n = 0
cur_c = ''
for i in range(len(yomigata['kanji']) + 1):
    try:
        cur_k = yomigata['kanji'][i]
    except:
        cur_k = ''
    if not isinstance(cur_k, float):
        if cur_c != '':
            kanji.append(cur_c)
            onyomi_temp = list(pd.DataFrame(yomigata[i-cur_n-1:i])['onyomi'])
            onyomi[cur_c] = []
            for j in onyomi_temp:
                if not isinstance(j, float):
                    onyomi[cur_c].append(j)
            kunyomi_temp = pd.DataFrame(yomigata[i-cur_n-1:i], columns=['kunyomi', 'sezzoku'])
            kunyomi[cur_c] = kunyomi_process(kunyomi_temp)
        cur_c = cur_k 
        cur_n = 0
    else:
        cur_n += 1
quick_dump((kanji, onyomi, kunyomi), 'yomigata.pkl')
    
def save_process(i, path='proc.log'):
    f = open(path, 'w') 
    f.writelines([str(i)])
    f.close()

def load_process(path='proc.log', default=0):
    try:
        f = open(path, 'r') 
        i = int(f.readlines()[0])
        f.close()
        return i
    except:
        return default
    
word_impt_path = 'word_importance.pkl'
temp = load_process('word_importance.log')
impt, exist, words_map = quick_load(word_impt_path, ([], [], {}))
words_map_t = list(words_map)
yomigata_map = {}
for i, k in zip(words['kanji'], words['yomigata']):
    i = i.split('，')
    for j in i:
        yomigata_map[j] = k
wm = len(words_map)
for i in range(temp, wm):
    print('Please enter the importance of the following word:')
    print('%s, %s'%(words_map_t[i], yomigata_map[words_map_t[i]]))
    #print('%s %s'%(words['kanji'][i], words['yomigata'][i]))
    print('%.2f%% [%d/%d] '%(100*i/wm, i, wm) + str(exist[i]))
    if exist[i]: t_ = input()
    else: t_ = 0
    if t_ == 'q': break
    elif t_ == '': impt_ = .0
    else: impt_ = float(t_)
    impt[i] = impt_
    save_process(i+1, 'word_importance.log')
    quick_dump((impt, exist, words_map), word_impt_path)