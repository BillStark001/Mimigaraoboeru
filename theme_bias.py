# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 10:48:39 2018

@author: BillStark001
"""

import pandas as pd
import numpy as np
import sklearn.decomposition
import pickle
import os
from tqdm import tqdm
import MeCab
mecab = MeCab.Tagger("-Ochasen")

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

lrc_conv_path = './lyrics_converted/'
lrc_fav_path = './lyrics_converted/0_fav/'
word_impt_path = 'word_importance.pkl'
impt, exist, words_map = quick_load(word_impt_path, ([], [], {}))
illegal_words = quick_load('illegal_words.pkl', [])
useful_words = ['動詞-自立', '副詞-一般', '名詞-サ変接続', '名詞-一般', '名詞-代名詞-一般', '形容詞-自立', '名詞-形容動詞語幹']

def separate(s, ans={}):
    s = mecab.parse(s).split('\n')[:-2]
    for i in range(len(s)):
        s[i] = s[i].split('\t')
    s = pd.DataFrame(s, columns=['text', 'yomigata', 'word', 'type', 'type_2', 'type_3'])
    for i in s.index:
        if not s['type'][i] in useful_words: continue
        if not s['word'][i] in ans:
            ans[s['word'][i]] = 0
        ans[s['word'][i]] += 1
    return ans

def generate_tfidf():
    idf = {}
    tf = {}
    lists = os.listdir(lrc_conv_path)#[:5]
    for i in tqdm(range(len(lists)), ncols=32, desc='LFIT'):#Lyrics Filtration
        c_dir = lrc_conv_path + lists[i] + '/'
        for file in tqdm(os.listdir(c_dir)[:10000], ncols=32, desc='LWCO'):#Lyric-Wise Collection
            id_cur = int(file.split('_')[0])
            if id_cur in tf: continue
            try:
                f = open(c_dir + file, 'r', encoding='utf-8')
                s = f.read()
                f.close()
                idf = separate(s, idf)
                tf[id_cur] = separate(s, {})
            except:
                pass
        f = open('proc_idf.log', 'w') 
        f.writelines([str(i)])
        f.close()
    return tf, idf

def clear_tf(tf):
    for i in tf:
        for j in illegal_words:
            if j in tf[i]:
                del(tf[i][j])
    return tf

def generate_impt(tf, div_t=24):
    ans = 0
    div = 0
    for i in tf:
        if i in words_map:
            ans += (1 - impt[words_map[i]]) * tf[i]
        else:
            ans += 0.5
        div += tf[i]
    if div <= div_t: ans = 0
    else: ans /= div
    return ans 

def generate_impts(tf):
    ans = []
    for i in tf:
        ans.append((i, generate_impt(tf[i])))
    ans.sort(key=lambda x: x[1], reverse=True)
    return ans

def get_tfidf_freq(tf, idf, t_freq=0.0001, t_idf=5):
    tfidf = {}
    del_idf = {}
    for i in tqdm(tf, ncols=32, desc='TFIT'): #TF Iteration
        tfidf_t = {}
        for j in tf[i]:
            if idf[j] < t_idf: 
                if not j in del_idf: del_idf[j] = 1
                del_idf[j] += 1
                continue#tqdm(tf[i], ncols=48, desc='WDIT'): #Word Iteration
            tfidf_t[j] = tf[i][j] / idf[j]
        tfidf_t = list(tfidf_t.items())
        tfidf_t.sort(key=lambda x: x[1], reverse=True)
        j = 0
        while j < len(tfidf_t):
            if tfidf_t[j][1] < t_freq:
                tfidf_t.pop(j)
            else: j += 1
        tfidf[i] = tfidf_t 
    for i in del_idf:
        del(idf[i])
    return tfidf

def get_simple_tfidf(tfidf, word_map):
    ans = np.zeros(word_map)
    for i in tfidf:
        ans[word_map[i]] = tfidf[i]
    return ans

def get_maps(tfidf):
    song_map = {}
    song_map_t = []
    word_map = {}
    word_map_t = []
    song_count = 0
    word_count = 0
    for i in tfidf:
        song_map[i] = song_count
        song_map_t.append(i)
        song_count += 1
        for j in tfidf[i]:
            if not j[0] in word_map:
                word_map[j[0]] = word_count
                word_map_t.append(j[0])
                word_count += 1
    return song_map, word_map, song_map_t, word_map_t

def get_lda_matrix(tfidf, song_map, word_map, threshold=(0, 1000)):
    song_count = len(song_map)
    word_count = len(word_map)
    if threshold[1] < threshold[0]: threshold = (threshold[1], threshold[0])
    t_ = threshold[1] - threshold[0]
    #if threshold[1] > len(tfidf): threshold = (threshold[0], len(tfidf))
    
    #print('Generating LDA Matrix: (%d, %d)'%(min(song_count, t_), word_count))
    ans = np.zeros((min(song_count - threshold[0], t_), word_count), dtype='float64')
    for i in tfidf:
        for j in tfidf[i]:
            if song_map[i] > threshold[0] and song_map[i] < threshold[1]:
                ans[song_map[i] - threshold[0], word_map[j[0]]] = j[1]
    return ans

def get_fav_lyrics():
    s = os.listdir(lrc_fav_path)
    ans1 = []
    ans2 = []
    for i in s:
        i = i.split('_')
        ans1.append(int(i[0]))
        ans2.append(int(i[1]))
    return ans1, ans2

def concatenate(l1, l2, percent=0.5, default=(0., 0.)):
    ans_ = {}
    ans = {}
    for i in l1: ans_[i[0]] = 0
    for i in l2: ans_[i[0]] = 0
    for i in l1: ans[i[0]] = .0
    for i in l2: ans[i[0]] = .0
    for i in l1: ans_[i[0]] += 1
    for i in l2: ans_[i[0]] += 2
    for i in l1:
        ans[i[0]] += i[1] * percent
        if ans_[i[0]] == 1: ans[i[0]] += default[1]
    for i in l2:
        ans[i[0]] += i[1] * (1 - percent)
        if ans_[i[0]] == 2: ans[i[0]] += default[0]
    ans = list(ans.items())
    ans.sort(key=lambda x: x[1], reverse=True)
    return ans

if __name__ == '__main__':
    '''
    tf, idf = generate_tfidf()
    tf = clear_tf(tf)
    tfidf = get_tfidf_freq(tf, idf)
    quick_dump((tf, idf, tfidf), 'tf_idf.pkl')
    '''
    pass
    '''
    tf, idf, tfidf = quick_load('tf_idf.pkl')
    lyric_importance = generate_impts(tf)
    
    n_topics = 12
    song_map, word_map, song_map_t, word_map_t = get_maps(tfidf)
    lda_mat = get_lda_matrix(tfidf, song_map, word_map, threshold=(0, 6000))
    LDA = sklearn.decomposition.LatentDirichletAllocation(n_components=n_topics)
    LDA.fit(lda_mat)
    topics = np.zeros((0, n_topics))
    sep = 6000
    for i in tqdm(range(0, len(tfidf), sep)):
        lda_mat = get_lda_matrix(tfidf, song_map, word_map, threshold=(i, i + sep))
        topics = np.concatenate((topics, LDA.transform(lda_mat)))
    ids, ws = get_fav_lyrics()
    w_fav = np.zeros((0, n_topics))
    for i in range(len(ids)):
        w_fav = np.concatenate((w_fav, topics[song_map[ids[i]]].reshape(1, n_topics) * ws[i]), axis=0)
    v_fav = w_fav.sum(axis=0)
    v_fav /= v_fav.max()
    theme_bias = []
    for i in range(topics.shape[0]):
        theme_bias.append((song_map_t[i], np.dot(v_fav, topics[i])))
    theme_bias.sort(key=lambda x: x[1], reverse=True)
    
    
    #bias = concatenate(lyric_importance, theme_bias)
    #quick_dump((bias, lyric_importance, theme_bias), 'theme_bias.pkl')
    '''
    _, lyric_importance, theme_bias = quick_load('theme_bias.pkl')
    music_bias = quick_load('music_bias.pkl')
    song_bias = concatenate(music_bias, theme_bias)
    final_order = concatenate(lyric_importance, song_bias)
    quick_dump(final_order, 'final_order.pkl')
    
    