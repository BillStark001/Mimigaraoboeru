# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 10:48:39 2018

@author: BillStark001
"""

import pandas as pd
import sklearn.decomposition
LDA = sklearn.decomposition.LatentDirichletAllocation(n_components=32)
import pickle
import langconv
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

lrc_path = './lyrics/'
lrc_conv_path = './lyrics_converted/'
lrc_fav_path = './lyrics_converted/0_fav/'
japanese_chars = list(pd.read_csv('shift_jls.csv')['char'][209: 378])
bracket_left = ['(', '[', '{', '（', '〔', '［', '｛', '〈', '《', '「', '『', '【']
bracket_right = [')', ']', '}', '）', '〕', '］', '｝', '〉', '》', '」', '』', '】']
illegal_chars = []
illegal_chars += list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890')
illegal_chars += list('~!@#$%^&*()_+`-=|\\:;"\'<>,./?')
illegal_chars += list(pd.read_csv('shift_jls.csv')['char'][1: 607])
illegal_words = quick_load('illegal_words.pkl', [])
kanjis = quick_load('yomigata.pkl')[0]
cnmap = quick_load('zhcn2jajp.pkl', 'rb')
useful_words = ['動詞-自立', '副詞-一般', '名詞-サ変接続', '名詞-一般', '名詞-代名詞-一般', '形容詞-自立', '名詞-形容動詞語幹']

def filtrate_bracket(s): #Extremely PROBLEMATIC!
    ans = ''
    cur = 0
    for i in range(len(s)):
        if s[i] in bracket_left:
            ans += s[cur: i]
        if s[i] in bracket_right:
            cur = i + 1
    ans += s[cur: len(s)]
    return ans
    
def filtrate_illegals(s, fill_char='\u3000'):
    for i in range(len(s)):
        if s[i] in illegal_chars and not s[i] in japanese_chars:
            s = s[:i] + fill_char + s[i+1:]
    return s
    
def to_hant(c):
    try:
        return langconv.Converter('zh-hant').convert(c)
    except:
        return c
    
def convert_kanji(s):
    for i in range(len(s)):
        if not s[i] in kanjis and not s[i] in japanese_chars:
            if s[i] in cnmap:
                s_ = cnmap[s[i]]
            else:
                s_ = to_hant(s[i])
            s = s[:i] + s_ + s[i+1:]
    return s

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
    
def filtrate(s, threshold=20, dummy_mark='NUKAGAWA'):
    s1 = filtrate_bracket(s)
    s2 = filtrate_illegals(s1)
    s3 = convert_kanji(s2)
    #if len(s3.split('\n')) <= threshold: return dummy_mark
    jpchars = 0
    for i in s3:
        if i in japanese_chars:
            jpchars += 1
    if jpchars < threshold: return dummy_mark
    return s3
    
def filtrate_file():
    lists = os.listdir(lrc_path)
    for i in tqdm(range(len(lists)), ncols=32, desc='LFIT'):#Lyrics Filtration
        s_dir = lrc_path + lists[i] + '/'
        c_dir = lrc_conv_path + lists[i] + '/'
        try:
            os.mkdir(c_dir)
        except FileExistsError:
            pass
        for file in tqdm(os.listdir(s_dir)[0:], ncols=32, desc='LWFI'):#Lyric-Wise Filtrarion
            try:
                f = open(s_dir + file, 'r', encoding='utf-8')
                s = f.read()
                f.close()
                s = filtrate(s)
                if not s == 'NUKAGAWA':
                    f = open(c_dir + file, 'w', encoding='utf-8')
                    f.write(s)
                    f.close()
            except:
                pass
        f = open('proc_lrc.log', 'w') 
        f.writelines([str(i)])
        f.close()
        
if __name__ == '__main__':
    '''
    d = 'test.txt'
    with open(d, encoding='utf-8') as f: s = f.read()
    s_ = filtrate(s)
    s_ = separate(s_)
    '''
    filtrate_file()
