# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 23:26:03 2018

@author: Zhao
"""

import os
import requests
from tqdm import tqdm
import pickle

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

bias, lyric_importance, theme_bias = quick_load('theme_bias.pkl', ([], [], []))
netease_api = 'http://music.163.com/song/media/outer/url?id={}.mp3'
save_path = './musics/{}_{}.mp3'

def download_file(url, save_path, chunk_size=1024):
    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                
def download_musics(names=['test'], ids=['1310321129']):
    for i in tqdm(range(0, len(ids)), ncols=32, desc='DWNL'):
        n_, i_ = names[i], ids[i]
        #print('Downloading({}/{}) CUR: {}[{}]...'.format(i, len(ids), n_, i_))
        download_file(netease_api.format(i_), save_path.format(i_, n_))
        f = open('download.log', 'w') 
        f.writelines([str(i)])
        f.close()
        
def get_music_names(path):
    ans = {}
    for i in os.listdir(path):
        for j in os.listdir(path + '/' + i):
            j = j.split('_')
            id = int(j[0])
            n = ''
            for k in j[1:]: n += k
            ans[id] = n[:-4]
    return ans

if __name__ == "__main__":
    pass
    music_names = get_music_names('./lyrics/')
    names = []
    #'''
    ids = {}
    t_ = 1000
    for i, j, k in zip(bias[:t_], lyric_importance[:t_], theme_bias[:t_]):
        ids[i[0]] = 0
        ids[j[0]] = 0
        ids[k[0]] = 0
    ids = list(ids)
    #ids.sort()
    #'''
    '''
    ids = os.listdir('./lyrics/0_fav/')
    for i in range(len(ids)):
        ids[i] = int(ids[i].split('_')[0])
    '''
    for i in ids:
        names.append(music_names[i])
    download_musics(names, ids)