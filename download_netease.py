# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 18:15:06 2018

@author: Zhao
"""

import pickle
import requests
from tqdm import tqdm

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

dwnl_dir = './netease_search/{}.mp3'
order = quick_load('final_order.pkl')
ids = []
for i in order[:300]: ids.append(i[0])
raw_headers = {
                'Accept': 'text/html, application/xhtml+xml, application/xml; q=0.9, */*; q=0.8',
                'Accept-Encoding': 'gzip, deflate, sdch',
                'Accept-Language': 'zh-CN,zh;q=0.8,en;q=0.6',
                'Cache-Control': 'no-cache',
                'Cookie':'_ntes_nnid=73794490c88b2790756a23cb36d25ec1,1507099821594; _ntes_nuid=73794490c88b2790756a23cb36d25ec1; _ngd_tid=LtmNY2JGJkw6wR3HF%2FpG2bY%2BtHhQDmOj; usertrack=c+xxC1nazueHBQJiCi7XAg==; JSESSIONID-WYYY=sJg6dw45PFKjn0VD2OuD0mzqC03xb3CnU3h4ac43kp7r9q9GJos%2BFDVyZmeGtz%5CHciN66cY5KAEW6jlHT%5COv0qzP8T3O3R5cq28%2BXJ3rc%2BkqsI4Y%2BrJIwZczDZGlvq225U%5CNWBP0iEjTnfdUG21swAhZA%5CfX29F4s9M6tz2EK7%2FESIpW%3A1507612773856; _iuqxldmzr_=32; MUSIC_U=e58d5af1daeedff199dcb9d14e06692f2db7395809fd3b393c0d6d53e13de2f484b4ab9877ef4e4ca1595168b12a45da86e425b9057634fc; __remember_me=true; __csrf=63e549f853ed105c4590d6fe622fb4f6',
                'Host': 'music.163.com',
                'Referer': 'http://music.163.com/',
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/64.0.3282.140 Safari/537.36 Edge/17.17134'
              }
illegal_chars = ['/', '\\', '"', ':', '<', '>', '?', '|', '*']

def file_legalize(s):
    for i in illegal_chars:
        s = s.replace(i, '-')
    return s

def get_page(url):
    r = requests.get(url, headers=raw_headers)
    return r.content.decode()

def get_music_info(m_id):
    orig = get_page('http://music.163.com/song?id={}'.format(m_id))
    orig = orig.split('data-res-action="share"\ndata-res-name="')[1]
    orig = orig.split('"')[:3]
    orig = orig[0] + ' - ' + orig[2]
    return orig

def generate_names(ids):
    ans = []
    for i in tqdm(ids, ncols=32, desc='GENE'):
        ans.append(get_music_info(i))
    return ans

def download_file(url, save_path, chunk_size=1024):
    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                
def download_musics(names=['test'], ids=['1310321129']):
    for i in tqdm(range(27, len(ids)), ncols=32, desc='DWNL'):
        n_, i_ = file_legalize(names[i]), ids[i]
        #print('Downloading({}/{}) CUR: {}[{}]...'.format(i, len(ids), n_, i_))
        download_file('http://music.163.com/song/media/outer/url?id={}.mp3'.format(i_), dwnl_dir.format(n_))
        f = open('download_ne.log', 'w') 
        f.writelines([str(i)])
        f.close()

if __name__ == '__main__':
    names = generate_names(ids)
    download_musics(names, ids)