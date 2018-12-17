# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 16:53:57 2018

@author: BillStark001
"""

from tqdm import tqdm, tqdm_gui
import requests
import json
import re
import os
import numpy as np

illegal_chars = ['/', '\\', '"', ':', '<', '>', '?', '|', '*']
def file_legalize(s):
    for i in illegal_chars:
        s = s.replace(i, '-')
    return s
 
def generate_headers():
    user_agent_list = [
        "Mozilla/5.0 (Macintosh; U; Intel Mac OS X 10_6_8; en-us) AppleWebKit/534.50 (KHTML, like Gecko) Version/5.1 Safari/534.50",
        "Mozilla/5.0 (Windows; U; Windows NT 6.1; en-us) AppleWebKit/534.50 (KHTML, like Gecko) Version/5.1 Safari/534.50",
        "Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Trident/5.0;",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.6; rv:2.0.1) Gecko/20100101 Firefox/4.0.1",
        "Mozilla/5.0 (Windows NT 6.1; rv:2.0.1) Gecko/20100101 Firefox/4.0.1",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_0) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.56 Safari/535.11",
        "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.1 (KHTML, like Gecko) Chrome/22.0.1207.1 Safari/537.1",
        "Mozilla/5.0 (X11; CrOS i686 2268.111.0) AppleWebKit/536.11 (KHTML, like Gecko) Chrome/20.0.1132.57 Safari/536.11",
        "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.6 (KHTML, like Gecko) Chrome/20.0.1092.0 Safari/536.6",
        "Mozilla/5.0 (Windows NT 6.2) AppleWebKit/536.6 (KHTML, like Gecko) Chrome/20.0.1090.0 Safari/536.6",
        "Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/537.1 (KHTML, like Gecko) Chrome/19.77.34.5 Safari/537.1",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/536.5 (KHTML, like Gecko) Chrome/19.0.1084.9 Safari/536.5",
        "Mozilla/5.0 (Windows NT 6.0) AppleWebKit/536.5 (KHTML, like Gecko) Chrome/19.0.1084.36 Safari/536.5",
        "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1063.0 Safari/536.3",
        "Mozilla/5.0 (Windows NT 5.1) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1063.0 Safari/536.3",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_8_0) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1063.0 Safari/536.3",
        "Mozilla/5.0 (Windows NT 6.2) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1062.0 Safari/536.3",
        "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1062.0 Safari/536.3",
        "Mozilla/5.0 (Windows NT 6.2) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1061.1 Safari/536.3",
        "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1061.1 Safari/536.3",
        "Mozilla/5.0 (Windows NT 6.1) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1061.1 Safari/536.3",
        "Mozilla/5.0 (Windows NT 6.2) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1061.0 Safari/536.3",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/535.24 (KHTML, like Gecko) Chrome/19.0.1055.1 Safari/535.24",
        "Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/535.24 (KHTML, like Gecko) Chrome/19.0.1055.1 Safari/535.24"
    ]
    
    x = np.random.randint(len(user_agent_list))
    user_agent = user_agent_list[x]
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
    return raw_headers
    
def get_page(url):
    r = requests.get(url, headers=generate_headers())
    return r.content.decode()
 
def get_lyrics(music_id):
    url = 'http://music.163.com/api/song/lyric?id={}&lv=1&kv=1&tv=-1'.format(music_id)
    lrc = json.loads(get_page(url))
    try:
        data=lrc['lrc']['lyric']
    except:
        return('NUKAGAWA')
        
    #利用正则表达式去掉歌词前面的时间戳
    re_lyrics=re.compile(r"\[.*\]")
    #将data字符串中的re_lyrcs替换成空
    lrc=re.sub(re_lyrics,"",data)
    lrc=lrc.strip()
    
    return lrc
 
def get_music_infos(url, s=None):
    if isinstance(s, str): html = s
    else: html = get_page(url)
    
    pat1 = r'<ul class="f-hide"><li><a href="/song\?id=\d*?">.*</a></li>'
    pat = re.compile(pat1,re.S)
    l = pat.findall(html)
    result = l[0]
    
    pat2 = r'<li><a href="/song\?id=\d*?">(.*?)</a></li>'#name
    pat3 = r'<li><a href="/song\?id=(\d*?)">.*?</a></li>'#id
    pat = re.compile(pat2)
    names = pat.findall(result)
    pat = re.compile(pat3)
    ids = pat.findall(result)
 
    return names,ids
    
def get_playlist_infos_in_range(url):
    html = str(get_page(url))
    names = []
    ids = []
    
    html = html.split('<ul class=')[1]
    html = html.split('</ul>')[0]
    
    html = html.split('<a title="')
    for i in range(len(html)):
        if not i % 3 == 1: continue
        cur = html[i].split('" class="msk">')[0]
        cur = cur.split('" href="/playlist?id=')
        names.append(cur[0])
        ids.append(cur[1])
    return names, ids
    
def get_playlist_infos(url, page_contains=35, max_index=1500):
    names = []
    ids = []
    for i in tqdm(range(0, max_index, page_contains), desc='LISC', ncols=64):#List Scrap
        n_, i_ = get_playlist_infos_in_range(url.format(i))
        names += n_
        ids += i_
    return names, ids
        
 
def write_to_file(names, ids, path):
    for i in tqdm(range(len(ids)), desc='LRCW', ncols=64):#LRC Write
        lrc = get_lyrics(ids[i].split('_')[0])
        if lrc == 'NUKAGAWA': continue
        filepath = os.path.join(path, '%s_'%ids[i] + names[i] + ".txt")
        try:
            f = open(filepath, 'w', encoding='utf-8')
            f.write(lrc)
            f.close()
        except:
            filepath = os.path.join(path, '%s_'%ids[i] + file_legalize(names[i]) + ".txt")
            try:
                f = open(filepath, 'w', encoding='utf-8')
                f.write(lrc)
                f.close()
            except:
                print(names[i])
 
def write_lyrics_by_playlists(names, ids, path):
    url='http://music.163.com/playlist?id={}'
    for i in tqdm(range(0, len(names)), desc='LWDL', ncols=64):#Download lyrics list-wisely
        n_, i_ = get_music_infos(url.format(ids[i]))
        cur_dir = path + '%s_'%ids[i] + names[i] + '/'
        try:
            os.mkdir(cur_dir)
        except OSError:
            cur_dir = path + '%s_'%ids[i] + file_legalize(names[i]) + '/'
            try:
                os.mkdir(cur_dir)
            except FileExistsError:
                pass
        except:
            print(cur_dir)
        write_to_file(n_, i_, cur_dir)
        f = open('proc.log', 'w') 
        f.writelines([str(i)])
        f.close()
        
        
if __name__=="__main__":
    path=r"./lyrics/"
    '''
    url='http://music.163.com/playlist?id=2363292213'#'http://music.163.com/artist?id=10559'
    music_name, music_id = get_music_infos(url)
    write_to_file(music_name, music_id, path)
    '''
    '''
    try:
        assert names, ids
    except:
        url = 'http://music.163.com/discover/playlist/?cat=%E6%97%A5%E8%AF%AD&offset={}'
        names, ids = get_playlist_infos(url)
    write_lyrics_by_playlists(names, ids, path)
    '''