# Mimigaraoboeru

It's just a simple content-based recommendation system.
I built it for a flashing idea that if I can find Japanese songs with lots of new words, good melodies and good lyrics.
See 1.png in detail.

# Required platform, libs and datasets

Python 3.6.*
Mecab 0.996
mecab-python
ffmpeg
tensorflow
keras

FMA Dataset

# Usage
Launch word_process.py to find all new words in words.csv (JLPT-N1 words list).
Launch craw_lyrics to craw song lyrics on NetEase Cloud Music's server; put favourite song lyrics in ./lyrics/0_fav/ and name them like <songid>_<number_of_times_heard——recently>_<songname>.txt (e.g. 560108_70_MEMORIA.txt)
Launch filtrate_lyrics to filtrate non-Japanese characters in all the lyric files.
Launch theme_bias.py to get the theme bias(the likelihood that the lyrics of the song will fit you) of each song and dump it to theme_bias.pkl .
Launch music_download.py to download songs that have high theme bias(too large if download all of them). * the variable t_ represents the number it will download.
Launch wave_render.py to do STFT(Short-Time Fourier Transmission) on downloaded .mp3 files.
Launch music_bias.py to predict the feature of each .mp3 file and get the music bias by calculating their cosine similarities.
Launch theme_bias.py to get the final order due to theme bias, music bias and lyric importance.
Launch download_netease.py to download the top rated songs of fianl_order.pkl .

** You may change some codes to finish all of those steps. Create an issue if you need help. I may do engineering optimization if I have enough time.
