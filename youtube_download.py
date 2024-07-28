
from pytubefix import YouTube, Playlist
from pytubefix.cli import on_progress
 
url = "https://www.youtube.com/playlist?list=PLunILarQwxnnfpqfSzQVbj4JWPAn1ByiF"
 
playlist_urls = [
    'https://www.youtube.com/playlist?list=PLBZBJbE_rGRV8D7XZ08LK6z-4zPoWzu5H',
    'https://www.youtube.com/playlist?list=PLdo5W4Nhv31bbKJzrsKfMpo_grxuLl8LU',
    'https://www.youtube.com/playlist?list=PLBlnK6fEyqRiVhbXDGLXDk_OQAeuVcp2O',
    'https://www.youtube.com/playlist?list=PLE7DDD91010BC51F8',
    'https://www.youtube.com/playlist?list=PL49CF3715CB9EF31D'
]

output_path = "/home/tiny_ling/projects/my_mistral/audios"


for i,url in enumerate(playlist_urls):
    pl = Playlist(url)
    for j,video in enumerate(pl.videos):
        ys = video.streams.get_audio_only()
        ys.download(output_path,mp3=True) # pass the parameter mp3=True to save in .mp3    