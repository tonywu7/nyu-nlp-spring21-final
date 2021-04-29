# NYU CS NLP 2021 Final Project

## For collaborators

**Lyrics database**

- Download the latest version of database (from Discord) 
- Rename the file to `index.db`
- Place the file in a directory named `instance` at the top-level (next to the
`song_classifier` folder and the `README.md` file, etc.)

**Playground**

To run the program in REPL mode:

```
python3 -i repl.py instance
```

Then you can access the database through the `db` variable:

```py
>>> db.total_songs()
49345
>>> db.total_playlists()
58912
>>> song = db.get_song_by_title('Get Lucky')
>>> song.playlists[0].title
'Fun Fun'
>>> [song.title for song in db.song_title_search('bohemian')]
['bohemian rhapsody', 'Bohemian Rhapsody - Remastered 2011']
>>> len(db.playlist_title_search('happy'))
178
```
