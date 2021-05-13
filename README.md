# NYU CS NLP 2021 Final Project

### Setup

```
python3 -m pip install -r requirements.txt
```

To confirm you have all requirements installed, try printing the program versions:

```
python3 -m song_classifier version
```

**Database:** [download](https://drive.google.com/drive/folders/1xCDUAUXCrkTXwtCYHiW3RYZZuj8_W8ZY?usp=sharing)

Place the `index.db` file at `{repo_root}/instance/index.db`, then specify `-a instance`
when running the commands.

### Running tests

```
python3 -m song_classifier -a <database-dir> run-test
```

Use the `--help` option for descriptions of configurable parameters.

### Playground

To run the program in REPL mode:

```
python3 -i repl.py <database-dir>
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
