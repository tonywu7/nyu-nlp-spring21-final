import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import math


# done for each category in training set

# songLyrics = an array of words in a song's lyrics
# allSongLyrics = an array of len(songs) containing the lyrics of each song
# allSongTitles = an array of song titles
# each allSongTitles[i] has to match allSongLyrics[i]

### get TF value for one song ###
def getTF(songLyrics):
    # creating a dict where the keys are each of the words in the song
    tfSong = dict.fromkeys(songLyrics,0)

    # counting the frequency of each word in the song
    for word in songLyrics:
        tfSong[word]+=1

    # dividing the frequency of each word by the length of the song
    tfSong = [tfSong[word]/len(songLyrics) for word in tfSong])

    return tfSong


def getAllTF(allSongLyrics,allSongTitles):
    tfSongs = {}
    for i,songLyrics in enumerate(allSongLyrics):
        songTitle = allSongTitles[i]

        tfSongs[songTitle] = getTF(songLyrics)

    return tfSongs



########### idf #############

# wordList is a list that contains all the words in all the lyrics

def getIDF(wordList, allSongLyrics):
    idfSongs = dict.fromkeys(wordList)

    for word in wordList:
        for song in allSongLyrics:
            if word in song:
                idfSongs[word]+=1

        idfSongs[word] = math.log(len(allSongLyrics)/idfSongs[word])
    
    return idfSongs


############ TF IDF ############


def getTFIDF(idfSongs, tfSongs, allSongLyrics, allSongTitles):
    tfidfSongs = {}
    for i,song in enumerate(allSongLyrics):
        title = allSongTitles[i]
        tfidfSongs[title] = dict.fromkeys(tfSongs[title],0)
        for word in song:
            tfidfSongs[title][word] = idfSongs[word] * tfSongs[title][word]
    
    return tfidfSongs
    


############ TF IDF category ###########

## average of TF IDF vectors of all songs in the category 
def getTFIDF_category(tfidfCategory, wordList):
    categoryVector = dict.fromkeys(wordList)
    for vec in tfidfCategory:
        for word in vec:
            caregoryVector[word]+=1
    
    length = len(tfidfCategory)

    for word in categoryVector:
        categoryVector[word] = categoryVector[word]/length
    
    return categoryVector


######## cosine similarity ##########

def getSimilarity(testSongVec, songTitle, categoryVectors):

    catNames = ["happy","relaxed","sad","angry"]
    songSimilarity = dict.fromkeys(catNames)

    for i,vec in enumerate(categoryVectors):
        categoryName = catNames[i]
        catVec = []
        for word in testSongVec:
            if word in vec:
                catVec.append(vec[word])
            else:
                catVec.append(0.0)
    
        numerator = 0
        docDenominator  = 0
        queryDenominator = 0

        for i,word in enumerate(testSongVec):
            numerator += (catVec[i]*testSongVec[word])
            catDenominator += (catVec[i]**2)
            songDenominator += (testSongVec[word]**2)

        denominator = math.sqrt(catDenominator * songDenominator)

        if denominator!=0:
            songSimilarity[categoryName] = numerator/denominator
        else:
            songSimilarity[categoryName] = 0

    songSimilarity = sorted(songSimilarity[categoryName], key=songSimilarity[categoryName].get, reverse=True)

    return songSimilarity


    

