# Desired size of development set / total samples
TESTING_RATIO = .8

CATEGORIES = ('happy', 'sad', 'relaxed', 'angry')
KEYWORDS = {
    'happy': ['happy', 'joy', 'awesome', 'party', 'city', 'love', 'sex', 'summer', 'spring', 'pop', 'yay', 'fun', 'club', 'nightlife', 'dance', 'romance', 'motivational', 'electro', 'beach', 'radio', 'beautiful', 'pretty', 'christmas', 'disco', 'birthday', 'edm', 'energetic', 'festival', 'inspirational', 'jog', 'uplifting', 'training', 'happiness'],
    'sad': ['sad', 'blues', 'breakup', 'ache', 'wish', 'die', 'alone', 'drowning', 'reminisce', 'funeral', 'dead', 'dark', 'broken', 'remember', 'forget', 'forgot', 'break', 'hope', 'lone', 'depressed', 'depression'],
    'relaxed': ['relax', 'chill', 'home', 'study', 'night', 'evening', 'high', 'weed', 'reggae', 'jazz', 'piano', 'winter', 'star', 'meditatcalm', 'soft', 'dream', 'work', 'classical', 'rap', 'hiphop', 'hip-hop', 'hip' 'hop', 'late', 'fall', 'autumn', 'sleep', 'asmr', 'country', 'indie', 'tranquil'],
    'angry': ['fuck', 'bitch', 'angry', 'mad', 'pissed', 'shit', 'rock', 'metal', 'death', 'gym', 'workout', 'hell', 'demon', 'punk', 'devil'],
}

CATEGORIES_ = ('positive', 'negative')
KEYWORDS_ = {
    'positive': ['happy', 'joy', 'awesome', 'party', 'city', 'love', 'sex', 'summer', 'spring', 'pop', 'yay', 'fun', 'club', 'nightlife', 'dance', 'romance', 'motivational', 'electro', 'beach', 'radio', 'beautiful', 'pretty', 'christmas', 'disco', 'birthday', 'edm', 'energetic', 'festival', 'inspirational', 'jog', 'uplifting', 'training', 'happiness', 'relax', 'chill', 'home', 'study', 'night', 'evening', 'high', 'weed', 'reggae', 'jazz', 'piano', 'winter', 'star', 'meditatcalm', 'soft', 'dream', 'work', 'classical', 'rap', 'hiphop', 'hip-hop', 'hip' 'hop', 'late', 'fall', 'autumn', 'sleep', 'asmr', 'country', 'indie', 'tranquil'],
    'negative': ['sad', 'blues', 'breakup', 'ache', 'wish', 'die', 'alone', 'drowning', 'reminisce', 'funeral', 'dead', 'dark', 'broken', 'remember', 'forget', 'forgot', 'break', 'hope', 'lone', 'depressed', 'depression', 'fuck', 'bitch', 'angry', 'mad', 'pissed', 'shit', 'rock', 'metal', 'death', 'gym', 'workout', 'hell', 'demon', 'punk', 'devil'],
}
