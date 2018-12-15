import os
import csv
import ast
import numpy as np
import librosa
import imageio
import pickle
import warnings

warnings.filterwarnings("ignore")

songs = []
metadata = []
genres = [2, 10, 12, 15, 17, 21, 38, 1235]
genreLabels = ["International", "Pop", "Rock", "Electronic", "Folk", "Hip-Hop", "Experimental", "Instrumental"]

def save(obj, val):
    with open('Data/TrainingData'+str(val)+'.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def oneHotEncode(x):
    encoded = np.zeros((len(x), 8))
    for idx, val in enumerate(x):
        encoded[idx][val] = 1
    return encoded

def normalize(x):
    min_val = np.min(x)
    max_val = np.max(x)
    x = (x-min_val) / (max_val-min_val)
    return x

def getMel(filename, genre):

    while(len(filename) < 6):
        filename = '0'+filename

    audio_path = 'Data/Songs/'+filename+'.mp3'

    sr = 8000
    numMels = 32

    y, _ = librosa.load(audio_path, sr=sr)

    # Let's make and display a mel-scaled power (energy-squared) spectrogram
    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=numMels)

    # Convert to log scale (dB). We'll use the peak power (max) as reference.
    log_S = librosa.power_to_db(S, ref=np.max)
    log_S = normalize(log_S)

    imageio.imwrite('Data/Datagrams/'+filename+'.png', log_S)
    return filename, {'data' : log_S, 'genre' : genre}

def loadData():

    for _, _, files in os.walk("./Data/Songs"):  
        for filename in files:
            songs.append(filename.split(".")[0].lstrip('0'))

    songs.sort(key=int)

    with open('Data/Metadata/tracks.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if(row[0] in songs):
                data = ast.literal_eval(row[41])
                final = 0
                for entry in data:
                    if(entry in genres):
                        final = genres.index(entry)
                metadata.append(final)

    encodedMetadata = oneHotEncode(metadata)

    print(len(songs))
    print(encodedMetadata.shape)

    #percentage = 0
    percentage = 90
    #dataCount = 0
    dataCount = 9
    finalDict = {}

    for i in range(7200, len(songs)):
        if(i%80 == 0):
            print(percentage,"%")
            percentage += 1

        if(i%800 == 0 and i > 7200):
            dataCount += 1
            save(finalDict, dataCount)
            finalDict = {}
            print("Saved")

        entryKey, entryVal = getMel(songs[i], encodedMetadata[i])
        finalDict.update({entryKey : entryVal})

    dataCount += 1
    save(finalDict, dataCount)
    finalDict = {}
    print("Saved")

loadData()