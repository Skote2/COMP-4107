import numpy as numpy
import pickle

def load():
    with open('Data/TestingData.pkl', 'rb') as f:
        return pickle.load(f)

test = load()

print(test)
print(len(test))