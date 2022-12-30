"""Load and first process of the dataset.
Source: https://ataspinar.com/2018/12/21/a-guide-for-using-the-wavelet-transform-in-machine-learning/
"""
import numpy as np
import os

def read_signals_ucihar(filename):
    # Process the data in the .txt files into lists.
    # Return a list in which every item is a list. The lists inside
    # represents each line. Then, each line-list is broken
    # into float elements representing every data in a line.
    with open(filename, 'r') as fp:
        data = fp.read().splitlines()
        data = map(lambda x: x.rstrip().lstrip().split(), data)
        data = [list(map(float, line)) for line in data]
    
    return data

def read_labels_ucihar(filename):        
    with open(filename, 'r') as fp:
        activities = fp.read().splitlines()
        activities = list(map(int, activities))
    return activities

def load_ucihar_data(folder):
    # Load the UCI HAR dataset from the specified folder path.
    
    # Setting the paths of the training and test folders.
    train_folder = folder + 'train/InertialSignals/'
    test_folder = folder + 'test/InertialSignals/'
    labelfile_train = folder + 'train/y_train.txt'
    labelfile_test = folder + 'test/y_test.txt'

    # List with data from all the signals.
    train_signals, test_signals = [], []

    # From the list of directories in the train_folder path.
    for input_file in os.listdir(train_folder):
        signal = read_signals_ucihar(train_folder + input_file)
        train_signals.append(signal)
    train_signals = np.transpose(np.array(train_signals), (1, 2, 0))

    # From the list of directories in the train_folder path.
    for input_file in os.listdir(test_folder):
        signal = read_signals_ucihar(test_folder + input_file)
        test_signals.append(signal)
    test_signals = np.transpose(np.array(test_signals), (1, 2, 0))
    
    train_labels = read_labels_ucihar(labelfile_train)
    test_labels = read_labels_ucihar(labelfile_test)
    return train_signals, train_labels, test_signals, test_labels

if __name__ == "__main__":
    folder_ucihar = './data/UCI_HAR/' 
    train_signals_ucihar, train_labels_ucihar, \
    test_signals_ucihar, test_labels_ucihar = load_ucihar_data(folder_ucihar)
