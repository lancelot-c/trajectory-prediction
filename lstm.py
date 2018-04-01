import os
from os import listdir
from os.path import isfile, isdir, join
import time
import warnings
import math
import numpy as np
import pandas
from numpy import newaxis
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras import optimizers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Hide messy TensorFlow warnings
warnings.filterwarnings("ignore") #Hide messy Numpy warnings



def load_target(target_filename, timestep):
    ellipses = pandas.read_csv(target_filename, sep=",", header=None, usecols=(0,1,13,14,15))
    ellipses = ellipses.ix[1:]; # Remove headers
    ellipses = ellipses.as_matrix(); # Convert to numpy matrix
    ellipses = ellipses.astype(float) # Convert to float
    ellipses = ellipses[ellipses.all(axis=1), :] # Remove rows containing a zero

    lastTime = -timestep # Because the first occurence of #1 needs to be true
    sampled_ellipses = []
    for ell in ellipses:
        
        # Sampling
        if ell[0] >= lastTime+timestep: #1
            sampled_ellipses.append(np.append(ell, 0))
            lastTime = ell[0]

            # Adding trajectory change
            # N.B. : to get the trajectory change at time t we need the sample at time t+1
            if len(sampled_ellipses) >= 2:
                timeT0 = sampled_ellipses[-2][0]
                cap = normalized_cap(timeT0, lastTime)
                sampled_ellipses[-2][-1] = cap


    # To be fixed, last timestep is useless for the input but useful for the output
    sampled_ellipses = np.array(sampled_ellipses[:-1]) # delete last timestep because we don't know its cap
    # print('\n\sampled_ellipses for '+target_filename+'\n', sampled_ellipses)
    return sampled_ellipses


def load_scenarios(seq_len, timestep, test_scenario, train_on_test=False, normalise=True):

    # Initializing
    sequence_length = seq_len + 1 # +1 because the y is also contained in the window
    whole_data = {}
    windows = {}
    x_train = []
    y_train = []
    ucavLoaded = False
    exploit_folder = "XXXXXXXXXXX"
    
    i = 0
    while(isdir(exploit_folder+"/S"+str(i))):

        scenario = "S"+str(i)
        geoloc_file = exploit_folder+"/"+scenario+"/XXXXXXXXXXX.csv"
        ucav_file = exploit_folder+"/"+scenario+"/XXXXXXXXXXX.csv"

        # Extracting ucav infos
        if ucavLoaded is not True and isfile(ucav_file):
            ucavLoaded = True
            global ucavInfos
            ucavInfos = pandas.read_csv(ucav_file, sep=",", usecols=['XXXXXXXXXXX'], squeeze=True)
            ucavInfos = ucavInfos.as_matrix() # Convert to numpy matrix
            #print('UCAV infos loaded using scenario ', scenario)


        # Extracting geoloc infos
        if (isfile(geoloc_file)):
            sampled_ellipses = load_target(geoloc_file, timestep)
                
            # Add scenario only if it contains enough data to create a window
            if len(sampled_ellipses) >= sequence_length:
                whole_data[scenario] = sampled_ellipses
                #print('Scenario ', scenario, ' loaded')
            else:
                print('WARNING : Scenario ', scenario, ' is irrelevant')
        else:
            print('WARNING : Scenario ', scenario, ' is missing')

        i += 1




    # Normalization
    if normalise:
        whole_data = normalize(whole_data)


    # Generate windows
    for target in whole_data:
        windows[target] = []

        for index in range(len(whole_data[target]) - sequence_length):
            window = whole_data[target][index: index + sequence_length]
            windows[target].append(window) # append window

        windows[target] = np.array(windows[target])



    # Generate training and test sets
    for target in windows:
        if target == test_scenario:
            x_test = windows[target][:, :-1, 1:]
            y_test = windows[target][:, -1, 1:-1]
            if train_on_test == False:
                continue

        for w in windows[target]:
            x_train.append(w[:-1, 1:])  # param1:get all but last timestep, param2:don't get time
            y_train.append(w[-1, 1:-1]) # param1:only get last timestep, param2:don't get time and cap

    x_train = np.array(x_train)
    y_train = np.array(y_train)


    # Randomize training set
    permutation = np.random.permutation(x_train.shape[0])
    x_train = x_train[permutation]
    y_train = y_train[permutation]

    return [whole_data, windows, x_train, y_train, x_test, y_test]



# Normalize every column except the first (time)
def normalize(whole_data):
    # Concatenate all the data
    data = []
    for target_data in whole_data.values():
        for timestep in target_data:
            data.append(timestep)
    data = np.array(data)

    # Compute mean and std
    global mean, std
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)

    # Leave time column unchanged
    mean[0] = 0
    std[0] = 1

    # Normalization
    for target in whole_data:
        whole_data[target] = (whole_data[target] - mean) / std

    return whole_data


# Denormalize windows in respect with the mean and std computed in normalize()
def denormalize(windows):
    if windows.shape[-1] == 5: #1
        return (windows * std[1:]) + mean[1:] # Remove mean and std of time
    if windows.shape[-1] == 4: #2
        return (windows * std[1:-1]) + mean[1:-1] # Remove mean and std of time and trajectory


def build_model(layers):
    model = Sequential()

    model.add(LSTM(
        input_dim=layers[0],
        output_dim=layers[1],
        return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(
        layers[2],
        return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(
        output_dim=layers[3]))
    #model.add(Activation("linear"))

    start = time.time()

    #rmsprop = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    model.compile(loss="mse", optimizer="rmsprop")
    print("> Compilation Time : ", time.time() - start)
    return model


#Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
def predict_point_by_point(model, windows):
    return model.predict(windows)


#Shift the window by 1 new prediction each time, re-run predictions on new window
def predict_sequence_full(model, windows, whole_data, window_size):
    curr_frame = windows[0]
    predicted = []

    for i in range(len(windows)):
        prediction = model.predict(curr_frame[newaxis,:,:])[0,:]
        predicted.append(prediction)
        cap = whole_data[window_size+i,-1]
        prediction = np.append(prediction, [cap])[newaxis,:]
        curr_frame = np.concatenate((curr_frame[1:,:], prediction))

    predicted = np.array(predicted)
    return predicted


#Predict sequence of 50 steps before shifting prediction run forward by 50 steps
def predict_sequences_multiple(model, data, window_size, prediction_len):
    prediction_seqs = []
    for i in range(int(len(data)/prediction_len)):
        curr_frame = data[i*prediction_len]
        predicted = []
        for j in range(prediction_len):
            predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
        prediction_seqs.append(predicted)
    return prediction_seqsb


def normalized_cap(timeT0, timeT1):
    capT0 = estimated_cap_at(timeT0)
    capT1 = estimated_cap_at(timeT1)
    return capS(capT0, capT1)


def estimated_cap_at(time):
    t1 = math.floor(time)
    t2 = t1+1
    delay = time-t1

    c = ucavInfos[t1] + delay * capS(ucavInfos[t1], ucavInfos[t2])
    c = (c+360)%360
    return c


# Symetric cap
def capS(capT0, capT1):
    c = capB(capT0, capT1)
    if c > 180:
        c -= 360
    return c


# T0-based cap
def capB(capT0, capT1):
    return (capT1-capT0+360)%360