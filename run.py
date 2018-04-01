import lstm
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy.random as rnd
import numpy as np
from keras.callbacks import EarlyStopping


def plot_variables(predicted_data, true_data):

    fig = plt.figure(facecolor='white')
    figSupTitle = 'Geoloc prediction on '+test_scenario+'\n'
    figSupTitle += 'training set=XXXXXXXXXXX - trained on test set:'+str(train_on_test)+' - epochs='+str(epochs)+' - sequence length='+str(seq_len)+' - timestep='+str(timestep)+'s'
    fig.suptitle(figSupTitle, fontsize=13)
    yLabels = ['Target distance (km)', 'a (m)', 'b (m)', 'Bearing (deg)'];

    for i in range(4):
        ax = fig.add_subplot(221+i)
        ax.plot(true_data[:,i], label='True data', linestyle='solid', color='b')
        x_predicted = list(range(seq_len, seq_len+predicted_data.shape[0]))
        plt.plot(x_predicted, predicted_data[:,i], label='Prediction', linestyle='solid', color='r')
        #plt.title(titles[i])
        plt.ylabel(yLabels[i])
        plt.xlabel('Index de l\'ellipse')
        plt.legend()

    plt.show()


def plot_ellipses():
    NUM = 250

    ells = [Ellipse(xy=rnd.rand(2)*10, width=rnd.rand(), height=rnd.rand(), angle=rnd.rand()*360) for i in range(NUM)]
    fig = plt.figure(0)
    ax = fig.add_subplot(111, aspect='equal')
    for e in ells:
        ax.add_artist(e)
        e.set_clip_box(ax.bbox)
        e.set_alpha(rnd.rand())
        e.set_facecolor(rnd.rand(3))

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    plt.show()


def plot_variables_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    #Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.show()



#Main Run Thread
if __name__=='__main__':
    # plot_ellipses()
    

    global_start_time = time.time()
    global seq_len # window size
    seq_len = 5
    global timestep # time between two ellipses used for sampling
    timestep = 3.0 #1.1 # in seconds
    global epochs
    epochs  = 7
    global test_scenario
    test_scenario='XXXXXXXXXXX'
    global train_on_test
    train_on_test = False
    global normalise
    normalise = True

    print('> Loading data... ')
    wholeData, windows, x_train, y_train, x_test, y_test = lstm.load_scenarios(seq_len, timestep, test_scenario, train_on_test, normalise)
    

    print('> Data Loaded. Compiling...')
    model = lstm.build_model([5, 50, 200, 4])

    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    # model.fit(X, y, validation_split=0.2, callbacks=[early_stopping])

    model.fit(
        x_train,
        y_train,
        shuffle=True,
        batch_size=512,
        nb_epoch=epochs,
        validation_split=0.2,
        callbacks=[early_stopping])

    print('Training duration (s) : ', time.time() - global_start_time)
    

    
    #predicted = lstm.predict_point_by_point(model, x_test)
    predicted = lstm.predict_sequence_full(model, x_test, wholeData[test_scenario], seq_len)

    predicted = lstm.denormalize(predicted)
    x_test = lstm.denormalize(x_test)
    y_test = lstm.denormalize(y_test)
    true_data = np.concatenate((x_test[0, :, :-1], y_test))

    plot_variables(predicted, true_data)


    #predictions = lstm.predict_sequences_multiple(model, x_test, seq_len, 10)
    #plot_variables_multiple(predictions, y_test, 10)
    