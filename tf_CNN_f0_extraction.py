#Author: Xufang Zhao
#Function: Using CNN and spectrogram to detect and extract F0 values continuously
#Note: This CNN is a regression deep-learning neural networks to output continuous F0 value, not a DNN for classifications

import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.transforms import Bbox
import os
import wave # for reading audio files in .wav format
import pylab # for plotting waveforms and spectrograms
from pathlib import Path
from scipy import signal
from scipy.io import wavfile # wavfile reads wav files and returns the sample rate (in samples/sec) and data as numpy array
from sklearn.metrics import confusion_matrix # for confusion matrix plot 
import itertools # itertools is used for efficient looping 
from tqdm import tqdm # for progress bar

import random

from keras import models, optimizers, backend
from keras.layers import core, convolutional, pooling
from sklearn import model_selection

#Set paths to input and output data
INPUT_DIR = 'Directory of the Audio Data Corpus'
SPECTRAL_DIR = 'Storage of spectrogram files transferred from the Audio Data Corpus'
OUTPUT_DIR = 'CNN Output directory'
parent_list = os.listdir(INPUT_DIR)

#Noise file that mix with the clean Audio Data Corpus
#Note! The file name needs DOUBLE back slash
Noise_File = '.\Road_Noise\\Noise_Sample_16K.wav'
noise_data,noise_srate = librosa.load(Noise_File, mono=True, sr=None)

#The list of the audio files in the Audio Data Corpus
Test_Index_File = open('.\Test_Index.txt','w')


plot_WIDTH = 10
plot_HEIGHT = 4
size_NFFT = 1024
size_OVERLAP = 900

#x and y are CNN i/o data
x_training = np.empty([0, 38, 65, 4], dtype=np.float32)
y_training = np.empty([0, 32], dtype=np.float32)
x_validation = np.empty([0, 38, 65, 4], dtype=np.float32)
y_validation = np.empty([0, 32], dtype=np.float32)
x_prediction = np.empty([0, 38, 65, 4], dtype=np.float32)
y_prediction = np.empty([0, 32], dtype=np.float32)

recordings = list(range(len(parent_list)))
for i in range(len(parent_list)): 
    recordings[i] = INPUT_DIR + '\\' + parent_list[i]
    wav_data,sample_rate = librosa.load(recordings[i], mono=True, sr=None)
    f0, voiced_flag, voiced_probs = librosa.pyin(wav_data, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))

    fig = plt.figure(figsize=(plot_WIDTH,plot_HEIGHT))
    plt.subplots_adjust(hspace=0.5)
    
    plot_a = fig.add_subplot(1,10,1)
    plot_a.set_title(parent_list[i])
    plot_a.set_xlabel('sample rate * time')
    plot_a.set_ylabel('energy')
    plot_a.plot(wav_data)
    
    plot_b = fig.add_subplot(1,10,2)
    plot_b.set_title(parent_list[i]+" spectogram")
    plot_b.set_xlabel('Time')
    plot_b.set_ylabel('Frequency')
    plot_b.specgram(wav_data, NFFT=size_NFFT, Fs=sample_rate, noverlap=size_OVERLAP)
    
    spectral_png = SPECTRAL_DIR + '\\Full\\' + parent_list[i] + '_SpecFull.png'
    extent = plot_b.get_window_extent().transformed(plot_b.figure.dpi_scale_trans.inverted())
    plot_b.figure.savefig(spectral_png, bbox_inches=extent)   #Save spectogram for CNN

    xmin, xmax = plot_b.get_xlim()
    ymin, ymax = plot_b.get_ylim()
    #print(xmax)
    x0,x1 = xmin, xmax
    y0,y1 = ymin, 1000
    bbox = Bbox([[x0,y0],[x1,y1]])
    spectral_png = SPECTRAL_DIR + '\\Part\\' + parent_list[i] + '_SpecPart.png'
    bbox = bbox.transformed(plot_b.transData).transformed(plot_b.figure.dpi_scale_trans.inverted())
    plot_b.figure.savefig(spectral_png, bbox_inches=bbox)   #Save partial spectogram for CNN

    #Split wave data to 1 sec. long for CNN training, validation, and testing
    numDice = random.randint(1, 10)
    if numDice == 1:
        dataUsage = 'Prediction'
    elif numDice < 4:
        dataUsage = 'Validation'
    else:
        dataUsage = 'Training'
    
    print(numDice,dataUsage)
    
    num_Samples = int(sample_rate) # get number of samples for 1 sec.
    n = len(fig.axes)
    for j in range(0,len(wav_data)-num_Samples,num_Samples):
        n = n + 1
        tWav_data = wav_data[j:j+num_Samples]
        t_f0, t_voiced_flag, t_voiced_probs = librosa.pyin(tWav_data, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        t_f0[np.isnan(t_f0)] = 0
        
        #Mixing with noise_data
        tWav_data = wav_data[j:j+num_Samples] + noise_data[numDice*500:numDice*500+num_Samples]
        wav_rms = np.sqrt(np.mean(wav_data[j:j+num_Samples]**2)*(2**15))
        noise_rms = np.sqrt(np.mean(noise_data[numDice*500:numDice*500+num_Samples]**2)*(2**15))
        SNR = 20*np.log10(wav_rms/noise_rms)
                
        if n <= 10:
            plot_t = fig.add_subplot(1,10,n)
            plot_t.specgram(tWav_data, NFFT=size_NFFT, Fs=sample_rate, noverlap=size_OVERLAP)
            spectral_png = SPECTRAL_DIR + '\\' + dataUsage + '\\' + parent_list[i] + str(n) + '_Spec.png'
            xmin, xmax = plot_t.get_xlim()
            ymin, ymax = plot_t.get_ylim()
            x0,x1 = xmin, xmax
            y0,y1 = ymin, 1000
            bbox = Bbox([[x0,y0],[x1,y1]])
            bbox = bbox.transformed(plot_t.transData).transformed(plot_t.figure.dpi_scale_trans.inverted())
            plot_t.figure.savefig(spectral_png, bbox_inches=bbox)
            
            spectral_f0 = SPECTRAL_DIR + '\\' + dataUsage + '\\' + parent_list[i] + str(n) + '_f0.csv'
            df_f0 = pd.DataFrame(t_f0)
            df_f0.to_csv(spectral_f0, header=False)
            print(spectral_f0)

            image = mpimg.imread(spectral_png)
            if dataUsage == 'Training':
                x_training = np.append(x_training, [image], axis=0)
                y_training = np.append(y_training, [t_f0], axis=0)
            if dataUsage == 'Validation':
                x_validation = np.append(x_validation, [image], axis=0)
                y_validation = np.append(y_validation, [t_f0], axis=0)
            if dataUsage == 'Prediction':
                x_prediction = np.append(x_prediction, [image], axis=0)
                y_prediction = np.append(y_prediction, [t_f0], axis=0)
                prediction_wav = SPECTRAL_DIR + '\\' + dataUsage + '\\' + parent_list[i] + str(n) + '_pred.wav'
                wavfile.write(prediction_wav, 16000, tWav_data)
                Test_Index_File.write(f"{prediction_wav},{SNR}\n")
           
            
        else:
            break
    
    plt.cla()
    plt.close(fig)


Test_Index_File.close()

#CNN Model architecture using TensorFlow
model = models.Sequential()
model.add(tf.keras.layers.Input(shape=(38, 65, 4)))
model.add(tf.keras.layers.Conv2D(16, 3, strides=2, padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(tf.keras.layers.Flatten())

model.add(core.Dense(500, activation='relu'))
model.add(core.Dropout(.5))
model.add(core.Dense(300, activation='relu'))
model.add(core.Dropout(.25))
model.add(core.Dense(200, activation='relu'))
model.add(core.Dense(32))

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(x_training, y_training, epochs=500, validation_data=(x_validation,y_validation))

preds = model.predict(x_prediction)

df_yPrediction = pd.DataFrame(y_prediction)
#Print the ground zero pith values
df_yPrediction.to_csv(f'.\\yGroundZero.csv', header=False, index=False)
#Print the pitch values that the CNN predicted 
df_preds = pd.DataFrame(preds)
df_preds.to_csv(f'.\\Prediction.csv', header=False, index=False)

    
