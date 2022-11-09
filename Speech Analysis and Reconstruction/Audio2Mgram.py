
'''
Title:      Audio2Mgram

Author:     Philip Schwarzmayr
Date:       09/11/2022
Purpose:    To convert inputted audio signals into Mel-Spectrogram Diagrams. 
            Low-frequency whale sounds and high frequency bird sounds are used.
'''
'''Initialisation'''
import numpy as np
import matplotlib.pyplot as plt
import librosa, librosa.display

'''Loading Audio Files and Plotting'''
plt.figure()
librosa.audio
yW, srW = librosa.load('nprWhale.wav')
# trim silent edges
whale_song, _ = librosa.effects.trim(yW)
librosa.display.waveshow(whale_song, sr=srW, color="blue") 
plt.title('Whale Song Waveplot', fontdict=dict(size=18))
plt.xlabel('Time', fontdict=dict(size=15))
plt.ylabel('Amplitude', fontdict=dict(size=15))

plt.figure()
yB, srB = librosa.load('wbBird.wav')
# trim silent edges
bird_song, _ = librosa.effects.trim(yB)
librosa.display.waveshow(bird_song, sr=srB, color="red") 
plt.title('Bird Song Waveplot', fontdict=dict(size=18))
plt.xlabel('Time', fontdict=dict(size=15))
plt.ylabel('Amplitude', fontdict=dict(size=15))

'''Mel Spectrogram Conversion'''

# this is the number of samples in a window per fft
n_fft = 2048
# The amount of samples we are shifting after each fft
hop_length = 512

mel_signal = librosa.feature.melspectrogram(y=whale_song, sr=srW, hop_length=hop_length, 
 n_fft=n_fft)
spectrogram = np.abs(mel_signal)
power_to_db = librosa.power_to_db(spectrogram, ref=np.max)
plt.figure()
librosa.display.specshow(power_to_db, sr=srW, x_axis='time', y_axis='mel', cmap='magma', 
 hop_length=hop_length)
plt.colorbar(label='dB')
plt.title('Whale Song Mel-Spectrogram (dB)', fontdict=dict(size=18))
plt.xlabel('Time', fontdict=dict(size=15))
plt.ylabel('Frequency', fontdict=dict(size=15))

mel_signal = librosa.feature.melspectrogram(y=bird_song, sr=srB, hop_length=hop_length, 
 n_fft=n_fft)
spectrogram = np.abs(mel_signal)
power_to_db = librosa.power_to_db(spectrogram, ref=np.max)
plt.figure()
librosa.display.specshow(power_to_db, sr=srB, x_axis='time', y_axis='mel', cmap='magma', 
 hop_length=hop_length)
plt.colorbar(label='dB')
plt.title('Bird Song Mel-Spectrogram (dB)', fontdict=dict(size=18))
plt.xlabel('Time', fontdict=dict(size=15))
plt.ylabel('Frequency', fontdict=dict(size=15))

plt.show()



