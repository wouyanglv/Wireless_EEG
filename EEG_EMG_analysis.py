import numpy as np
import pandas as pd
import os, sys, csv, math, glob, re, time, pathlib
import scipy.signal as signal 
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def spec_gen(voltage, fs, nyq, lowcut, highcut, windowLen, windowHop):
    lower = lowcut/nyq
    upper = highcut/nyq

    numtaps = np.round(3*(fs/lowcut))
    b = signal.firwin(int(numtaps), [lower, upper], pass_zero=False)

    voltage = voltage.replace(np.nan, 0)
    voltage = pd.to_numeric(voltage, errors='coerce')
    voltage = voltage - voltage.mean()

    voltage = signal.filtfilt(b, 1, voltage)
    length = len(voltage) 

    nfft = windowLen*2
    nfft2 = next_power_of_2(nfft)      # number of fft points (recommended to be power of 2)
    window = np.hanning(windowLen)  # our half cosine window
    f, t, s = signal.stft(voltage, fs, window, windowLen, windowHop, nfft2)
    s.shape
    s = np.abs(s)   # real numbers from FFT

    return voltage, f, t, s


dir = '/path/of/folder/with/EEG/data/'
dataFiles = glob.glob(dir + 'EEG_EMG_*.csv')      # analyzes all files in directory with certain attributes

start_time = datetime.strptime('21-07-19-14_25_00', '%y-%m-%d-%H_%M_%S').timestamp()
file_start_time = start_time
file_end_time = start_time

################## Hyperparameters ##################
fs = 256
nyq = 0.5 * fs  
EEG_lowcut = 0.5     # Change this value as needed
EEG_highcut = 30   # Change this value as needed
EMG_lowcut = 0.5    # Change this value as needed
EMG_highcut = 100   # Change this value as needed

windowLen = 2*fs
windowHop = windowLen*0.5 

################## Iterate through files ##################
for i in range(len(dataFiles)):
    start = time.time()
    print(f'Processing data {i+1}')
    fileName = f'EEG_EMG_{i+1}'
    data = pd.read_csv(dir + fileName + '.csv', sep=',', header = 0)
    data_EEG = data['EEG']
    data_EMG = data['EMG']

    file_start_time = file_end_time
    file_end_time = file_start_time + len(data_EEG)/fs

    ################## Process EEG ##################
    data_EEG, f_EEG, t_EEG, s_EEG = spec_gen(data_EEG, fs, nyq, EEG_lowcut, EEG_highcut, windowLen, windowHop)
      
    s_sum_EEG = s_EEG.sum(axis = 0)      # This sums the power in each column (total power per FFT time)
    s_norm_EEG = s_EEG / s_sum_EEG[None,:]   # Divides each element in a column by the column mean

    delta_low = (np.abs(f_EEG-1)).argmin()
    delta_high = (np.abs(f_EEG-4)).argmin()
    normDeltaPower = np.mean(s_norm_EEG[delta_low:delta_high+1], axis=0)
    meanDeltaPower = np.mean(normDeltaPower)
    StDevDeltaPower = np.std(normDeltaPower)

    theta_low = (np.abs(f_EEG-6)).argmin()
    theta_high = (np.abs(f_EEG-9)).argmin()
    normThetaPower = np.mean(s_norm_EEG[theta_low:theta_high+1], axis=0)
    meanNormThetaPower = np.mean(normThetaPower)
    StDevNormThetaPower = np.std(normThetaPower)
    
    normTDratio = normThetaPower/normDeltaPower
    meanNormTDratio = np.mean(normTDratio)
    StDevNormTDratio = np.std(normTDratio)
                        
    ################## Process EMG ##################
    data_EMG, f_EMG, t_EMG, s_EMG = spec_gen(data_EMG, fs, nyq, EMG_lowcut, EMG_highcut, windowLen, windowHop)
    
    s_sum_EMG = s_EMG.sum(axis = 0)      # This sums the power in each column (total power per FFT time)
    s_norm_EMG = s_EMG / s_sum_EMG[None,:]      
    s_mean_EMG = np.mean(s_EMG, 0)

    ################## Plotting ##################
    
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=5, ncols=1, sharex=True)

    ax1.plot(np.arange(len(data_EEG))/fs, data_EEG, linewidth = 0.5)
    ax1.set_ylabel('EEG (uV)', fontsize = 9)
    ax1.set_ylim([-100,100])

    ax2.levels = np.arange(.005,.04,.001)
    spect = ax2.contourf(t_EEG, f_EEG, s_norm_EEG, ax2.levels, cmap="jet")
    ax2.set_ylim(0.5,15)
    plt.setp(ax2.get_yticklabels(), fontsize = 8)
    ax2.set_ylabel('f (Hz)', fontsize = 9)
    ax2.set_yticks([0,5,10,15])

    ax3.plot(np.arange(len(data_EMG))/fs, data_EMG, linewidth = 0.5)
    ax3.set_ylabel('EMG (uV)', fontsize = 9)
    ax3.set_ylim([-300,300])
        
    ax4.plot(t_EEG, normDeltaPower, label='Delta (1-4 Hz)')
    ax4.plot(t_EEG, normThetaPower, label='Theta (6-9 Hz)')
    ax4.legend(loc='upper right', fontsize='x-small', frameon=False)
    plt.setp(ax4.get_yticklabels(), fontsize = 8)
    ax4.set_ylabel('Power (a.u.)', fontsize = 9)
    ax4.set_ylim([0,0.05])
    ax4.set_yticks([0, 0.025, 0.05])

    ax5.plot(t_EEG, normTDratio, label='TD Ratio')
    ax5.set_xlim(0,t_EEG[-1])
    plt.setp(ax5.get_xticklabels(), fontsize = 8)
    ax5.set_xlabel('Time (mins)', fontsize = 9)
    plt.setp(ax5.get_yticklabels(), fontsize = 8)
    ax5.set_ylabel('TD ratio', fontsize = 9)
    ax5.set_ylim([0,2])
    ax5.set_yticks([0,1,2])
    ticks = np.arange(0, len(data_EMG)/fs, 60*5)
    ax5.set_xticks(ticks)
    tick_labels = [datetime.fromtimestamp(tick+ file_start_time).strftime("%H:%M") for tick in ticks]
    ax5.set_xticklabels(tick_labels)

    fig.tight_layout()
    fig.align_ylabels()
    plt.show()
    plt.savefig(dir + fileName + '.png')
    plt.close()
    
    elapsed = time.time() - start
    print(f'Elapsed: {elapsed} s')