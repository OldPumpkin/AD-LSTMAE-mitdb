# Title : ecgPreprocessing.py
# Author : Chang-Gil, Jeong
# Last update : July 19, 2023

# Used tools:
# python 3.9.15
# wfdb 4.1.0
# scipy
# tqdm
# matplotlib

# Procedure:
# 1. open record class using wfdb tools.
# 2. extract analog value and annotations from wfdb.Record class
# 3. filtering(LPF,cutoff 20hz)
# 4. normalization
# loop
#   segmentation
#   resampling
# 5. save as '.npy' file format
#
#
# Update log: July 19, 2023
# save with annotation label
# 
# Update log: July 22, 2023
# min-max norm individual segment
#
# Update log: July 24, 2023
# file name marking 


#imports
import numpy as np
import scipy.signal as sig
import wfdb

import tqdm
import matplotlib.pyplot as plt

order = 3  # order

# dbpath : datbase path, wTime : length of windows Size(seconds), fsTarget : target sampling rate(hz)
def ecgProcessing(dbpath,wTime,fsTarget,fs=360):
    wSize = int(wTime*fsTarget)
    sSize = int(wSize/2) # sliding size
    # read subject List from "RECORDS" file
    sbjListFile = open(dbpath+"RECORDS","r")
    sbjList = list(map(str.strip, sbjListFile.readlines()))
    
    N = [] # normal signals
    Nf = [] # normal fore signal
    A = [] # Arrhythmia signals (APC marked)
    AN = [] # another abnormal signals

    # id marking
    idN = [] 
    idA = [] 
    idAN = [] 

    for sbjNum in tqdm.tqdm(sbjList):
        # read subject file and make wfdb object
        record = wfdb.rdrecord(dbpath+sbjNum)
        
        sigLen = record.sig_len
        targetLen = int(sigLen*fsTarget/fs)

        # getting only single channel signals
        # physical signal to numpy array
        pSig = np.nan_to_num(record.p_signal)[:,0]

        # Butterworth lowpass filter
        sos_1 = sig.butter(order, [20], 'low', fs=fs, output='sos')
        xFilt = sig.sosfilt(sos_1, pSig)
        xFilt = sig.resample(xFilt,int(sigLen*(fsTarget/fs)))
        xFilt = 2*((xFilt-np.min(xFilt))/(np.max(xFilt)-np.min(xFilt)))-1

        # read annotation file and make Annoation object
        ann = wfdb.rdann(dbpath+sbjNum,extension='atr',\
                        return_label_elements=['symbol','description'])
        symbol = ann.symbol
        sample = np.array(list(map(lambda x: x/fs*fsTarget, ann.sample)),dtype='int')

        
        annStart = annEnd = 0
        # Segmentation 
        for start, end in zip(range(0,targetLen-wSize-sSize,sSize),range(wSize,targetLen-sSize,sSize)):
            # slice filtered signal
            segment = xFilt[start:end]
            # segment = 2*((segment-np.min(segment))/(np.max(segment)-np.min(segment)))-1

            # annotation range
            # annEnd = annStart+1
            while end > sample[annEnd]:
                if annEnd+1 > len(sample):
                    break
                annEnd += 1

            # annotation classify
            symSeg = symbol[annStart:annEnd]
            if not symSeg: # exception
                AN.append(segment)
                idAN.append(sbjNum)
            elif 'A' in symSeg: # Arrhythmia signals (APC marked)
                A.append(segment)
                idA.append(sbjNum)
            elif 'N' in symSeg and len(symSeg) == symSeg.count('N')+symSeg.count('+'):
                foreSegment = xFilt[start+sSize:end+sSize]
                N.append(segment)
                Nf.append(foreSegment)
                #Nf.append(2*((foreSegment-np.min(foreSegment))/(np.max(foreSegment)-np.min(foreSegment)))-1)
                idN.append(sbjNum)
            else:
                AN.append(segment)
                idAN.append(sbjNum)
                
            annStart = annEnd

    AN = np.array(AN)
    A = np.array(A)
    N = np.array(N)
    Nf = np.array(Nf)

    print('Normal:{}'.format(N.shape))
    print('fore Normal:{}'.format(Nf.shape))
    print('Arrhythmia:{}'.format(A.shape))
    print('Abnormal:{}'.format(AN.shape))
    
    # Save as numpy file
    np.save('./N',N)
    np.save('./Nf',Nf)
    np.save('./A',A)
    np.save('./AN',AN)

    np.save('./idN',idN)
    np.save('./idA',idA)
    np.save('./idAN',idAN)