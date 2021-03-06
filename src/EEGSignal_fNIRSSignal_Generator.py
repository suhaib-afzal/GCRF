# -*- coding: utf-8 -*-
#
#File: EEGSignal_fNIRSSignal_Generator.py
#
'''
Created on Fri Sep 25 21:00:00 2020

Module ***EEGSignal_fNIRSSignal_Generator.py***

This module implements the class :class:`fNIRSSignalGenerator <fNIRSSignalGenerator>`.

:Log:

+-------------+---------+------------------------------------------------------+
| Date        | Authors | Description                                          |
+=============+=========+======================================================+
| 03-Jul-2020 |   FOE   | - Class :class:`fNIRSSignalGenerator` created but    |
|             |   JJR   |   unfinished.                                        |
+-------------+--------+------------------------------------------------------+


.. sectionauthor:: Felipe Orihuela-Espina <f.orihuela-espina@inaoep.mx> and Jesús Joel Rivas <jrivas@inaoep.mx>
.. codeauthor::    Felipe Orihuela-Espina <f.orihuela-espina@inaoep.mx> and Jesús Joel Rivas <jrivas@inaoep.mx>

'''

import copy
import warnings

import numpy as np
import random
import math

import matplotlib.pyplot as plt

from scipy.interpolate import interp1d

from scimeth.data.smTimeline import smTimeline

from scimeth.data.smTimelineEvent import smTimelineEvent

from EEGSignalGenerator import EEGSignalGenerator

from fNIRSSignalGenerator import fNIRSSignalGenerator


#from scipy import stats

from src.EEGSignalGenerator import plotSyntheticEEG


def main():
    iD_Run = 2
    print(f'Run ID:       \t\t {iD_Run:03d}')
    print("")

    samplingTime = 200   # time in secs
    print("Sampling time:\t\t", samplingTime, "secs")
    print("")

    print("EEG information")
    print("--------------------------")
    #sgEEG = EEGSignalGenerator(nSamples=3000, nChannels=6)
    sgEEG = EEGSignalGenerator()
    #EEGsamplingRate = sgEEG.samplingRate
    sgEEG.samplingRate = 128
    sgEEG.nSamples = int(sgEEG.samplingRate * samplingTime)
    sgEEG.nChannels = 6
    print("Sampling rate:\t\t",    sgEEG.samplingRate, "Hz")
    print("Number of samples:\t",  sgEEG.nSamples)
    print("Number of channels:\t", sgEEG.nChannels)
    sgEEG.execute()
    # print(sg.data)
    plotSyntheticEEG(sgEEG.data)
    print("")

    print("fNIRS information")
    print("--------------------------")
    #sgfNIRS = fNIRSSignalGenerator(nSamples=3000, nChannels=4)
    sgfNIRS = fNIRSSignalGenerator()
    #fNIRSsamplingRate = sgfNIRS.samplingRate
    sgfNIRS.samplingRate = 10
    sgfNIRS.nSamples = int(sgfNIRS.samplingRate * samplingTime)
    sgfNIRS.nChannels = 4
    print("Sampling rate:\t\t",    sgfNIRS.samplingRate, "Hz")
    print("Number of samples:\t",  sgfNIRS.nSamples)
    print("Number of channels:\t", sgfNIRS.nChannels)
    sgfNIRS.execute()
    print("")

    sgEEG_Timeline   = smTimeline(length = sgEEG.nSamples)
    sgfNIRS_Timeline = smTimeline(length = sgfNIRS.nSamples)

    print("Timeline information")
    print("--------------------------")
    print("EEG Timeline:\t\t",   "length: ", sgEEG_Timeline.length,   "\tinit: ",   sgEEG_Timeline.init,   "\tend: ", sgEEG_Timeline.end)
    print("fNIRS Timeline:\t\t", "length: ", sgfNIRS_Timeline.length, "\t\tinit: ", sgfNIRS_Timeline.init, "\tend: ", sgfNIRS_Timeline.end)
    print("")

    ev = smTimelineEvent(onset=0, duration=1)
    print(ev)

    sgEEG_Timeline.addEvents(ev)
    sgfNIRS_Timeline.addEvents(ev)

    print(sgEEG_Timeline.getEventsID())
    print("")

    print(sgfNIRS_Timeline.getEventsID())
    print("")

    if sgEEG.nSamples > sgfNIRS.nSamples:
        step = (sgEEG.nSamples - 1) / (sgfNIRS.nSamples - 1)  # Synchronization begins with the first sample of both signals,
                                                              # so there are (nSamples - 1) left that must be synchronized.
    else:   # sgEEG.nSamples <= sgfNIRS.nSamples
        step = (sgfNIRS.nSamples - 1) / (sgEEG.nSamples - 1)

    print("Synchronization step: ", step)
    print("")

    step = round(step)
    print("Rounded synchronization step: ", step)
    print("")

    if sgEEG.nSamples > sgfNIRS.nSamples:
    # Knowing which of the signals has the largest nSamples, then a tensor with zeros is created
    # for the smallest with the nSamples of the largest
        sgfNIRS_NewData = np.zeros((sgEEG.nSamples, sgfNIRS.nChannels, 2),dtype=float)

        x      = np.linspace(0, sgEEG.nSamples-1, num=sgfNIRS.nSamples, endpoint=True)
        xInter = np.linspace(0, sgEEG.nSamples-1, num=sgEEG.nSamples,   endpoint=True)

        for nCh in range(0, sgfNIRS.nChannels):     #Ends in sgfNIRS.nChannels - 1
            for nSg in range(0, 2):                 #Ends in 1
                y = copy.deepcopy(sgfNIRS.data[:, nCh, nSg])
                f = interp1d(x, y)
                fxInter = f(xInter)
                sgfNIRS_NewData[:, nCh, nSg] = copy.deepcopy(fxInter)
                plt.plot(x, y, 'o', xInter, fxInter, '-')
                plt.show()

        # Creation of the output file for the EEG synthetic data
        fileName = "synthetic{:03d}_EEG_{}Chs_{:.0f}Hz_{}secs.txt"
        fileOut = open(fileName.format(iD_Run, sgEEG.nChannels, sgEEG.samplingRate, samplingTime), "w")
        fileOut.write(f'File: {fileName.format(iD_Run, sgEEG.nChannels, sgEEG.samplingRate, samplingTime)}\n')
        fileOut.write(f'Run ID:       \t\t{iD_Run:03d}\n')
        fileOut.write(f'Number of samples:\t{sgEEG.nSamples}\n')
        fileOut.write(f'Number of channels:\t{sgEEG.nChannels}\n')
        fileOut.write(f'Sampling rate:\t\t{sgEEG.samplingRate:.0f} Hz\n')
        fileOut.write(f'Sampling time:\t\t{samplingTime} secs\n')

        nChs = sgEEG.nChannels
        for i in range(sgEEG.nSamples):
            sgEEG_list = sgEEG.data[i,:,0].tolist()
            for j in range(nChs-1):
                fileOut.write(f'{sgEEG_list[j]}, ')
            fileOut.write(f'{sgEEG_list[-1]}\n')

        fileOut.close()

        # Creation of the output file for the fNIRs synthetic data
        fileName = "synthetic{:03d}_fNIRs_{}Chs_{:.0f}Hz_{}secs.txt"
        fileOut = open(fileName.format(iD_Run, sgfNIRS.nChannels, sgfNIRS.samplingRate, samplingTime), "w")
        fileOut.write(f'File: {fileName.format(iD_Run, sgfNIRS.nChannels, sgfNIRS.samplingRate, samplingTime)}\n')
        fileOut.write(f'Run ID:       \t\t{iD_Run:03d}\n')
        fileOut.write(f'Number of samples:\t{sgfNIRS.nSamples} \tNumber of samples after interpolation:\t{sgEEG.nSamples}\n')
        fileOut.write(f'Number of channels:\t{sgfNIRS.nChannels}\n')
        fileOut.write(f'Sampling rate:\t\t{sgfNIRS.samplingRate:.0f} Hz\n')
        fileOut.write(f'Sampling time:\t\t{samplingTime} secs\n')

        nChs = sgfNIRS.nChannels
        for i in range(sgEEG.nSamples):
            sgfNIRS_list = sgfNIRS_NewData[i, :, 0].tolist()
            for j in range(nChs - 1):
                fileOut.write(f'{sgfNIRS_list[j]}, ')
            fileOut.write(f'{sgfNIRS_list[-1]}\n')

            sgfNIRS_list = sgfNIRS_NewData[i, :, 1].tolist()
            for j in range(nChs - 1):
                fileOut.write(f'{sgfNIRS_list[j]}, ')
            fileOut.write(f'{sgfNIRS_list[-1]}\n')

        fileOut.close()

    else:   # sgEEG.nSamples <= sgfNIRS.nSamples
        sgEEG_NewData = np.zeros((sgfNIRS.nSamples, sgEEG.nChannels, 1),dtype=float)

        x      = np.linspace(0, sgfNIRS.nSamples-1, num=sgEEG.nSamples,   endpoint=True)
        xInter = np.linspace(0, sgfNIRS.nSamples-1, num=sgfNIRS.nSamples, endpoint=True)

        for nCh in range(0, sgEEG.nChannels):     #Ends in sgEEG.nChannels - 1
            y = copy.deepcopy(sgEEG.data[:, nCh, 0])
            f = interp1d(x, y)
            fxInter = f(xInter)
            sgEEG_NewData[:, nCh, 0] = copy.deepcopy(fxInter)
            plt.plot(x, y, 'o', xInter, fxInter, '-')
            plt.show()

        # Creation of the output file for the EEG synthetic data
        fileName = "synthetic{:03d}_EEG_{}Chs_{:.0f}Hz_{}secs.txt"
        fileOut = open(fileName.format(iD_Run, sgEEG.nChannels, sgEEG.samplingRate, samplingTime), "w")
        fileOut.write(f'File: {fileName.format(iD_Run, sgEEG.nChannels, sgEEG.samplingRate, samplingTime)}\n')
        fileOut.write(f'Run ID:       \t\t{iD_Run:03d}\n')
        fileOut.write(f'Number of samples:\t{sgEEG.nSamples} \tNumber of samples after interpolation:\t{sgfNIRS.nSamples}\n')
        fileOut.write(f'Number of channels:\t{sgEEG.nChannels}\n')
        fileOut.write(f'Sampling rate:\t\t{sgEEG.samplingRate:.0f} Hz\n')
        fileOut.write(f'Sampling time:\t\t{samplingTime} secs\n')

        nChs = sgEEG.nChannels
        for i in range(sgfNIRS.nSamples):
            sgEEG_list = sgEEG_NewData[i,:,0].tolist()
            for j in range(nChs-1):
                fileOut.write(f'{sgEEG_list[j]}, ')
            fileOut.write(f'{sgEEG_list[-1]}\n')

        fileOut.close()

        # Creation of the output file for the fNIRs synthetic data
        fileName = "synthetic{:03d}_fNIRs_{}Chs_{:.0f}Hz_{}secs.txt"
        fileOut = open(fileName.format(iD_Run, sgfNIRS.nChannels, sgfNIRS.samplingRate, samplingTime), "w")
        fileOut.write(f'File: {fileName.format(iD_Run, sgfNIRS.nChannels, sgfNIRS.samplingRate, samplingTime)}\n')
        fileOut.write(f'Run ID:       \t\t{iD_Run:03d}\n')
        fileOut.write(f'Number of samples:\t{sgfNIRS.nSamples}\n')
        fileOut.write(f'Number of channels:\t{sgfNIRS.nChannels}\n')
        fileOut.write(f'Sampling rate:\t\t{sgfNIRS.samplingRate:.0f} Hz\n')
        fileOut.write(f'Sampling time:\t\t{samplingTime} secs\n')

        nChs = sgfNIRS.nChannels
        for i in range(sgfNIRS.nSamples):
            sgfNIRS_list = sgfNIRS.data[i, :, 0].tolist()
            for j in range(nChs - 1):
                fileOut.write(f'{sgfNIRS_list[j]}, ')
            fileOut.write(f'{sgfNIRS_list[-1]}\n')

            sgfNIRS_list = sgfNIRS.data[i, :, 1].tolist()
            for j in range(nChs - 1):
                fileOut.write(f'{sgfNIRS_list[j]}, ')
            fileOut.write(f'{sgfNIRS_list[-1]}\n')

        fileOut.close()

# end main()


if __name__ == '__main__':
    main()
