import numpy as np

import copy
import warnings

import numpy as np
import random
import math 

import matplotlib.pyplot as plt

import csv
import pandas as pd
import scipy.io
import h5py
import sys
import os

from optodeArrayInfo import optodeArrayInfo

from channelLocationMap import channelLocationMap

from fNIRSdatagen import fNIRSSignalGenerator

newId = 3
newDescription = 'First New Config'
newNChannels = 4
newNOptodes = 4
newChLocations = np.array([[1, 2, 0], [0, 1, 0], [2, 1, 0], [1, 0, 0]])
newOptodesLocations = np.array([[0, 2, 0], [2, 2, 0], [0, 0, 0], [2, 0, 0]])
newOptodesTypes = np.array([1, 2, 2, 1])  # Remember {0: Unknown, 1: Emission or source, 2: Detector}
#newReferencePoints = dict({'Nz': np.array([0, -18.5, 0]), 'Iz': np.array([0, 18.5, 0]),
                           #'LPA': np.array([17.5, 0, 0]), 'RPA': np.array([-17.5, 0, 0]),
                           #'Cz': np.array([0, 0, 0])})
newSurfacePositioningSystem = 'UI 10/20'
newChSurfacePositions = tuple(('Fz', 'C3', 'C4', 'Cz'))
newOptodesSurfacePositions = tuple(('FC5', 'CP3', 'FC6', 'CP4'))
newChOptodeArrays = np.array([0, 0, 0, 0])
newOptodesOptodeArrays = np.array([0, 0, 0, 0])
newPairings = np.array([[0, 1], [0, 2], [3, 1], [3, 2]])

NewChTopoArrangement = np.array([[1, 2, 0], [0, 1, 0], [2, 1, 0], [1, 0, 0]])
NewOptodesTopoArrangement = np.array([[0, 2, 0], [2, 2, 0], [0, 0, 0], [2, 0, 0]])

oaInfo = optodeArrayInfo(nChannels=newNChannels, nOptodes=newNOptodes, \
                        mode='HITACHI ETG-4000 2x2 optode array', typeOptodeArray='adult', \
                        chTopoArrangement=NewChTopoArrangement, \
                        optodesTopoArrangement=NewOptodesTopoArrangement)

newOptodeArrays = np.array([oaInfo])


sg = fNIRSSignalGenerator(nSamples = 3000, id = newId, description = newDescription,
                          nChannels = newNChannels, nOptodes  = newNOptodes,
                          chLocations = newChLocations, optodesLocations = newOptodesLocations,
                          optodesTypes = newOptodesTypes,
                          surfacePositioningSystem = newSurfacePositioningSystem,
                          chSurfacePositions = newChSurfacePositions,
                          optodesSurfacePositions = newOptodesSurfacePositions,
                          chOptodeArrays = newChOptodeArrays, optodesOptodeArrays = newOptodesOptodeArrays,
                          pairings = newPairings, optodeArrays = newOptodeArrays)

Result=sg.execute(isHRF=1,Exertion = 1)
NeuroImage = Result[0]
deoxy = NeuroImage[:,:,1]
oxy = NeuroImage[:,:,0]
pd.DataFrame(oxy).to_csv("synthoxy_5.csv")
pd.DataFrame(deoxy).to_csv("sythdeoxy_5.csv")