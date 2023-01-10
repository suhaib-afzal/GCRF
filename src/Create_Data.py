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

def Create_Data(Iterations=1):
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
    round0 = [0,0,0,0,0]
    round1 = [0,0,0,1,0]
    round2 = [1,0,0,0,0]
    round3 = [0,1,0,0,0]
    round4 = [0,0,1,0,0]
    round5 = [1,1,1,0,0]
    round6 = [1,1,1,1,0]
    round7 = [0,0,0,0,1]
    
    NoiseType = [round0, round1, round2, round3, round4, round5, round6, round7]
    
    NoVar = [0,0]
    BoxVar = [1,0]
    ChanVar = [0,1]
    AllVar = [1,1]
    
    Var = [NoVar, BoxVar, ChanVar, AllVar]
    
    #imported_datas, Exertion = 0, boxVar=0, chanVar=0, type3 = 0, indv = 0, session = 0, Breath=0, Vaso=0, Heart=0, Gauss=0, Experi=0, Plot=0
    distsVecList = sg.import_distsVec()
    D = sg.import_datums(distsVecList, nSamples=3000)
    
    BX = np.empty((10,4))
    CH = np.empty((10,4))
    
    Iterations = int(Iterations)
    
    path = os.path.join(r"C:\Users\suhai\Downloads\ProjDirectNoGit\ProjDirectNoGit",r"Data")
    os.makedirs(path, exist_ok = True)
    print("Made Data Directory")
    os.chdir(path)
    for m in range(8):
        st_m = str(m)
        path1 = os.path.join(r"C:\Users\suhai\Downloads\ProjDirectNoGit\ProjDirectNoGit\Data",r"NoiseType"+st_m)
        os.makedirs(path1, exist_ok = True)
        print("Made Noise"+st_m+" Directory")
        os.chdir(path1)
        Noise = NoiseType[m]
        for l in range(Iterations):
            st_l = str(l)
            path2 = os.path.join(r"C:\Users\suhai\Downloads\ProjDirectNoGit\ProjDirectNoGit\Data\NoiseType"+st_m,r"Iteration"+st_l)
            os.makedirs(path2, exist_ok = True)
            print("Made Iterations" + st_l+" Directory")
            os.chdir(path2)
            for k in range(4):
                st_k = str(k)
                path3 = os.path.join(r"C:\Users\suhai\Downloads\ProjDirectNoGit\ProjDirectNoGit\Data\NoiseType"+st_m+r"\Iteration"+st_l,r"Var"+st_k)
                os.makedirs(path3, exist_ok = True)
                print("Made Variability"+st_k+" Directory")
                os.chdir(path3)
                CurVar = Var[k]
                for j in range(3): 
                    st_j = str(j+1)
                    path4 = os.path.join(r"C:\Users\suhai\Downloads\ProjDirectNoGit\ProjDirectNoGit\Data\NoiseType"+st_m+r"\Iteration"+st_l+r"\Var"+st_k, r"Session"+st_j)
                    os.makedirs(path4, exist_ok = True)
                    print("Made Session"+st_j+ " Directory")
                    os.chdir(path4)
                    for i in range(10):
                        st_i = str(i+1)
                        path5 = os.path.join(r"C:\Users\suhai\Downloads\ProjDirectNoGit\ProjDirectNoGit\Data\NoiseType"+st_m+"\Iteration"+st_l+r"\Var"+st_k+r"\Session"+st_j, r"Subj"+st_i)
                        print("Made Subject"+st_i+ " Directory")
                        os.makedirs(path5, exist_ok = True)
                        os.chdir(path5)
                        Result = sg.execute(D, Exertion=j, boxVar=CurVar[0], chanVar=CurVar[1], Breath = Noise[0], Vaso = Noise[1], Heart = Noise[2], Gauss = Noise[3], Experi = Noise[4])
                        print("Generated Subject"+st_i+ " Data")
                        NeuroImage = Result[0]
                        deoxy = NeuroImage[:,:,1]
                        oxy = NeuroImage[:,:,0]
                        pd.DataFrame(oxy).to_csv("synthoxy_"+str(i+1)+".csv")
                        pd.DataFrame(deoxy).to_csv("sythdeoxy_"+str(i+1)+".csv")
                        print("Saved Subject"+st_i+ " Data")
                        bx = Result[1]
                        BX[i,:]=bx
                        ch = Result[2]
                        CH[i,:]=ch
                        #os.chdir(path4)
                    os.chdir(path4)
                    pd.DataFrame(BX).to_csv("BoxcarAmplitudes.csv")
                    pd.DataFrame(CH).to_csv("ChannelAmplitudes.csv")
                os.chdir(path3)
            os.chdir(path2)
        os.chdir(path1)
    os.chdir(path)
        
    BX = np.empty((10,4))
    CH = np.empty((10,4))
    
    Result0 = sg.execute(D, Exertion=0, boxVar=0, chanVar=0, type3 = 0, indv = 0, session = 0, Breath=0, Vaso=0, Heart=0, Gauss=0, Experi=0, Plot=0)
    #print(Result0)
    Result1 = sg.execute(D, Exertion=1, boxVar=0, chanVar=0, type3 = 0, indv = 0, session = 0, Breath=0, Vaso=0, Heart=0, Gauss=0, Experi=0, Plot=0)
    #print(Result1)
    Result2 = sg.execute(D, Exertion=2, boxVar=0, chanVar=0, type3 = 0, indv = 0, session = 0, Breath=0, Vaso=0, Heart=0, Gauss=0, Experi=0, Plot=0)
    #print(Result2)
        
    path = os.path.join(r"C:\Users\suhai\Downloads\ProjDirectNoGit\ProjDirectNoGit",r"Data")
    #os.makedirs(path, exist_ok = True)
    print("Accessed Data Directory")
    os.chdir(path)
    for m in range(8):
        st_m = str(m)
        path1 = os.path.join(r"C:\Users\suhai\Downloads\ProjDirectNoGit\ProjDirectNoGit\Data",r"NoiseType"+st_m)
        #os.makedirs(path1, exist_ok = True)
        print("Accessed Noise"+st_m+" Directory")
        os.chdir(path1)
        for l in range(Iterations):
            st_l = str(l)
            path2 = os.path.join(r"C:\Users\suhai\Downloads\ProjDirectNoGit\ProjDirectNoGit\Data\NoiseType"+st_m,r"Iteration"+st_l)
            #os.makedirs(path2, exist_ok = True)
            print("Accessed Iterations"+st_l+" Directory")
            os.chdir(path2)
            for k in range(4):
                st_k = str(k)
                path3 = os.path.join(r"C:\Users\suhai\Downloads\ProjDirectNoGit\ProjDirectNoGit\Data\NoiseType"+st_m+r"\Iteration"+st_l,r"Var"+st_k)
                #os.makedirs(path3, exist_ok = True)
                print("Accessed Variability"+st_k+" Directory")
                os.chdir(path3)
                for j in range(3):
                    if j==0:
                        st_j = 'ZeroHRF004'
                        Result = Result0
                    elif j==1:
                        st_j = 'HalfHRF005'
                        Result = Result1
                    elif j==2:
                        st_j = 'HRF006'
                        Result = Result2
                    path12 = os.path.join(r"C:\Users\suhai\Downloads\ProjDirectNoGit\ProjDirectNoGit\Data\NoiseType"+st_m+r"\Iteration"+st_l+r"\Var"+st_k, st_j)
                    os.makedirs(path12, exist_ok = True)
                    os.chdir(path12)
                    print("Made HRF Sessions")
                    NeuroImage = Result[0]
                    deoxy = NeuroImage[:,:,1]
                    oxy = NeuroImage[:,:,0]
                    pd.DataFrame(oxy).to_csv("synthoxy_.csv")
                    pd.DataFrame(deoxy).to_csv("sythdeoxy_.csv")
                    print("Saved HRF Data")
                os.chdir(path3)
            os.chdir(path2)
        os.chdir(path1)
    os.chdir(path)
    

        
    # BX = np.empty((20,4))
    # CH = np.empty((20,4))
    
    
    # if not os.path.isdir(r"C:\Users\suhai\Downloads\ProjectDirectory\Data_Type3"):
        # path6 = os.path.join(r"C:\Users\suhai\Downloads\ProjectDirectory\Data_Type3\NoiseType6",r"Type3")
        # os.makedirs(path6, exist_ok = True)
        # os.chdir(path6)
        # for a in range(2):
            # st_a = str(a)
            # path7 = os.path.join(r"C:\Users\suhai\Downloads\ProjectDirectory\Data_Type3\NoiseType6\Type3",r"Session"+st_a)
            # os.makedirs(path7, exist_ok = True)
            # os.chdir(path7)
            # for b in range(20):
                # st_b = str(b)
                # path8 = os.path.join(r"C:\Users\suhai\Downloads\ProjectDirectory\Data_Type3\NoiseType6\Type3\Session"+st_a,r"Subj"+st_b)
                # os.makedirs(path8, exist_ok = True)
                # os.chdir(path8)
                # T = sg.execute(D, type3 = 1, indv = b, session = a, Breath=1, Vaso=1, Heart=1, Gauss=1)
                # NeuroImage = Result[0]
                # deoxy = NeuroImage[:,:,1]
                # oxy = NeuroImage[:,:,0]
                # pd.DataFrame(oxy).to_csv("synthoxy_"+str(i+1)+".csv")
                # pd.DataFrame(deoxy).to_csv("sythdeoxy_"+str(i+1)+".csv")
                # bx = Result[1]
                # BX[i,:]=bx
                # ch = Result[2]
                # CH[i,:]=ch
                
            # os.chdir(path7)
            # pd.DataFrame(BX).to_csv("BoxcarAmplitudes.csv")
            # pd.DataFrame(CH).to_csv("ChannelAmplitudes.csv")
            
        # path9 = os.path.join(r"C:\Users\suhai\Downloads\ProjectDirectory\Data_Type3\NoiseType7",r"Type3")
        # os.makedirs(path9, exist_ok = True)
        # os.chdir(path9)
        # for a in range(2):
            # st_a = str(a)
            # path10 = os.path.join(r"C:\Users\suhai\Downloads\ProjectDirectory\Data_Type3\NoiseType7\Type3",r"Session"+st_a)
            # os.makedirs(path10, exist_ok = True)
            # os.chdir(path10)
            # for b in range(20):
                # st_b = str(b)
                # path11 = os.path.join(r"C:\Users\suhai\Downloads\ProjectDirectory\Data_Type3\NoiseType7\Type3\Session"+st_a,r"Subj"+st_b)
                # os.makedirs(path11, exist_ok = True)
                # os.chdir(path11)
                # T = sg.execute(D, type3 = 1, indv = b, session = a, Experi = 1)
                # NeuroImage = Result[0]
                # deoxy = NeuroImage[:,:,1]
                # oxy = NeuroImage[:,:,0]
                # pd.DataFrame(oxy).to_csv("synthoxy_"+str(i+1)+".csv")
                # pd.DataFrame(deoxy).to_csv("sythdeoxy_"+str(i+1)+".csv")
                # bx = Result[1]
                # BX[i,:]=bx
                # ch = Result[2]
                # CH[i,:]=ch
            
            # os.chdir(path10)
            # pd.DataFrame(BX).to_csv("BoxcarAmplitudes.csv")
            # pd.DataFrame(CH).to_csv("ChannelAmplitudes.csv")
    #end

if __name__ == '__main__':             
    Create_Data(Iterations=3)