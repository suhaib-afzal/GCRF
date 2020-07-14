# -*- coding: utf-8 -*-
#
#File: fNIRSSignalGenerator.py
#
'''
Created on Fri Jul 03 21:27:00 2020

Module ***fNIRSSignalGenerator***

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



class fNIRSSignalGenerator:
	'''
	A basic class to generate synthetic fNIRS signals.
	'''

	def __init__(self, nSamples = 1, nChannels = 1):
		'''
		Class constructor.
		
		
		:Properties:
		
		data: The fNIRS data tensor.
		#frequency_bands: The EEG frequency bands.
		
		
		:Parameters:
		
		:param nSamples: Number of temporal samples.
			Optional. Default is 1.
		:type nSamples: int (positive)
		:param nChannels: Number of channels.
			Optional. Default is 1.
		:type nChannels: int (positive)
		'''
		
		#Ensure all properties exist
		self.__data = np.zeros((0,0,0),dtype=float)
		self.__samplingRate = 10 #[Hz]

		#Check parameters
		if type(nSamples) is not int:
			msg = self.getClassName() + ':__init__: Unexpected parameter type for parameter ''nSamples''.'
			raise ValueError(msg)
		if nSamples < 0:
			msg = self.getClassName() + ':__init__: Unexpected parameter value for parameter ''nSamples''.'
			raise ValueError(msg)
		if type(nChannels) is not int:
			msg = self.getClassName() + ':__init__: Unexpected parameter type for parameter ''nChannels''.'
			raise ValueError(msg)
		if nChannels < 0:
			msg = self.getClassName() + ':__init__: Unexpected parameter value for parameter ''nChannels''.'
			raise ValueError(msg)
		
		#Initialize
		self.data = np.zeros((nSamples,nChannels,2),dtype=float)
		
		return
	#end __init__(self, nSamples = 1, nChannels = 1)


	#Properties getters/setters
	#
	# Remember: Sphinx ignores docstrings on property setters so all
	#documentation for a property must be on the @property method

	@property
	def data(self): #data getter
		'''
		The data tensor.
		
		The data tensor always have 3 dimensions, namely:
		
		* Temporal (temporal samples)
		* Spatial (channels)
		* Signals (for fNIRS this is fixed to 2; the oxygenated (HbO2) and the deoxygenated (HHb) hemoglobin)
		
		:getter: Gets the data.
		:setter: Sets the data.
		:type: int
		'''

		return copy.deepcopy(self.__data)
	#end data(self)

	@data.setter
	def data(self,newData): #data setter

		#Check parameters
		if type(newData) is not np.ndarray:
			msg = self.getClassName() + ':data: Unexpected attribute type.'
			raise ValueError(msg)
		if newData.ndim != 3:
			msg = self.getClassName() + ':data: Unexpected attribute value. ' \
					+ 'Data tensor must be 3D <temporal, spatial, signal>'
			raise ValueError(msg)
		if newData.shape[2] != 2:
			msg = self.getClassName() + ':data: Unexpected attribute value. ' \
					+ 'Number of signals in fNIRS must be 2.'
			raise ValueError(msg)
			
		self.__data = copy.deepcopy(newData)

		return None
	#end data(self,newData)


	@property
	def nChannels(self): #nChannels getter
		'''
		Number of channels.
		
		When setting the number of channels:
		
		* if the number of channels is smaller than 
		the current number of channels, a warning is issued
		and the channels indexed rightmost in the data tensor will be
		removed. 
		* if the number of channels is greater than 
		the current number of channels, the new channels will
		be filled with zeros.
		
		
		:getter: Gets the number of channels.
		:setter: Sets the number of channels.
		:type: int
		'''

		return self.__data.shape[1]
	#end nChannels(self)

	@nChannels.setter
	def nChannels(self,newNChannels): #nChannels setter

		#Check parameters
		if type(newNChannels) is not int:
			msg = self.getClassName() + ':nChannels: Unexpected attribute type.'
			raise ValueError(msg)
		if newNChannels < 0:
			msg = self.getClassName() + ':nChannels: Unexpected attribute value. Number of channels must be greater or equal than 0.'
			raise ValueError(msg)
		
		if newNChannels > self.nChannels:
			#Add channels with zeros
			tmpNChannels = newNChannels-self.nChannels
			tmpData = np.zeros((self.nSamples,tmpNChannels,1),dtype=float)
			self.data = np.concatenate((self.data,tmpData), axis=1)
		elif newNChannels < self.nChannels:
			msg = self.getClassName() + ':nChannels: New number of channels is smaller than current number of channels. Some data will be lost.'
			warnings.warn(msg,RuntimeWarning)
			self.data = copy.deepcopy(self.data[:,0:newNChannels,:])

		return None
	#end nChannels(self,newNChannels)


	@property
	def nSamples(self): #nSamples getter
		'''
		Number of temporal samples.
		
		When setting the number of temporal samples:
		
		* if the number of temporal samples is smaller than 
		the current number of temporal samples, a warning is issued
		and the last temporal samples will be removed. 
		* if the number of temporal samples is greater than 
		the current number of temporal samples, the new temporal samples will
		be filled with zeros.
		
		
		:getter: Gets the number of temporal samples.
		:setter: Sets the number of temporal samples.
		:type: int
		'''

		return self.__data.shape[0]
	#end nSamples(self)

	@nSamples.setter
	def nSamples(self,newNSamples): #nSamples setter

		#Check parameters
		if type(newNSamples) is not int:
			msg = self.getClassName() + ':nSamples: Unexpected attribute type.'
			raise ValueError(msg)
		if newNSamples < 0:
			msg = self.getClassName() + ':nSamples: Unexpected attribute value. Number of temporal samples must be greater or equal than 0.'
			raise ValueError(msg)
		
		if newNSamples > self.nSamples:
			#Add channels with zeros
			tmpNSamples = newNSamples-self.nSamples
			tmpData = np.zeros((tmpNSamples,self.nChannels,1),dtype=float)
			self.data = np.concatenate((self.data,tmpData), axis=0)
		elif newNSamples < self.nSamples:
			msg = self.getClassName() + ':nSamples: New number of temporal samples is smaller than current number of temporal samples. Some data will be lost.'
			warnings.warn(msg,RuntimeWarning)
			self.data = copy.deepcopy(self.data[0:newNSamples,:,:])

		return None
	#end nSamples(self,newNSamples)


	@property
	def samplingRate(self): #samplingrate getter
		'''
		Sampling rate at which the synthetic data will be generated.
		
		:getter: Gets the sampling rate.
		:setter: Sets the sampling rate.
		:type: float
		'''

		return self.__samplingRate
	#end samplingRate(self)

	@samplingRate.setter
	def samplingRate(self,newSamplingRate): #samplingrate setter

		#Check parameters
		if type(newSamplingRate) is int:
			newSamplingRate = float(newSamplingRate)
		if type(newSamplingRate) is not float:
			msg = self.getClassName() + ':samplingrate: Unexpected attribute type.'
			raise ValueError(msg)
		if newSamplingRate <= 0:
			msg = self.getClassName() + ':samplingrate: Unexpected attribute value. ' \
					+ 'Sampling rate must be strictly positive.'
			raise ValueError(msg)
			
		self.__samplingRate = newSamplingRate

		return None
	#end samplingRate(self,newSamplingRate)


	#Private methods
	
	
	#Protected methods
	

	#Public methods

	def getClassName(self):
		'''Gets the class name.
		
		:return: The class name
		:rtype: str
		'''

		return type(self).__name__
	#end getClassName(self)


	def addStimulusResult(self, boxCarList=list(), channelsList=list(), initSample=0, endSample=-1, \
						   tau_p=6, tau_d=10, amplitudeScalingFactor=6):
		'''
		Adds a stimulus result to the data tensor.

		This method calls :meth:`generateStimulusResult` for generating
		the new synthetic data corresponding to the stimulus whose times of occurrence
		are on the supplied boxcar.
		Here, such newly generated data tensor is added to the class :attr:`data`.

		:Parameters:

		:param boxCarList: List of tuples. Each tuple is a pair (xi, yi).
			(xi, yi) is an interval where the boxcar is equal to 1.
			0 <= xi, yi < nSamples/samplingRate, xi < yi.
			Default is the empty list.
		:type boxCarList: list
		:param channelsList: List of channels affected. Default is the empty list.
		:type channelsList: list
		:param initSample: Initial temporal sample. Default is 0.
		:type initSample: int (positive)
		:param endSample: Last temporal sample. A positive value
			explicitly indicates a sample. A value -1 indicates the last
			sample of :attr:`data`. If not -1, then the endSample must be
			greater than the initSample. Default is -1.
		:type endSample: int (positive or -1)
		:param tau_p: stands for the first peak delay, which is basically set to 6 sec. Default is 6.
		:type tau_p: int (positive)
		:param tau_d: stands for the second peak delay, which is basically set to 10 sec. Default is 10.
			Represents the delay of undershoot to response.
		:type tau_d: int (positive)
		:param amplitudeScalingFactor: A scaling factor for the amplitude.
			It is the amplitude ratio between the first and second peaks.
			It was set to 6 sec. as in typical fMRI studies.
			Default is 6.
		:type amplitudeScalingFactor: float (positive)

		:return: None
		:rtype: NoneType
		'''

		# Check parameters
		if type(channelsList) is not list:
			msg = self.getClassName() + ':addStimulusResult: Unexpected parameter type for parameter ''channelList''.'
			raise ValueError(msg)
		for elem in channelsList:
			if type(elem) is not int:
				msg = self.getClassName() + ':addStimulusResult: Unexpected parameter value for parameter ''channelList''.'
				raise ValueError(msg)
			if elem < 0 or elem >= self.nChannels:  # Ensure the nChannels exist
				msg = self.getClassName() + ':addStimulusResult: Unexpected parameter value for parameter ''channelList''.'
				raise ValueError(msg)

		if type(initSample) is not int:
			msg = self.getClassName() + ':addStimulusResult: Unexpected parameter type for parameter ''initSample''.'
			raise ValueError(msg)
		if initSample < 0 or initSample >= self.nSamples:  # Ensure the nSamples exist
			msg = self.getClassName() + ':addStimulusResult: Unexpected parameter value for parameter ''initSample''.'
			raise ValueError(msg)

		if type(endSample) is not int:
			msg = self.getClassName() + ':addStimulusResult: Unexpected parameter type for parameter ''endSample''.'
			raise ValueError(msg)
		if endSample < -1 or endSample >= self.nSamples:  # Ensure the nSamples exist
			msg = self.getClassName() + ':addStimulusResult: Unexpected parameter value for parameter ''endSample''.'
			raise ValueError(msg)
		if endSample == -1:  # If -1, substitute by the maximum last sample
			endSample = self.nSamples - 1
		if endSample <= initSample:  # Ensure the endSample is posterior to the initSample
			msg = self.getClassName() + ':addStimulusResult: Unexpected parameter value for parameter ''endSample''.'
			raise ValueError(msg)
		#No need to type check boxCarList, tau_p, tau_d and amplitudeScalingFactor as these
		#are passed to method generateStimulusResult.

		channelsList = list(set(channelsList))  # Unique and sort elements
		nChannels = len(channelsList)
		nSamples = endSample - initSample
		tmpData = self.generateStimulusResult(boxCarList=boxCarList, \
											  nSamples=nSamples, \
											  nChannels=nChannels, \
											  tau_p=tau_p, \
											  tau_d=tau_d, \
											  amplitudeScalingFactor=amplitudeScalingFactor)
		self.__data[initSample:endSample, channelsList, :] = \
			self.__data[initSample:endSample, channelsList, :] + tmpData

		return

	# end addStimulusResult(self,channelsList = list(), initSample = 0, ... , amplitudeScalingFactor=6)


	def generateStimulusResult(self, boxCarList=list(), nSamples = 100, nChannels = 1, \
							    tau_p = 6, tau_d = 10, amplitudeScalingFactor = 6):
		'''
		Generates synthetic data for the stimulus whose times of occurrence are on the supplied boxcar

		:Parameters:

		:param boxCarList: List of tuples. Each tuple is a pair (xi, yi).
			(xi, yi) is an interval where the boxcar is equal to 1.
			0 <= xi, yi < nSamples/samplingRate, xi < yi.
			Default is the empty list.
		:type boxCarList: list
		:param nSamples: Number of temporal samples.
			Optional. Default is 1.
		:type nSamples: int (positive)
		:param nChannels: Number of channels.
			Optional. Default is 1.
		:type nChannels: int (positive)
		:param tau_p: stands for the first peak delay, which is basically set to 6 sec. Default is 6.
		:type tau_p: int (positive)
		:param tau_d: stands for the second peak delay, which is basically set to 10 sec. Default is 10.
			Represents the delay of undershoot to response.
		:type tau_d: int (positive)
		:param amplitudeScalingFactor: A scaling factor for the amplitude.
			It is the amplitude ratio between the first and second peaks.
			It was set to 6 sec. as in typical fMRI studies.
			Default is 6.
		:type amplitudeScalingFactor: float (positive)

		:return: A data tensor.
		:rtype: np.ndarray
		'''

		#Check parameters
		if type(boxCarList) is not list:
			msg = self.getClassName() + ':addStimulusResult: Unexpected parameter type for parameter ''boxCarList''.'
			raise ValueError(msg)
		for elem in boxCarList:
			if type(elem) is not tuple:
				msg = self.getClassName() + ':addStimulusResult: Unexpected parameter value for parameter ''boxCarList''.'
				raise ValueError(msg)
			if elem[0] < 0 or elem[0] >= self.nSamples/self.samplingRate:  # Ensure 0 <= xi < nSamples/samplingRate
				msg = self.getClassName() + ':addStimulusResult: Unexpected parameter value for parameter ''boxCarList''.'
				raise ValueError(msg)
			if elem[1] < 0 or elem[1] >= self.nSamples/self.samplingRate:  # Ensure 0 <= yi < nSamples/samplingRate
				msg = self.getClassName() + ':addStimulusResult: Unexpected parameter value for parameter ''boxCarList''.'
				raise ValueError(msg)
			if elem[0] >= elem[1]:  # Ensure xi < yi
				msg = self.getClassName() + ':addStimulusResult: Unexpected parameter value for parameter ''boxCarList''.'
				raise ValueError(msg)

		if type(nSamples) is not int:
			msg = self.getClassName() + ':generateStimulusResult: Unexpected parameter type for parameter ''nSamples''.'
			raise ValueError(msg)
		if nSamples <= 0:
			msg = self.getClassName() + ':generateStimulusResult: Unexpected parameter value for parameter ''nSamples''.'
			raise ValueError(msg)

		if type(nChannels) is not int:
			msg = self.getClassName() + ':generateStimulusResult: Unexpected parameter type for parameter ''nChannels''.'
			raise ValueError(msg)
		if nChannels <= 0:
			msg = self.getClassName() + ':generateStimulusResult: Unexpected parameter value for parameter ''nChannels''.'
			raise ValueError(msg)

		if type(tau_p) is not int:
			msg = self.getClassName() + ':generateStimulusResult: Unexpected parameter type for parameter ''tau_p''.'
			raise ValueError(msg)
		if tau_p <= 0:
			msg = self.getClassName() + ':generateStimulusResult: Unexpected parameter value for parameter ''tau_p''.'
			raise ValueError(msg)

		if type(tau_d) is not int:
			msg = self.getClassName() + ':generateStimulusResult: Unexpected parameter type for parameter ''tau_d''.'
			raise ValueError(msg)
		if tau_d <= 0:
			msg = self.getClassName() + ':generateStimulusResult: Unexpected parameter value for parameter ''tau_d''.'
			raise ValueError(msg)

		if type(amplitudeScalingFactor) is int:
			amplitudeScalingFactor = float(amplitudeScalingFactor)
		if type(amplitudeScalingFactor) is not float:
			msg = self.getClassName() + ':generateStimulusResult: Unexpected parameter type for parameter ''amplitudeScalingFactor''.'
			raise ValueError(msg)
		if amplitudeScalingFactor <= 0:
			msg = self.getClassName() + ':generateStimulusResult: Unexpected parameter value for parameter ''amplitudeScalingFactor''.'
			raise ValueError(msg)

		timestamps = np.arange(0, nSamples/self.samplingRate, \
								  1/self.samplingRate, dtype = float)

		ntimestamps = len(timestamps)
		boxCar = np.zeros(ntimestamps) #creation of the boxcar with 0s

		for elem in boxCarList:
			i = np.searchsorted(timestamps, elem[0])
			j = np.searchsorted(timestamps, elem[1]) + 1
			boxCar[i:j] = 1

		plt.plot(boxCar, color='black')
		plt.show()


		HRF = ( pow(timestamps, tau_p) * np.exp(-1 * timestamps) ) / math.factorial(tau_p)
		HRF = HRF - ( pow(timestamps, tau_p+tau_d) * np.exp(-1 * timestamps) ) / \
			  		( amplitudeScalingFactor * math.factorial(tau_p+tau_d) )

		# This is only for visualizing the doble gamma function
		timestamps1 = np.arange(0, 25, 0.1, dtype=float)
		HRF1 = ( pow(timestamps1, tau_p) * np.exp(-1 * timestamps1) ) / math.factorial(tau_p)
		HRF1 = HRF1 - ( pow(timestamps1, tau_p+tau_d) * np.exp(-1 * timestamps1) ) / \
			  		( amplitudeScalingFactor * math.factorial(tau_p+tau_d) )
		plt.plot(HRF1, color='green')
		plt.show()

		HbO2 = np.convolve(boxCar, HRF)

		plt.plot(HbO2, color='red')
		plt.show()

		HbO2 = HbO2.reshape(-1, 1) #Reshape to column vector
		HbO2 = np.tile(HbO2, nChannels)

		HHb = (-1/3) * HbO2

		synthData = np.zeros((nSamples, nChannels, 2)) #The synthetic data tensor

		synthData[:, :, 0] = synthData[:, :, 0] + HbO2[0:nSamples, :]
		synthData[:, :, 1] = synthData[:, :, 1] + HHb[0:nSamples, :]

		return synthData
	#end generateStimulusResult(self, nSamples = 100, nChannels = 1, ... , amplitudeScalingFactor=6)


	def addGaussianNoise(self, channelsList=list(), initSample=0, endSample=-1):
		'''
		Adds Gaussian noise to the data tensor.
		The generated noise is added to the class :attr:`data`.

		:Parameters:

		:param channelsList: List of channels affected. Default is the empty list.
		:type channelsList: list
		:param initSample: Initial temporal sample. Default is 0.
		:type initSample: int (positive)
		:param endSample: Last temporal sample. A positive value
			explicitly indicates a sample. A value -1 indicates the last
			sample of :attr:`data`. If not -1, then the endSample must be
			greater than the initSample. Default is -1.
		:type endSample: int (positive or -1)

		:return: None
		:rtype: NoneType
		'''

		#Check parameters
		if type(channelsList) is not list:
			msg = self.getClassName() + ':addGaussianNoise: Unexpected parameter type for parameter ''channelList''.'
			raise ValueError(msg)
		for elem in channelsList:
			if type(elem) is not int:
				msg = self.getClassName() + ':addGaussianNoise: Unexpected parameter value for parameter ''channelList''.'
				raise ValueError(msg)
			if elem < 0 or elem >= self.nChannels:  # Ensure the nChannels exist
				msg = self.getClassName() + ':addGaussianNoise: Unexpected parameter value for parameter ''channelList''.'
				raise ValueError(msg)

		if type(initSample) is not int:
			msg = self.getClassName() + ':addGaussianNoise: Unexpected parameter type for parameter ''initSample''.'
			raise ValueError(msg)
		if initSample < 0 or initSample >= self.nSamples:  # Ensure the nSamples exist
			msg = self.getClassName() + ':addGaussianNoise: Unexpected parameter value for parameter ''initSample''.'
			raise ValueError(msg)

		if type(endSample) is not int:
			msg = self.getClassName() + ':addGaussianNoise: Unexpected parameter type for parameter ''endSample''.'
			raise ValueError(msg)
		if endSample < -1 or endSample >= self.nSamples:  # Ensure the nSamples exist
			msg = self.getClassName() + ':addGaussianNoise: Unexpected parameter value for parameter ''endSample''.'
			raise ValueError(msg)
		if endSample == -1:  # If -1, substitute by the maximum last sample
			endSample = self.nSamples - 1
		if endSample <= initSample:  # Ensure the endSample is posterior to the initSample
			msg = self.getClassName() + ':addGaussianNoise: Unexpected parameter value for parameter ''endSample''.'
			raise ValueError(msg)

		channelsList = list(set(channelsList))  # Unique and sort elements
		nChannels = len(channelsList)
		nSamples = endSample - initSample

		timestamps = np.arange(0, nSamples/self.samplingRate, \
								  1/self.samplingRate, dtype = float)
		timestamps = timestamps.reshape(-1, 1) #Reshape to column vector
		timestamps = np.tile(timestamps,nChannels)

		#noiseHbO2 = np.random.normal(0, 1, timestamps.shape)
		noiseHbO2 = np.random.normal(0, 0.3, timestamps.shape)
		noiseHHb  = np.random.normal(0, 0.3, timestamps.shape)

		noiseHbO2_plot = np.random.normal(0, 0.3, timestamps.shape)

		plt.plot(noiseHbO2_plot[0:nSamples,0], color='blue')
		plt.show()

		#plt.plot(noiseHHb[0:nSamples,0], color='blue')
		#plt.show()

		self.__data[0:nSamples,channelsList,0] = \
				self.__data[0:nSamples,channelsList,0] + noiseHbO2

		self.__data[0:nSamples,channelsList,1] = \
				self.__data[0:nSamples,channelsList,1] + noiseHHb

		return
	#end addGaussianNoise(self, channelsList=list(), initSample=0, endSample=-1)


	def addNoiseBreathingRate(self, channelsList=list(), initSample=0, endSample=-1, \
							   frequencyResolutionStep = 0.01):
		'''
		Adds noise of breathing rate to the data tensor.
		The generated noise is added to the class :attr:`data`.

		:Parameters:

		:param channelsList: List of channels affected. Default is the empty list.
		:type channelsList: list
		:param initSample: Initial temporal sample. Default is 0.
		:type initSample: int (positive)
		:param endSample: Last temporal sample. A positive value
			explicitly indicates a sample. A value -1 indicates the last
			sample of :attr:`data`. If not -1, then the endSample must be
			greater than the initSample. Default is -1.
		:type endSample: int (positive or -1)
		:param frequencyResolutionStep: The step for generating evenly spaced values
			within the interval of frequencies of the noise to be simulated.
			Optional. Default is 0.01.
		:type frequencyResolutionStep: float (positive)

		:return: None
		:rtype: NoneType
		'''

		#Check parameters
		if type(channelsList) is not list:
			msg = self.getClassName() + ':addNoiseBreathingRate: Unexpected parameter type for parameter ''channelList''.'
			raise ValueError(msg)
		for elem in channelsList:
			if type(elem) is not int:
				msg = self.getClassName() + ':addNoiseBreathingRate: Unexpected parameter value for parameter ''channelList''.'
				raise ValueError(msg)
			if elem < 0 or elem >= self.nChannels:  # Ensure the nChannels exist
				msg = self.getClassName() + ':addNoiseBreathingRate: Unexpected parameter value for parameter ''channelList''.'
				raise ValueError(msg)

		if type(initSample) is not int:
			msg = self.getClassName() + ':addNoiseBreathingRate: Unexpected parameter type for parameter ''initSample''.'
			raise ValueError(msg)
		if initSample < 0 or initSample >= self.nSamples:  # Ensure the nSamples exist
			msg = self.getClassName() + ':addNoiseBreathingRate: Unexpected parameter value for parameter ''initSample''.'
			raise ValueError(msg)

		if type(endSample) is not int:
			msg = self.getClassName() + ':addNoiseBreathingRate: Unexpected parameter type for parameter ''endSample''.'
			raise ValueError(msg)
		if endSample < -1 or endSample >= self.nSamples:  # Ensure the nSamples exist
			msg = self.getClassName() + ':addNoiseBreathingRate: Unexpected parameter value for parameter ''endSample''.'
			raise ValueError(msg)
		if endSample == -1:  # If -1, substitute by the maximum last sample
			endSample = self.nSamples - 1
		if endSample <= initSample:  # Ensure the endSample is posterior to the initSample
			msg = self.getClassName() + ':addNoiseBreathingRate: Unexpected parameter value for parameter ''endSample''.'
			raise ValueError(msg)

		if type(frequencyResolutionStep) is not float:
			msg = self.getClassName() + ':addNoiseBreathingRate: Unexpected parameter type for parameter ''frequencyResolutionStep''.'
			raise ValueError(msg)
		if frequencyResolutionStep <= 0:
			msg = self.getClassName() + ':addNoiseBreathingRate: Unexpected parameter value for parameter ''frequencyResolutionStep''.'
			raise ValueError(msg)

		channelsList = list(set(channelsList))  # Unique and sort elements
		nChannels = len(channelsList)
		nSamples = endSample - initSample

		tmpData = np.zeros((nSamples, nChannels, 2)) #The temporal data tensor for saving the generated noise

		timestamps = np.arange(0, nSamples/self.samplingRate, \
								  1/self.samplingRate, dtype = float)
		timestamps = timestamps.reshape(-1, 1) #Reshape to column vector
		timestamps = np.tile(timestamps,nChannels)
		#timestamps = np.tile(timestamps, [1, 1, 2]) # for generating a timestamps tensor

		frequencySet = np.arange(0.22-2*0.07, 0.22+2*0.07+frequencyResolutionStep, \
								 frequencyResolutionStep, dtype = float)   # From paper (Elwell et al., 1999)
		amplitudeScalingFactor = 1   # estandarizada para la distribución tenga media 0 y desv 1 z-score
		for freq in frequencySet:
			#Amplitude. One random amplitude per channel
			A = amplitudeScalingFactor*np.random.rand(1,nChannels)
			A = np.tile(A,[nSamples,1])
			#Phase [rad]. One random phase per channel
			theta = 2* math.pi * np.random.rand(1,nChannels) - math.pi
			theta = np.tile(theta,[nSamples,1])
			#theta = 0
			#Generate the fundamental signal
			tmpSin = A * np.sin(2*math.pi*freq*timestamps+theta)
			#Elment-wise multiplication with the amplitude
				#NOTE: In python NumPy, a*b among ndarrays is the
				#element-wise product. For matrix multiplication, one
				#need to do np.matmul(a,b)
			tmpData[:,:,0] = tmpData[:,:,0] + tmpSin   # definir constante para HbO2 y otra para Hbb
			tmpData[:,:,1] = tmpData[:,:,1] + (-1/3)*tmpSin

		#plt.plot(tmpSin[0:nSamples,0], color='blue')
		#plt.show()

		# al tener la señal final se debe estandarizar z_score  eliminar la media y dividir por la desv. stand

		self.__data[0:nSamples,channelsList,:] = \
				self.__data[0:nSamples,channelsList,:] + tmpData

		return
	#end addNoiseBreathingRate(self, channelsList=list(), initSample=0, ... , frequencyResolutionStep = 0.01)


	def execute(self):
		'''
		Generates the synthetic fNIRS data from the properties
		information.

		This method calls :meth:`generateStimulusResult` for generating
		the new synthetic data.

		:return: A 3D data tensor
		:rtype: np.ndarray
		'''
		self.addStimulusResult(boxCarList = [(35, 45), (105, 120), (175, 195), (240, 265)], \
		#self.addStimulusResult(boxCarList = [(35, 45), (105, 120), (240, 265)], \
							   channelsList=list(range(0, self.nChannels)), \
							   initSample=0, endSample=-1, \
							   tau_p=6, tau_d=10, \
							   amplitudeScalingFactor=6)

		#self.addStimulusResult(boxCarList = [(175, 195)], \
		#					   channelsList=list(range(0, self.nChannels)), \
		#					   initSample=0, endSample=-1, \
		#					   tau_p=6, tau_d=10, \
		#					   amplitudeScalingFactor=6)

		plotSyntheticfNIRS(self.data)

		self.addNoiseBreathingRate(channelsList=list(range(0, self.nChannels)), \
								   initSample=0, endSample=-1, \
								   frequencyResolutionStep = 0.01)

		plotSyntheticfNIRS(self.data)

		self.addGaussianNoise(channelsList=list(range(0, self.nChannels)), \
							  initSample=0, endSample=-1)

		return copy.deepcopy(self.data)
	#end execute(self)


def plotSyntheticfNIRS(tensor):
	'''
	Quick rendering of the synthetic fNIRS data tensor.
	'''

	nChannels = tensor.shape[1]
	for iCh in range(0,nChannels):
		plt.plot(tensor[:,iCh,0]+20*iCh, color='red')
		plt.plot(tensor[:,iCh,1]+20*iCh, color='blue')
	plt.xlabel('Time [samples]')
	plt.ylabel('Channels [A.U.]')
	plt.show()

	return
#end plotSyntheticfNIRS(tensor)


def main():
	sg = fNIRSSignalGenerator(nSamples = 3000, nChannels = 4)
	sg.execute()
	#print(sg.data)
	plotSyntheticfNIRS(sg.data)
#end main()


if __name__ == '__main__':
	main()


