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

#from scipy import stats

#import CONST

from optodeArrayInfo import optodeArrayInfo

from channelLocationMap import channelLocationMap

# Class fNIRSSignalGenerator is a subclass of channelLocationMap
class fNIRSSignalGenerator(channelLocationMap):
	'''
	A basic class to generate synthetic fNIRS signals.
	'''

	#def __init__(self, nSamples = 1, nChannels = 1):   # __init__ used before the creation of the class channelLocationMap
	def __init__(self, nSamples=1, id=1, description='ChannelLocationMap0001', nChannels=1, nOptodes=1,
				 chLocations=np.array([[np.NaN, np.NaN, np.NaN]]),
				 optodesLocations=np.array([[np.NaN, np.NaN, np.NaN]]),
				 optodesTypes=np.array([np.NaN]), referencePoints=dict(), surfacePositioningSystem='UI 10/20',
				 chSurfacePositions=tuple(('',)), optodesSurfacePositions=tuple(('',)),
				 chOptodeArrays=np.array([np.NaN]),
				 optodesOptodeArrays=np.array([np.NaN]), pairings=np.array([[np.NaN, np.NaN]]),
				 optodeArrays=np.array([optodeArrayInfo()])):
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
		# Initialization of an object of the superclass channelLocationMap
		super().__init__(id = id, description = description, nChannels = nChannels, nOptodes = nOptodes,
						 chLocations = chLocations, optodesLocations = optodesLocations,
						 optodesTypes = optodesTypes, referencePoints = referencePoints,
						 surfacePositioningSystem = surfacePositioningSystem,
						 chSurfacePositions = chSurfacePositions, optodesSurfacePositions = optodesSurfacePositions,
						 chOptodeArrays = chOptodeArrays,
						 optodesOptodeArrays = optodesOptodeArrays, pairings = pairings,
						 optodeArrays = optodeArrays)

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

		# Create constant HBO2 = 0 and constant HHB = 1
		# Represent HbO2 (Oxi) and HHb (Desoxi) respectively
		# for the third component of the tensor data.
		self.__HBO2 = 0
		self.__HHB  = 1

		return
	#end __init__(self, nSamples = 1, nChannels = 1)


	#Properties getters/setters
	#
	# Remember: Sphinx ignores docstrings on property setters so all
	#documentation for a property must be on the @property method

	# Note that python does not have constants nor static constants,
	# so in order to have a constant, a new property is defined
	# with only a getter method and the setter method raises an error message.
	@property
	def HBO2(self):  # HBO2 getter
		'''
		Constant HBO2 = 0
		Represents HbO2 (Oxi) for the third component of the tensor data.
		:getter: Gets constant HBO2.
		:type: int
		'''

		return 0
	# end HBO2(self)

	@HBO2.setter
	def HBO2(self, value):  # HBO2 setter
		'''
		Constant HBO2 = 0
		Represents HbO2 (Oxi) for the third component of the tensor data.
		:setter: Raise an error message because the value of constant HBO2 is being tried to be changed.
		:type: int
		'''

		msg = self.getClassName() + ':HBO2: ConstantError: Can not rebind const.'
		raise ValueError(msg)
	# end HBO2(self, value)

	@property
	def HHB(self):  # HHB getter
		'''
		Constant HHB = 1
		Represents HHb (Desoxi) for the third component of the tensor data.
		:getter: Gets constant HHB.
		:type: int
		'''

		return 1
	# end HHB(self)

	@HHB.setter
	def HHB(self, value):  # HHB setter
		'''
		Constant HHB = 1
		Represents HHb (Desoxi) for the third component of the tensor data.
		:setter: Raise an error message because the value of constant HHB is being tried to be changed.
		:type: int
		'''

		msg = self.getClassName() + ':HHB: ConstantError: Can not rebind const.'
		raise ValueError(msg)
	# end HHB(self, value)


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
		:type: numpy.ndarray [nSamples x nChannels x 2]
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
			tmpData = np.zeros((self.nSamples,tmpNChannels,2),dtype=float)
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
			tmpData = np.zeros((tmpNSamples,self.nChannels,2),dtype=float)
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


	def addStimulusResult(self, channelsList=list(), boxCarList=list(), initSample=0, endSample=-1, \
						   tau_p=6, tau_d=10, amplitudeScalingFactor=6, \
						   enableHbO2Channels = np.ones(1, dtype=int), \
						   enableHHbChannels = np.ones(1, dtype=int), \
						   enableHbO2Blocks = np.ones(1, dtype=int), \
						   enableHHbBlocks = np.ones(1, dtype=int)):
		'''
		Adds a stimulus result to the data tensor.
		This method calls :meth:`generateStimulusResult` for generating
		the new synthetic data corresponding to the stimulus whose times of occurrence
		are on the supplied boxcar.
		Here, such newly generated data tensor is added to the class :attr:`data`.
		:Parameters:
		:param channelsList: List of channels affected. Default is the empty list.
		:type channelsList: list
		:param boxCarList: List of tuples. Each tuple is a pair (xi, yi).
			(xi, yi) is an interval where the boxcar is equal to 1.
			0 <= xi, yi < nSamples/samplingRate, xi < yi.
			Default is the empty list.
		:type boxCarList: list
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
		:param enableHbO2Channels: A vector whose length is nChannels. Each position contains an integer value: 1 or 0.
			The value 1 indicates that the corresponding channel has the HbO2 signal enabled, and 0 indicates otherwise.
			Default is array([1]). This array of 1´s is extended automatically, by the compiler, to the number of
			channels (columns) and the number of samples (rows) when the operation * (element-wise matrix multiplication)
			is applied.
		:type numpy.ndarray
		:param enableHHbChannels: A vector whose length is nChannels. Each position contains an integer value: 1 or 0.
			The value 1 indicates that the corresponding channel has the HHb signal enabled, and 0 indicates otherwise.
			Default is array([1]). This array of 1´s is extended automatically, by the compiler, to the number of
			channels (columns) and the number of samples (rows) when the operation * (element-wise matrix multiplication)
			is applied.
		:type numpy.ndarray
		:param enableHbO2Blocks: A vector whose length is nBlocks. Each position contains an integer value: 1 or 0.
			The value 1 indicates that the corresponding block has the HbO2 signal enabled, and 0 indicates otherwise.
			Default is array([1]). This array of 1´s is extended automatically, by the compiler, to the number of
			blocks (columns) and the number of samples (rows) when the operation * (element-wise matrix multiplication)   OJO
			is applied.
		:type numpy.ndarray
		:param enableHHbBlocks: A vector whose length is nBlocks. Each position contains an integer value: 1 or 0.
			The value 1 indicates that the corresponding block has the HHb signal enabled, and 0 indicates otherwise.
			Default is array([1]). This array of 1´s is extended automatically, by the compiler, to the number of
			blocks (columns) and the number of samples (rows) when the operation * (element-wise matrix multiplication)   OJO
			is applied.
		:type numpy.ndarray
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
		#No need to type check boxCarList, tau_p, tau_d, amplitudeScalingFactor, enableHbO2Channels, enableHHbChannels,
		#enableHbO2Blocks, and enableHHbBlocks as these
		#are passed to method generateStimulusResult.

		channelsList = list(set(channelsList))  # Unique and sort elements
		nChannels = len(channelsList)
		nSamples = endSample - initSample
		tmpData = self.generateStimulusResult(boxCarList=boxCarList, \
											  nSamples=nSamples, \
											  nChannels=nChannels, \
											  tau_p=tau_p, \
											  tau_d=tau_d, \
											  amplitudeScalingFactor=amplitudeScalingFactor, \
											  enableHbO2Channels=enableHbO2Channels, \
											  enableHHbChannels=enableHHbChannels, \
											  enableHbO2Blocks=enableHbO2Blocks, \
											  enableHHbBlocks=enableHHbBlocks)
		self.__data[initSample:endSample, channelsList, :] = \
			self.__data[initSample:endSample, channelsList, :] + tmpData

		return
	# end addStimulusResult(self,channelsList = list(), boxCarList=list(), initSample = 0, ... , enableHHbBlocks = np.ones(1, dtype=int))


	def double_gamma_function(self, timestamps = np.arange(25, dtype=float), \
							    tau_p = 6, tau_d = 10, amplitudeScalingFactor = 6):
		'''
		Generates double gamma function in the domain of timestamps
		:Parameters:
		:param timestamps: array of evenly spaced values representing temporal samples.
			Optional. Default is array([0., 1., 2., ..., 24.]).
		:type timestamps: array of float (positive)
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
		:return: An array.
		:rtype: numpy.ndarray
		'''

		#Check parameters
		if type(timestamps) is not np.ndarray:
			msg = self.getClassName() + ':double_gamma_function: Unexpected parameter type for parameter ''timestamps''.'
			raise ValueError(msg)

		if type(tau_p) is not int:
			msg = self.getClassName() + ':double_gamma_function: Unexpected parameter type for parameter ''tau_p''.'
			raise ValueError(msg)
		if tau_p <= 0:
			msg = self.getClassName() + ':double_gamma_function: Unexpected parameter value for parameter ''tau_p''.'
			raise ValueError(msg)

		if type(tau_d) is not int:
			msg = self.getClassName() + ':double_gamma_function: Unexpected parameter type for parameter ''tau_d''.'
			raise ValueError(msg)
		if tau_d <= 0:
			msg = self.getClassName() + ':double_gamma_function: Unexpected parameter value for parameter ''tau_d''.'
			raise ValueError(msg)

		if type(amplitudeScalingFactor) is int:
			amplitudeScalingFactor = float(amplitudeScalingFactor)
		if type(amplitudeScalingFactor) is not float:
			msg = self.getClassName() + ':double_gamma_function: Unexpected parameter type for parameter ''amplitudeScalingFactor''.'
			raise ValueError(msg)
		if amplitudeScalingFactor <= 0:
			msg = self.getClassName() + ':double_gamma_function: Unexpected parameter value for parameter ''amplitudeScalingFactor''.'
			raise ValueError(msg)

		HRF = ( pow(timestamps, tau_p) * np.exp(-1 * timestamps) ) / math.factorial(tau_p)
		HRF = HRF - ( pow(timestamps, tau_p+tau_d) * np.exp(-1 * timestamps) ) / \
			  		( amplitudeScalingFactor * math.factorial(tau_p+tau_d) )

		return HRF
	#end double_gamma_function(self, timestamps = np.arange(25, dtype=float), tau_p=6, ... , amplitudeScalingFactor=6)


	def generateStimulusResult(self, boxCarList=list(), nSamples = 100, nChannels = 1, \
							    tau_p = 6, tau_d = 10, amplitudeScalingFactor = 6, \
							    enableHbO2Channels=np.ones(1, dtype=int), \
							    enableHHbChannels =np.ones(1, dtype=int), \
							    enableHbO2Blocks  =np.ones(1, dtype=int), \
							    enableHHbBlocks   =np.ones(1, dtype=int)):

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
		:param enableHbO2Channels: A vector whose length is nChannels. Each position contains an integer value: 1 or 0.
			The value 1 indicates that the corresponding channel has the HbO2 signal enabled, and 0 indicates otherwise.
			Default is array([1]). This array of 1´s is extended automatically, by the compiler, to the number of
			channels (columns) and the number of samples (rows) when the operation * (element-wise matrix multiplication)
			is applied.
		:type numpy.ndarray
		:param enableHHbChannels: A vector whose length is nChannels. Each position contains an integer value: 1 or 0.
			The value 1 indicates that the corresponding channel has the HHb signal enabled, and 0 indicates otherwise.
			Default is array([1]). This array of 1´s is extended automatically, by the compiler, to the number of
			channels (columns) and the number of samples (rows) when the operation * (element-wise matrix multiplication)
			is applied.
		:type numpy.ndarray
		:param enableHbO2Blocks: A vector whose length is nBlocks. Each position contains an integer value: 1 or 0.
			The value 1 indicates that the corresponding block has the HbO2 signal enabled, and 0 indicates otherwise.
			Default is array([1]). This array of 1´s is extended automatically, by the compiler, to the number of
			blocks (columns) and the number of samples (rows) when the operation * (element-wise matrix multiplication)   OJO
			is applied.
		:type numpy.ndarray
		:param enableHHbBlocks: A vector whose length is nBlocks. Each position contains an integer value: 1 or 0.
			The value 1 indicates that the corresponding block has the HHb signal enabled, and 0 indicates otherwise.
			Default is array([1]). This array of 1´s is extended automatically, by the compiler, to the number of
			blocks (columns) and the number of samples (rows) when the operation * (element-wise matrix multiplication)   OJO
			is applied.
		:type numpy.ndarray
		:return: A data tensor.
		:rtype: numpy.ndarray
		'''

		#Check parameters
		if type(boxCarList) is not list:
			msg = self.getClassName() + ':generateStimulusResult: Unexpected parameter type for parameter ''boxCarList''.'
			raise ValueError(msg)
		for elem in boxCarList:
			if type(elem) is not tuple:
				msg = self.getClassName() + ':generateStimulusResult: Unexpected parameter value for parameter ''boxCarList''.'
				raise ValueError(msg)
			if elem[0] < 0 or elem[0] >= self.nSamples/self.samplingRate:  # Ensure 0 <= xi < nSamples/samplingRate
				msg = self.getClassName() + ':generateStimulusResult: Unexpected parameter value for parameter ''boxCarList''.'
				raise ValueError(msg)
			if elem[1] < 0 or elem[1] >= self.nSamples/self.samplingRate:  # Ensure 0 <= yi < nSamples/samplingRate
				msg = self.getClassName() + ':generateStimulusResult: Unexpected parameter value for parameter ''boxCarList''.'
				raise ValueError(msg)
			if elem[0] >= elem[1]:  # Ensure xi < yi
				msg = self.getClassName() + ':generateStimulusResult: Unexpected parameter value for parameter ''boxCarList''.'
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

		if type(enableHbO2Channels) is list:
			enableHbO2Channels = np.array(enableHbO2Channels)
		if type(enableHbO2Channels) is not np.ndarray:
			msg = self.getClassName() + ':generateStimulusResult: Unexpected parameter type for parameter ''enableHbO2Channels''.'
			raise ValueError(msg)
		for i in range(0, len(enableHbO2Channels)):
			if enableHbO2Channels[i] != 1 and enableHbO2Channels[i] != 0:
				msg = self.getClassName() + ':generateStimulusResult: Unexpected parameter type for parameter ''enableHbO2Channels''.'
				raise ValueError(msg)

		if type(enableHHbChannels) is list:
			enableHHbChannels = np.array(enableHHbChannels)
		if type(enableHHbChannels) is not np.ndarray:
			msg = self.getClassName() + ':generateStimulusResult: Unexpected parameter type for parameter ''enableHHbChannels''.'
			raise ValueError(msg)
		for i in range(0, len(enableHHbChannels)):
			if enableHHbChannels[i] != 1 and enableHHbChannels[i] != 0:
				msg = self.getClassName() + ':generateStimulusResult: Unexpected parameter type for parameter ''enableHHbChannels''.'
				raise ValueError(msg)

		if type(enableHbO2Blocks) is list:
			enableHbO2Blocks = np.array(enableHbO2Blocks)
		if type(enableHbO2Blocks) is not np.ndarray:
			msg = self.getClassName() + ':generateStimulusResult: Unexpected parameter type for parameter ''enableHbO2Blocks''.'
			raise ValueError(msg)
		for i in range(0, len(enableHbO2Blocks)):
			if enableHbO2Blocks[i] != 1 and enableHbO2Blocks[i] != 0:
				msg = self.getClassName() + ':generateStimulusResult: Unexpected parameter type for parameter ''enableHbO2Blocks''.'
				raise ValueError(msg)

		if type(enableHHbBlocks) is list:
			enableHHbBlocks = np.array(enableHHbBlocks)
		if type(enableHHbBlocks) is not np.ndarray:
			msg = self.getClassName() + ':generateStimulusResult: Unexpected parameter type for parameter ''enableHHbBlocks''.'
			raise ValueError(msg)
		for i in range(0, len(enableHHbBlocks)):
			if enableHHbBlocks[i] != 1 and enableHHbBlocks[i] != 0:
				msg = self.getClassName() + ':generateStimulusResult: Unexpected parameter type for parameter ''enableHHbBlocks''.'
				raise ValueError(msg)

		timestamps = np.arange(0, nSamples/self.samplingRate, \
								  1/self.samplingRate, dtype = float)

		ntimestamps = len(timestamps)
		boxCar      = np.zeros(ntimestamps) #creation of the boxcar with 0s
		boxCarHbO2  = np.zeros(ntimestamps) #creation of the boxcar for HbO2 with 0s
		boxCarHHb   = np.zeros(ntimestamps) #creation of the boxcar for HHb with 0s

		iBlock = 0
		for elem in boxCarList:
			i = np.searchsorted(timestamps, elem[0])
			j = np.searchsorted(timestamps, elem[1]) + 1
			boxCar[i:j] = 1
			if enableHbO2Blocks[iBlock] == 1:
				boxCarHbO2[i:j] = 1
			if enableHHbBlocks[iBlock] == 1:
				boxCarHHb[i:j] = 1
			iBlock+=1

		plt.plot(boxCar, color='black')
		plt.title('BoxCar')
		plt.show()

		plt.plot(boxCarHbO2, color='red')
		plt.title('BoxCar with the enabled blocks for HbO2 signal')
		plt.show()

		plt.plot(boxCarHHb, color='blue')
		plt.title('BoxCar with the enabled blocks for HHb signal')
		plt.show()

		HRF = self.double_gamma_function(timestamps, tau_p, tau_d, amplitudeScalingFactor)

		# This is only for visualizing the doble gamma function
		timestamps1 = np.arange(0, 25, 0.1, dtype=float)
		HRF1 = self.double_gamma_function(timestamps1, tau_p, tau_d, amplitudeScalingFactor)
		plt.plot(HRF1, color='green')
		plt.title('Double gamma function')
		plt.xlabel('Time [samples]')
		plt.ylabel('HRF')
		plt.show()

		HbO2 = np.convolve(boxCarHbO2, HRF)

		plt.plot(HbO2, color='red')
		plt.title('Result of the convolution of HRF and the BoxCar for HbO2 signal')
		plt.show()

		HbO2 = HbO2.reshape(-1, 1) #Reshape to column vector
		HbO2 = np.tile(HbO2, nChannels)

		HbO2forHHb = np.convolve(boxCarHHb, HRF)

		plt.plot(HbO2forHHb, color='blue')
		plt.title('Result of the convolution of HRF and the BoxCar for HHb signal')
		plt.show()

		HbO2forHHb = HbO2forHHb.reshape(-1, 1) #Reshape to column vector
		HbO2forHHb = np.tile(HbO2forHHb, nChannels)

		HHb = (-1/3) * HbO2forHHb

		synthData = np.zeros((nSamples, nChannels, 2)) #The synthetic data tensor

		#print(HbO2[0:nSamples, :])
		#print(HHb[0:nSamples, :])

		#print(enableHHbChannels)

		#print(HHb[0:nSamples, :] * enableHHbChannels)

		synthData[:, :, self.HBO2] = synthData[:, :, self.HBO2] + HbO2[0:nSamples, :] * enableHbO2Channels
		synthData[:, :, self.HHB]  = synthData[:, :, self.HHB]  + HHb[0:nSamples, :] * enableHHbChannels

		return synthData
	#end generateStimulusResult(self, boxCarList=list(), nSamples = 100, nChannels = 1, ... , enableHHbChannels = np.ones(1, dtype=int))


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

		noiseHbO2 = np.random.normal(0, 1, timestamps.shape)
		#noiseHbO2 = np.random.normal(0, 0.3, timestamps.shape)
		noiseHHb  = np.random.normal(0, 1, timestamps.shape)

		noiseHbO2_plot = np.random.normal(0, 0.3, timestamps.shape)

		plt.plot(noiseHbO2_plot[0:nSamples,0], color='blue')
		plt.title('Gaussian Noise')
		plt.show()

		#plt.plot(noiseHHb[0:nSamples,0], color='blue')
		#plt.show()

		self.__data[0:nSamples,channelsList,0] = \
				self.__data[0:nSamples,channelsList,0] + noiseHbO2

		self.__data[0:nSamples,channelsList,1] = \
				self.__data[0:nSamples,channelsList,1] + noiseHHb

		return
	#end addGaussianNoise(self, channelsList=list(), initSample=0, endSample=-1)


	def addPhysiologicalNoise(self, channelsList=list(), initSample=0, endSample=-1, \
							   frequencyMean = 0.22, frequencySD = 0.07, \
							   frequencyResolutionStep = 0.01):
		'''
		Adds physiological noise to the data tensor.
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
		:param frequencyMean: The frequency mean that corresponds to the physiological noise
			Optional. Default is 0.22 (this is for the breathing rate noise.
			It could be the corresponding of any of the other physiological noises)
		:type frequencyMean: float (positive)
		:param frequencySD: The frequency standard deviation that corresponds to the physiological noise
			Optional. Default is 0.07 (this is for the breathing rate noise.
			It could be the corresponding of any of the other physiological noises)
		:type frequencySD: float (positive)
		:param frequencyResolutionStep: The step for generating evenly spaced values
			within the interval of frequencies of the noise to be simulated.
			Optional. Default is 0.01.
		:type frequencyResolutionStep: float (positive)
		:return: None
		:rtype: NoneType
		'''

		#Check parameters
		if type(channelsList) is not list:
			msg = self.getClassName() + ':addPhysiologicalNoise: Unexpected parameter type for parameter ''channelList''.'
			raise ValueError(msg)
		for elem in channelsList:
			if type(elem) is not int:
				msg = self.getClassName() + ':addPhysiologicalNoise: Unexpected parameter value for parameter ''channelList''.'
				raise ValueError(msg)
			if elem < 0 or elem >= self.nChannels:  # Ensure the nChannels exist
				msg = self.getClassName() + ':addPhysiologicalNoise: Unexpected parameter value for parameter ''channelList''.'
				raise ValueError(msg)

		if type(initSample) is not int:
			msg = self.getClassName() + ':addPhysiologicalNoise: Unexpected parameter type for parameter ''initSample''.'
			raise ValueError(msg)
		if initSample < 0 or initSample >= self.nSamples:  # Ensure the nSamples exist
			msg = self.getClassName() + ':addPhysiologicalNoise: Unexpected parameter value for parameter ''initSample''.'
			raise ValueError(msg)

		if type(endSample) is not int:
			msg = self.getClassName() + ':addPhysiologicalNoise: Unexpected parameter type for parameter ''endSample''.'
			raise ValueError(msg)
		if endSample < -1 or endSample >= self.nSamples:  # Ensure the nSamples exist
			msg = self.getClassName() + ':addPhysiologicalNoise: Unexpected parameter value for parameter ''endSample''.'
			raise ValueError(msg)
		if endSample == -1:  # If -1, substitute by the maximum last sample
			endSample = self.nSamples - 1
		if endSample <= initSample:  # Ensure the endSample is posterior to the initSample
			msg = self.getClassName() + ':addPhysiologicalNoise: Unexpected parameter value for parameter ''endSample''.'
			raise ValueError(msg)

		if type(frequencyMean) is not float:
			msg = self.getClassName() + ':addPhysiologicalNoise: Unexpected parameter type for parameter ''frequencyMean''.'
			raise ValueError(msg)
		if frequencyMean <= 0:
			msg = self.getClassName() + ':addPhysiologicalNoise: Unexpected parameter value for parameter ''frequencyMean''.'
			raise ValueError(msg)

		if type(frequencySD) is not float:
			msg = self.getClassName() + ':addPhysiologicalNoise: Unexpected parameter type for parameter ''frequencySD''.'
			raise ValueError(msg)
		if frequencySD <= 0:
			msg = self.getClassName() + ':addPhysiologicalNoise: Unexpected parameter value for parameter ''frequencySD''.'
			raise ValueError(msg)

		if type(frequencyResolutionStep) is not float:
			msg = self.getClassName() + ':addPhysiologicalNoise: Unexpected parameter type for parameter ''frequencyResolutionStep''.'
			raise ValueError(msg)
		if frequencyResolutionStep <= 0:
			msg = self.getClassName() + ':addPhysiologicalNoise: Unexpected parameter value for parameter ''frequencyResolutionStep''.'
			raise ValueError(msg)

		channelsList = list(set(channelsList))  # Unique and sort elements
		nChannels = len(channelsList)
		nSamples = endSample - initSample

		tmpData = np.zeros((nSamples, nChannels, 2)) #The temporal data tensor for saving the generated noise

		timestamps = np.arange(0, nSamples/self.samplingRate, \
								  1/self.samplingRate, dtype = float)
		timestamps = timestamps.reshape(-1, 1) #Reshape to column vector
		timestamps = np.tile(timestamps,nChannels)

		frequencySet = np.arange(frequencyMean-2*frequencySD, \
								 frequencyMean+2*frequencySD+frequencyResolutionStep, \
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
			tmpData[:,:,self.HBO2] = tmpData[:,:,self.HBO2] + tmpSin
			tmpData[:,:,self.HHB]  = tmpData[:,:,self.HHB]  + (-1/3)*tmpSin

		#plt.plot(tmpSin[0:nSamples,0], color='blue')
		#plt.show()

		#TODO: al tener la señal final se debe estandarizar z_score  eliminar la media y dividir por la desv. stand

		self.__data[0:nSamples,channelsList,:] = \
				self.__data[0:nSamples,channelsList,:] + tmpData

		return
	#end addPhysiologicalNoise(self, channelsList=list(), initSample=0, ... , frequencyResolutionStep = 0.01)


	def addHeartRateNoise(self, channelsList=list(), initSample=0, endSample=-1, \
						  frequencyResolutionStep = 0.01):
		'''
		Adds noise of heart rate to the data tensor.
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
		#No need to type check channelsList, initSample, endSample, and frequencyResolutionStep as
		#these are passed to method addPhysiologicalNoise.

		self.addPhysiologicalNoise(channelsList, initSample, endSample, \
								  frequencyMean=1.08, frequencySD=0.16, \
								  frequencyResolutionStep=0.01)  # From paper (Elwell et al., 1999)

		return
	#end addHeartRateNoise(self, channelsList=list(), initSample=0, ... , frequencyResolutionStep = 0.01)


	def addBreathingRateNoise(self, channelsList=list(), initSample=0, endSample=-1, \
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
		#No need to type check channelsList, initSample, endSample, and frequencyResolutionStep as
		#these are passed to method addPhysiologicalNoise.

		self.addPhysiologicalNoise(channelsList, initSample, endSample, \
								  frequencyMean=0.22, frequencySD=0.07, \
								  frequencyResolutionStep=0.01)  # From paper (Elwell et al., 1999)

		return
	#end addBreathingRateNoise(self, channelsList=list(), initSample=0, ... , frequencyResolutionStep = 0.01)


	def addVasomotionNoise(self, channelsList=list(), initSample=0, endSample=-1, \
						   frequencyResolutionStep = 0.01):
		'''
		Adds noise of vasomotion to the data tensor.
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
		#No need to type check channelsList, initSample, endSample, and frequencyResolutionStep as
		#these are passed to method addPhysiologicalNoise.

		self.addPhysiologicalNoise(channelsList, initSample, endSample, \
								  frequencyMean=0.082, frequencySD=0.016, \
								  frequencyResolutionStep=0.01)  # From paper (Elwell et al., 1999)

		return
	#end addVasomotionNoise(self, channelsList=list(), initSample=0, ... , frequencyResolutionStep = 0.01)


	def execute(self):
		'''
		Generates the synthetic fNIRS data from the properties
		information.
		This method calls :meth:`generateStimulusResult` for generating
		the new synthetic data.
		:return: A 3D data tensor
		:rtype: np.ndarray
		'''

		channelsList = list(range(0, self.nChannels))

		enableHbO2Channels = [1, 0, 1, 0] # every other channel enabled to simulate  Oxy vs Deoxy Channels
		#enableHbO2Channels[1] = 0   # channel 2 is disabled for the HbO2 signal
		#enableHbO2Channels[2] = 0  # channel 3 is disabled for the HbO2 signal

		#It is possible to express 'enableHbO2Channels' as a list (the program transforms it into an array later)
		#enableHbO2Channels = [1, 0, 1, 1] # this is for an example of 4 channels

		enableHHbChannels = [0, 1, 0, 1] # every other channel enabled to simulate  Oxy vs Deoxy Channels
		#enableHHbChannels[1] = 0   # channel 2 is disabled for the HHb signal
		#enableHHbChannels[2] = 0  # channel 3 is disabled for the HHb signal

		#It is possible to express 'enableHHbChannels' as a list (the program transforms it into an array later)
		#enableHHbChannels = [1, 0, 1, 1] # this is for an example of 4 channels

		#An alternative to provide the information for creating the boxCar is by the indication of the onsets and durations.
		#For this approach, the information is provided as a list of tuples (onset, duration)
		boxCarList_OnsetDurations = [(35, 10), (105, 15), (175, 20), (240, 25)]
		#boxCarList_OnsetDurations = [(35, 10), (105, 15), (175, 20)]

		#The boxCarList_OnsetDurations is used to generate boxCarList, which is a list of tuples (onset, end)
		#boxCarList is the boxCarList format expected for the methods of class fNIRSSignalGenerator
		boxCarList = list()
		for elem in boxCarList_OnsetDurations:
			boxCarList.append((elem[0], elem[0]+elem[1]))

		#print(boxCarList)

		#boxCarList = [(175, 195)]
		#boxCarList = [(35, 45), (105, 120), (240, 265)]
		#boxCarList = [(35, 45), (105, 120), (175, 195), (240, 265)]

		boxCarListSet = list(set(boxCarList))  # Unique and sort elements
		nBlocks = len(boxCarListSet)

		enableHbO2Blocks = np.ones(nBlocks, dtype=int) # all the blocks of the boxCar are enabled for the HbO2 signal
		#enableHbO2Blocks[1] = 0   # block 2 is disabled for the HbO2 signal in all the channels
		#enableHbO2Blocks[2] = 0  # block 3 is disabled for the HbO2 signal in all the channels

		#It is possible to express 'enableHbO2Blocks' as a list (the program transforms it into an array later)
		#enableHbO2Blocks = [1, 0, 1, 1] # this is for an example of 4 blocks

		enableHHbBlocks = np.ones(nBlocks, dtype=int) # all the blocks of the boxCar are enabled for the HHb signal
		#enableHHbBlocks[1] = 0   # block 2 is disabled for the HHb signal in all the channels
		#enableHHbBlocks[2] = 0  # block 3 is disabled for the HHb signal in all the channels

		#It is possible to express 'enableHHbBlocks' as a list (the program transforms it into an array later)
		#enableHHbBlocks = [1, 0, 1, 1] # this is for an example of 4 blocks

		self.addStimulusResult(channelsList, boxCarList,
							   initSample=0, endSample=-1,
							   tau_p=6, tau_d=10,
							   amplitudeScalingFactor=6,
							   enableHbO2Channels=enableHbO2Channels,
							   enableHHbChannels=enableHHbChannels,
							   enableHbO2Blocks=enableHbO2Blocks,
							   enableHHbBlocks=enableHHbBlocks)

		#plotSyntheticfNIRS(self.data, title='Synthetic fNIRS', enableHbO2Channels=enableHbO2Channels, enableHHbChannels=enableHHbChannels)

		#self.addBreathingRateNoise(channelsList, initSample=0, endSample=-1, \
								   #frequencyResolutionStep = 0.01)

		#plotSyntheticfNIRS(self.data, title='Synthetic fNIRS + Breathing rate noise', \
						   #enableHbO2Channels=enableHbO2Channels, enableHHbChannels=enableHHbChannels)

		#self.addHeartRateNoise(channelsList, initSample=0, endSample=-1, \
								   #frequencyResolutionStep = 0.01)

		#plotSyntheticfNIRS(self.data, title='Synthetic fNIRS + Noises: Breathing rate and Heart rate', \
						   #enableHbO2Channels=enableHbO2Channels, enableHHbChannels=enableHHbChannels)

		#self.addVasomotionNoise(channelsList, initSample=0, endSample=-1, \
								   #frequencyResolutionStep = 0.01)

		#plotSyntheticfNIRS(self.data, title='Synthetic fNIRS + Noises: Breathing rate, Heart rate, and Vasomotion', \
						   #enableHbO2Channels=enableHbO2Channels, enableHHbChannels=enableHHbChannels)

		self.addGaussianNoise(channelsList, initSample=0, endSample=-1)

		plotSyntheticfNIRS(self.data, title='Synthetic fNIRS + Gaussian Noise', \
						   enableHbO2Channels=enableHbO2Channels, enableHHbChannels=enableHHbChannels)

		return copy.deepcopy(self.data)
	#end execute(self)

#class fNIRSSignalGenerator


def plotSyntheticfNIRS(tensor, title='', enableHbO2Channels=np.ones(1, dtype=int), enableHHbChannels=np.ones(1, dtype=int)):
	'''
	Quick rendering of the synthetic fNIRS data tensor.
	'''

	nChannels = tensor.shape[1]
	for iCh in range(0,nChannels):
		if enableHbO2Channels[iCh]:
			plt.plot(tensor[:,iCh,0]+20*iCh, color='red')
		if enableHHbChannels[iCh]:
			plt.plot(tensor[:,iCh,1]+20*iCh, color='blue')
	plt.title(title)
	plt.xlabel('Time [samples]')
	plt.ylabel('Channels [A.U.]')
	plt.show()

	return
#end plotSyntheticfNIRS(tensor)


def main():
	# Specifying the channel location map for the EEG signal
	newId = 3
	newDescription = 'First New Config'
	newNChannels = 4
	newNOptodes = 4
	newChLocations = np.array([[1, 2, 0], [0, 1, 0], [2, 1, 0], [1, 0, 0]])
	newOptodesLocations = np.array([[0, 2, 0], [2, 2, 0], [0, 0, 0], [2, 0, 0]])
	newOptodesTypes = np.array([1, 2, 1, 2])  # Remember {0: Unknown, 1: Emission or source, 2: Detector}
	newReferencePoints = dict({'Nz': np.array([0, -18.5, 0]), 'Iz': np.array([0, 18.5, 0]),
							   'LPA': np.array([17.5, 0, 0]), 'RPA': np.array([-17.5, 0, 0]),
							   'Cz': np.array([0, 0, 0])})
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
							  optodesTypes = newOptodesTypes, referencePoints = newReferencePoints,
							  surfacePositioningSystem = newSurfacePositioningSystem,
							  chSurfacePositions = newChSurfacePositions,
							  optodesSurfacePositions = newOptodesSurfacePositions,
							  chOptodeArrays = newChOptodeArrays, optodesOptodeArrays = newOptodesOptodeArrays,
							  pairings = newPairings, optodeArrays = newOptodeArrays)

	# testing that constants can not receive other value
	#print("Valor de HBO2", sg.HBO2) # The value for HBO2 constant is 0
	#sg.HBO2 = 2
	#print("Nuevo Valor de HBO2", sg.HBO2)

	#print("Valor de HHB", sg.HHB) # The value for HHB constant is 1
	#sg.HHB = 3
	#print("Nuevo Valor de HHB", sg.HHB)

	sg.showAttributesValues()
	sg.execute()
	#print(sg.data)
#end main()


if __name__ == '__main__':
	main()
