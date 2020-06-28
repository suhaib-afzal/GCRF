# -*- coding: utf-8 -*-
#
#File: EEGSignalGenerator.py
#
'''
Created on Fri May 29 19:34:21 2020

Module ***EEGSignalGenerator***

This module implements the class :class:`EEGSignalGenerator <EEGSignalGenerator>`.

:Log:

+-------------+--------+------------------------------------------------------+
| Date        | Author | Description                                          |
+=============+========+======================================================+
| 29-May-2020 | FOE    | - Class :class:`EEGSignalGenerator` created but      |
|             |        |   unfinished.                                        |
+-------------+--------+------------------------------------------------------+
| 27-Jun-2020 | JJR    | - Class :class:`EEGSignalGenerator`                  |
|             |        |   adding some changes.                               |
+-------------+--------+------------------------------------------------------+


.. sectionauthor:: Felipe Orihuela-Espina <f.orihuela-espina@inaoep.mx>
.. codeauthor:: Felipe Orihuela-Espina <f.orihuela-espina@inaoep.mx>

'''

import copy
import warnings

import numpy as np
import random
import math

import matplotlib.pyplot as plt



class EEGSignalGenerator:
	'''
	A basic class to generate synthetic EEG signals.
	
	'''
	def __init__(self, nSamples = 1, nChannels = 1):
		'''
		Class constructor.
		
		
		:Properties:
		
		data: The EEG data tensor.
		frequency_bands: The EEG frequency bands.
		
		
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
		self.__frequency_bands = dict() #Frequency bands in [Hz]
		self.__frequency_bands['delta'] = [0.5,4]
		self.__frequency_bands['theta'] = [4,8]
		self.__frequency_bands['alpha'] = [8,13]
		self.__frequency_bands['beta']  = [13,30]
		self.__frequency_bands['gamma'] = [30,40]
		self.__samplingRate = 512 #[Hz]
		
		
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
		self.data = np.zeros((nSamples,nChannels,1),dtype=float)
		
		return


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
		* Signals (for EEG this is fixed to 1; the voltages)
		
		:getter: Gets the data.
		:setter: Sets the data.
		:type: int
		'''
		return copy.deepcopy(self.__data)

	@data.setter
	def data(self,newData): #data setter
		if type(newData) is not np.ndarray:
			msg = self.getClassName() + ':data: Unexpected attribute type.'
			raise ValueError(msg)
		if newData.ndim != 3:
			msg = self.getClassName() + ':data: Unexpected attribute value. ' \
					+ 'Data tensor must be 3D <temporal, spatial, signal>'
			raise ValueError(msg)
		if newData.shape[2] != 1:
			msg = self.getClassName() + ':data: Unexpected attribute value. ' \
					+ 'Number of signals in EEG must be 1.'
			raise ValueError(msg)
			
		self.__data = copy.deepcopy(newData)
		return None


	@property
	def frequency_bands(self): #frequency_bands getter
		'''
		The eeg frequency bands.
		
		This is a read-only property
		
		:getter: Gets the eeg frequency bands
		:type: dict
		'''
		return copy.deepcopy(self.__frequency_bands)


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

	@nChannels.setter
	def nChannels(self,newNChannels): #nChannels setter
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

	@nSamples.setter
	def nSamples(self,newNSamples): #nSamples setter
		if type(newNSamples) is not int:
			msg = self.getClassName() + ':nSamples: Unexpected attribute type.'
			raise ValueError(msg)
		if newNSamples < 0:
			msg = self.getClassName() + ':nChannels: Unexpected attribute value. Number of temporal samples must be greater or equal than 0.'
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


	@property
	def samplingRate(self): #samplingrate getter
		'''
		Sampling rate at which the synthetic data will be generated.
		
		:getter: Gets the sampling rate.
		:setter: Sets the sampling rate.
		:type: float
		'''
		return self.__samplingRate

	@samplingRate.setter
	def samplingRate(self,newSamplingRate): #samplingrate setter
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


	#Private methods
	
	
	#Protected methods
	
	#Public methods
	def getClassName(self):
		'''Gets the class name.
		
		:return: The class name
		:rtype: str
		'''
		return type(self).__name__

	def addFrequencyBand(self,channelsList = list(), initSample = 0, endSample = -1, \
						  freqBand = 'alpha', amplitudeScalingFactor = 1):
		'''
		Adds a frequency band to the data tensor.
		
		This method calls :meth:`generateFrequencyBand` for generating
		the new synthetic data. Here, such newly generated data tensor
		is added to the class :attr:`data`.
		
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
		:param freqBand: The frequency band to be simulated
		:type freqBand: str or interval (list or tuple of 2 elements)
			If str, then the classical frequency bands can be indicated, e.g.
			''alpha''. If interval, then provide a list of min and max
			frequencies. Default is 'alpha'.
		:param amplitudeScalingFactor: A scaling factor for the amplitude.
			Amplitude of the individual fundamental frequencies is random,
			but normalized to [0 1] by default. This scalign factor permits
			scaling amplitudes to [0 amplitudeScalingFactor].
			Optional. Default is 1.
		:type amplitudeScalingFactor: float (positive)
		
		:return: None
		:rtype: NoneType
		
		.. todo::
			* Permit the channel list to be expressed by standard positioning
			  systems.

		.. seealso:: generateFrequencyBand
		
		'''
		if type(channelsList) is not list:
			msg = self.getClassName() + ':addFrequencyBand: Unexpected parameter type for parameter ''channelList''.'
			raise ValueError(msg)
		for elem in channelsList:
			if type(elem) is not int:
				msg = self.getClassName() + ':addFrequencyBand: Unexpected parameter value for parameter ''channelList''.'
				raise ValueError(msg)
			if elem < 0 or elem >= self.nChannels: #Ensure the channel exist
				msg = self.getClassName() + ':addFrequencyBand: Unexpected parameter value for parameter ''channelList''.'
				raise ValueError(msg)
		if type(initSample) is not int:
			msg = self.getClassName() + ':addFrequencyBand: Unexpected parameter type for parameter ''initSample''.'
			raise ValueError(msg)
		if initSample < 0 or initSample >= self.nSamples: #Ensure the initSample exist
			msg = self.getClassName() + ':addFrequencyBand: Unexpected parameter value for parameter ''initSample''.'
			raise ValueError(msg)
		if type(endSample) is not int:
			msg = self.getClassName() + ':addFrequencyBand: Unexpected parameter type for parameter ''endSample''.'
			raise ValueError(msg)
		if endSample < -1 or endSample >= self.nSamples: #Ensure the endSample exist
			msg = self.getClassName() + ':addFrequencyBand: Unexpected parameter value for parameter ''endSample''.'
			raise ValueError(msg)
		if endSample == -1: #If -1, substitute by the maximumlast sample
			endSample = self.nSamples-1
		if endSample <= initSample: #Ensure the endSample is posterior to the initSample
			msg = self.getClassName() + ':addFrequencyBand: Unexpected parameter value for parameter ''endSample''.'
			raise ValueError(msg)
		#No need to type check freqBand and amplitudeScalingFactor as these
		#are passed to method generateFrequencyBand.

		channelsList = list(set(channelsList)) #Unique and sort elements
		nChannels = len(channelsList)
		nSamples  = endSample - initSample
		tmpData = self.generateFrequencyBand(freqBand = freqBand, \
										nSamples = nSamples, \
										nChannels=nChannels, \
										amplitudeScalingFactor = amplitudeScalingFactor)
		self.__data[initSample:endSample,channelsList,:] = \
				self.__data[initSample:endSample,channelsList,:] + tmpData
		
		return




	def execute(self):
		'''
		Generates the synthetic data from the properties
		information.
		
		:return: A 3D data tensor
		:rtype: np.ndarray
		'''
		
		self.addFrequencyBand(channelsList= list(range(0,self.nChannels)), \
								initSample = 0, endSample = -1, \
								freqBand = 'alpha')
		self.addFrequencyBand(channelsList= [1,4,5], \
								initSample = round(self.nSamples/2), endSample = -1, \
								freqBand = 'theta')
		self.addFrequencyBand(channelsList= [2,3,4], \
								initSample = round(self.nSamples/4), \
								endSample  = round(3*self.nSamples/4), \
								freqBand = 'delta')
		self.addFrequencyBand(channelsList= [1,5], \
								initSample = 158, \
								endSample  = 846, \
								freqBand = 'gamma',\
								amplitudeScalingFactor = 2.2)
		
		return copy.deepcopy(self.data)
	
	
	def generateFrequencyBand(self,freqBand = 'alpha',nSamples = 100, \
							   nChannels=1, amplitudeScalingFactor = 1):
		'''
		Generate synthetic data with energy in the chosen frequency band

		:Parameters:
		
		:param freqBand: The frequency band to be simulated
		:type freqBand: str or interval (list or tuple of 2 elements)
			If str, then the classical frequency bands can be indicated, e.g.
			''alpha''. If interval, then provide a list of min and max
			frequencies. Default is 'alpha'.
		:param nSamples: Number of temporal samples.
			Optional. Default is 1.
		:type nSamples: int (positive)
		:param nChannels: Number of channels.
			Optional. Default is 1.
		:type nChannels: int (positive)
		:param amplitudeScalingFactor: A scaling factor for the amplitude.
			Amplitude of the individual fundamental frequencies is random,
			but normalized to [0 1] by default. This scalign factor permits
			scaling amplitudes to [0 amplitudeScalingFactor].
			Optional. Default is 1.
		:type amplitudeScalingFactor: float (positive)
		
		:return: A data tensor.
		:rtype: np.ndarray
		
		.. todo::
			* Add parameter for frequencyResolutionStep
		
		.. seealso:: generateFrequencyBand

		'''
		#Check parameters
		if type(freqBand) is str:
			if not freqBand in self.frequency_bands.keys():
				msg = self.getClassName() + ':generateFrequencyBand: Unexpected parameter value for parameter ''freqBand''.'
				raise ValueError(msg)
			freqBand = self.frequency_bands[freqBand]
		if type(freqBand) is tuple:
			freqBand = list(freqBand)
		if type(freqBand) is not list:
				msg = self.getClassName() + ':generateFrequencyBand: Unexpected parameter type for parameter ''freqBand''.'
				raise ValueError(msg)
		if len(freqBand) != 2:
			msg = self.getClassName() + ':generateFrequencyBand: Unexpected parameter value for parameter ''freqBand''.'
			raise ValueError(msg)
		if type(nSamples) is not int:
			msg = self.getClassName() + ':generateFrequencyBand: Unexpected parameter type for parameter ''nSamples''.'
			raise ValueError(msg)
		if nSamples <= 0:
			msg = self.getClassName() + ':generateFrequencyBand: Unexpected parameter value for parameter ''nSamples''.'
			raise ValueError(msg)
		if type(nChannels) is not int:
			msg = self.getClassName() + ':generateFrequencyBand: Unexpected parameter type for parameter ''nChannels''.'
			raise ValueError(msg)
		if nChannels <= 0:
			msg = self.getClassName() + ':generateFrequencyBand: Unexpected parameter value for parameter ''nChannels''.'
			raise ValueError(msg)
		if type(amplitudeScalingFactor) is int:
			amplitudeScalingFactor = float(amplitudeScalingFactor)
		if type(amplitudeScalingFactor) is not float:
			msg = self.getClassName() + ':generateFrequencyBand: Unexpected parameter type for parameter ''amplitudeScalingFactor''.'
			raise ValueError(msg)
		if amplitudeScalingFactor <= 0:
			msg = self.getClassName() + ':generateFrequencyBand: Unexpected parameter value for parameter ''amplitudeScalingFactor''.'
			raise ValueError(msg)
		
		
		frequencyResolutionStep = 0.1 #[Hz] - Possibly make a parameter
		freqBand.sort() #Ensure the min frequency is the first element.
		timestamps = np.arange(0, nSamples/self.samplingRate, \
								  1/self.samplingRate, dtype = float)
		timestamps = timestamps.reshape(-1, 1) #Reshape to column vector
		timestamps = np.tile(timestamps,nChannels)
		synthData = np.zeros((nSamples, nChannels, 1)) #The synthetic data tensor
		
		frequencySet = np.arange(freqBand[0], freqBand[1]+frequencyResolutionStep,\
						    frequencyResolutionStep, dtype = float)
		for freq in frequencySet:
			#Amplitude. One random amplitude per channel
			A = amplitudeScalingFactor*np.random.rand(1,nChannels)
			A = np.tile(A,[nSamples,1])
			#Phase [rad]. One random phase per channel
			theta = 2* math.pi * np.random.rand(1,nChannels) - math.pi
			theta = np.tile(theta,[nSamples,1])
			#Generate the fundamental signal
			tmpSin = A * np.sin(2*math.pi*freq*timestamps+theta)
				#Elment-wise multiplication with the amplitude
				#NOTE: In python NumPy, a*b among ndarrays is the
				#element-wise product. For matrix multiplication, one
				#need to do np.matmul(a,b)
			synthData[:,:,0] = synthData[:,:,0] + tmpSin
			
		return synthData



def plotSyntheticEEG(tensor):
	'''
	Quick rendering of the synthetic EEG data tensor.
	'''
	nChannels = tensor.shape[1]
	for iCh in range(0,nChannels):
		plt.plot(tensor[:,iCh,0]+10*iCh)
	plt.xlabel('Time [samples]')
	plt.ylabel('Channels [A.U.]')
	plt.show()
	return
	

def main():
	sg = EEGSignalGenerator(nSamples = 3000, nChannels = 6)
	sg.execute()
	#print(sg.data)
	plotSyntheticEEG(sg.data)
	

if __name__ == '__main__':
	main()


