# -*- coding: utf-8 -*-
#
# File: EEGSignalGenerator.py
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

from optodeArrayInfo import optodeArrayInfo

from channelLocationMap import channelLocationMap

# Class EEGSignalGenerator is a subclass of channelLocationMap
class EEGSignalGenerator(channelLocationMap):
    '''
	A basic class to generate synthetic EEG signals.

	'''

    #def __init__(self, nSamples=1, nChannels=1):   # __init__ used before the creation of the class channelLocationMap
    def __init__(self, nSamples=1, id = 1, description = 'ChannelLocationMap0001', nChannels = 1, nOptodes  = 1,
                 chLocations = np.array([[np.NaN, np.NaN, np.NaN]]), optodesLocations = np.array([[np.NaN, np.NaN, np.NaN]]),
                 optodesTypes = np.array([np.NaN]), referencePoints = dict(), surfacePositioningSystem = 'UI 10/20',
                 chSurfacePositions = tuple(('',)), optodesSurfacePositions = tuple(('',)), chOptodeArrays = np.array([np.NaN]),
                 optodesOptodeArrays = np.array([np.NaN]), pairings = np.array([[np.NaN, np.NaN]]),
                 optodeArrays = np.array([optodeArrayInfo()])):
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
        # Initialization of an object of the superclass channelLocationMap
        super().__init__(id = id, description = description, nChannels = nChannels, nOptodes  = nOptodes,
                         chLocations = chLocations, optodesLocations = optodesLocations,
                         optodesTypes = optodesTypes, referencePoints = referencePoints,
                         surfacePositioningSystem = surfacePositioningSystem,
                         chSurfacePositions = chSurfacePositions, optodesSurfacePositions = optodesSurfacePositions,
                         chOptodeArrays = chOptodeArrays,
                         optodesOptodeArrays = optodesOptodeArrays, pairings = pairings,
                         optodeArrays = optodeArrays)

        # Ensure all properties exist
        self.__data = np.zeros((0, 0, 0), dtype=float)
        self.__frequency_bands = dict()  # Frequency bands in [Hz]
        self.__frequency_bands['delta'] = [0.5, 4]
        self.__frequency_bands['theta'] = [4, 8]
        self.__frequency_bands['alpha'] = [8, 13]
        self.__frequency_bands['beta'] = [13, 30]
        self.__frequency_bands['gamma'] = [30, 40]
        self.__samplingRate = 512  # [Hz]

        # Check parameters
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

        # Initialize
        self.data = np.zeros((nSamples, nChannels, 1), dtype=float)

        return
    # end __init__(self, nSamples = 1, nChannels = 1)

    # Properties getters/setters
    #
    # Remember: Sphinx ignores docstrings on property setters so all
    # documentation for a property must be on the @property method

    @property
    def data(self):  # data getter
        '''
		The data tensor.

		The data tensor always have 3 dimensions, namely:

		* Temporal (temporal samples)
		* Spatial (channels)
		* Signals (for EEG this is fixed to 1; the voltages)

		:getter: Gets the data.
		:setter: Sets the data.
		:type: numpy.ndarray [nSamples x nChannels x 1]
		'''

        return copy.deepcopy(self.__data)
    # end data(self)

    @data.setter
    def data(self, newData):  # data setter

        # Check parameters
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
    # end data(self,newData)


    @property
    def frequency_bands(self):  # frequency_bands getter
        '''
		The EEG frequency bands.

		This is a read-only property

		:getter: Gets the EEG frequency bands
		:type: dict
		'''

        return copy.deepcopy(self.__frequency_bands)
    # end frequency_bands(self)


    @property
    def nChannels(self):  # nChannels getter
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
    # end nChannels(self)

    @nChannels.setter
    def nChannels(self, newNChannels):  # nChannels setter

        # Check parameters
        if type(newNChannels) is not int:
            msg = self.getClassName() + ':nChannels: Unexpected attribute type.'
            raise ValueError(msg)
        if newNChannels < 0:
            msg = self.getClassName() + ':nChannels: Unexpected attribute value. Number of channels must be greater or equal than 0.'
            raise ValueError(msg)

        if newNChannels > self.nChannels:
            # Add channels with zeros
            tmpNChannels = newNChannels - self.nChannels
            tmpData = np.zeros((self.nSamples, tmpNChannels, 1), dtype=float)
            self.data = np.concatenate((self.data, tmpData), axis=1)
        elif newNChannels < self.nChannels:
            msg = self.getClassName() + ':nChannels: New number of channels is smaller than current number of channels. Some data will be lost.'
            warnings.warn(msg, RuntimeWarning)
            self.data = copy.deepcopy(self.data[:, 0:newNChannels, :])

        return None
    # end nChannels(self,newNChannels)


    @property
    def nSamples(self):  # nSamples getter
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
    # end nSamples(self)

    @nSamples.setter
    def nSamples(self, newNSamples):  # nSamples setter

        # Check parameters
        if type(newNSamples) is not int:
            msg = self.getClassName() + ':nSamples: Unexpected attribute type.'
            raise ValueError(msg)
        if newNSamples < 0:
            msg = self.getClassName() + ':nSamples: Unexpected attribute value. Number of temporal samples must be greater or equal than 0.'
            raise ValueError(msg)

        if newNSamples > self.nSamples:
            # Add channels with zeros
            tmpNSamples = newNSamples - self.nSamples
            tmpData = np.zeros((tmpNSamples, self.nChannels, 1), dtype=float)
            self.data = np.concatenate((self.data, tmpData), axis=0)
        elif newNSamples < self.nSamples:
            msg = self.getClassName() + ':nSamples: New number of temporal samples is smaller than current number of temporal samples. Some data will be lost.'
            warnings.warn(msg, RuntimeWarning)
            self.data = copy.deepcopy(self.data[0:newNSamples, :, :])

        return None
    # end nSamples(self,newNSamples)


    @property
    def samplingRate(self):  # samplingrate getter
        '''
		Sampling rate at which the synthetic data will be generated.

		:getter: Gets the sampling rate.
		:setter: Sets the sampling rate.
		:type: float
		'''

        return self.__samplingRate
    # end samplingRate(self)

    @samplingRate.setter
    def samplingRate(self, newSamplingRate):  # samplingrate setter

        # Check parameters
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
    # end samplingRate(self,newSamplingRate)

    # Private methods

    # Protected methods

    # Public methods

    def getClassName(self):
        '''Gets the class name.

		:return: The class name
		:rtype: str
		'''

        return type(self).__name__
    # end getClassName(self)

    def addFrequencyBand(self, channelsList=list(), initSample=0, endSample=-1, \
                         freqBand='alpha', amplitudeScalingFactor=1, \
                         frequencyResolutionStep=0.1):
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
		:param frequencyResolutionStep: The step for generating evenly spaced values
			within the interval of frequencies of the band to be simulated.
			Optional. Default is 0.1.
		:type frequencyResolutionStep: float (positive)

		:return: None
		:rtype: NoneType

		.. todo::
			* Permit the channel list to be expressed by standard positioning
			  systems.

		.. seealso:: generateFrequencyBand

		'''

        # Check parameters
        if type(channelsList) is not list:
            msg = self.getClassName() + ':addFrequencyBand: Unexpected parameter type for parameter ''channelList''.'
            raise ValueError(msg)
        for elem in channelsList:
            if type(elem) is not int:
                msg = self.getClassName() + ':addFrequencyBand: Unexpected parameter value for parameter ''channelList''.'
                raise ValueError(msg)
            if elem < 0 or elem >= self.nChannels:  # Ensure the nChannels exist
                msg = self.getClassName() + ':addFrequencyBand: Unexpected parameter value for parameter ''channelList''.'
                raise ValueError(msg)
        if type(initSample) is not int:
            msg = self.getClassName() + ':addFrequencyBand: Unexpected parameter type for parameter ''initSample''.'
            raise ValueError(msg)
        if initSample < 0 or initSample >= self.nSamples:  # Ensure the nSamples exist
            msg = self.getClassName() + ':addFrequencyBand: Unexpected parameter value for parameter ''initSample''.'
            raise ValueError(msg)
        if type(endSample) is not int:
            msg = self.getClassName() + ':addFrequencyBand: Unexpected parameter type for parameter ''endSample''.'
            raise ValueError(msg)
        if endSample < -1 or endSample >= self.nSamples:  # Ensure the nSamples exist
            msg = self.getClassName() + ':addFrequencyBand: Unexpected parameter value for parameter ''endSample''.'
            raise ValueError(msg)
        if endSample == -1:  # If -1, substitute by the maximum last sample
            endSample = self.nSamples - 1
        if endSample <= initSample:  # Ensure the endSample is posterior to the initSample
            msg = self.getClassName() + ':addFrequencyBand: Unexpected parameter value for parameter ''endSample''.'
            raise ValueError(msg)
        # No need to type check freqBand, amplitudeScalingFactor and frequencyResolutionStep as these
        # are passed to method generateFrequencyBand.

        channelsList = list(set(channelsList))  # Unique and sort elements
        nChannels = len(channelsList)
        # nSamples  = endSample - initSample + 1
        nSamples = endSample - initSample
        tmpData = self.generateFrequencyBand(freqBand=freqBand, \
                                             nSamples=nSamples, \
                                             nChannels=nChannels, \
                                             amplitudeScalingFactor=amplitudeScalingFactor, \
                                             frequencyResolutionStep=frequencyResolutionStep)
        self.__data[initSample:endSample, channelsList, :] = \
            self.__data[initSample:endSample, channelsList, :] + tmpData

        return
    # end addFrequencyBand(self,channelsList = list(), initSample = 0, ... , frequencyResolutionStep = 0.1)

    def generateFrequencyBand(self, freqBand='alpha', nSamples=100, \
                              nChannels=1, amplitudeScalingFactor=1, \
                              frequencyResolutionStep=0.1):
        '''
		Generates synthetic data with energy in the chosen frequency band

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
		:param frequencyResolutionStep: The step for generating evenly spaced values
			within the interval of frequencies of the band to be simulated.
			Optional. Default is 0.1.
		:type frequencyResolutionStep: float (positive)

		:return: A data tensor.
		:rtype: numpy.ndarray

		.. #todo::
			#* Add parameter for frequencyResolutionStep

		.. #seealso:: generateFrequencyBand

		'''

        # Check parameters
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
        if amplitudeScalingFactor < 0:
            msg = self.getClassName() + ':generateFrequencyBand: Unexpected parameter value for parameter ''amplitudeScalingFactor''.'
            raise ValueError(msg)
        if type(frequencyResolutionStep) is not float:
            msg = self.getClassName() + ':generateFrequencyBand: Unexpected parameter type for parameter ''frequencyResolutionStep''.'
            raise ValueError(msg)
        if frequencyResolutionStep <= 0:
            msg = self.getClassName() + ':generateFrequencyBand: Unexpected parameter value for parameter ''frequencyResolutionStep''.'
            raise ValueError(msg)

        freqBand.sort()  # Ensure the min frequency is the first element.
        timestamps = np.arange(0, nSamples / self.samplingRate, \
                               1 / self.samplingRate, dtype=float)
        timestamps = timestamps.reshape(-1, 1)  # Reshape to column vector
        timestamps = np.tile(timestamps, nChannels)
        synthData = np.zeros((nSamples, nChannels, 1))  # The synthetic data tensor

        frequencySet = np.arange(freqBand[0], freqBand[1] + frequencyResolutionStep, \
                                 frequencyResolutionStep, dtype=float)
        for freq in frequencySet:
            # Amplitude. One random amplitude per channel
            A = amplitudeScalingFactor * np.random.rand(1, nChannels)
            A = np.tile(A, [nSamples, 1])
            # Phase [rad]. One random phase per channel
            theta = 2 * math.pi * np.random.rand(1, nChannels) - math.pi
            theta = np.tile(theta, [nSamples, 1])
            # Generate the fundamental signal
            tmpSin = A * np.sin(2 * math.pi * freq * timestamps + theta)
            # Elment-wise multiplication with the amplitude
            # NOTE: In python NumPy, a*b among ndarrays is the
            # element-wise product. For matrix multiplication, one
            # need to do np.matmul(a,b)
            synthData[:, :, 0] = synthData[:, :, 0] + tmpSin

        # Add the EEG background noise
        # synthData[:,:,0] = synthData[:,:,0] + (1/pow(freq, 2)) * tmpSin

        return synthData
    # end generateFrequencyBand(self,freqBand = 'alpha',nSamples = 100, ... , frequencyResolutionStep = 0.1)

    def addBackgroundNoise(self, channelsList: object = list(), initSample: object = 0, endSample: object = -1, \
						   allFreqBands: object = [0.5, 40], exp_alpha: object = -2, amplitudeScalingFactor: object = 1, \
						   frequencyResolutionStep: object = 0.1) -> object:
        '''
		Adds background noise to the data tensor.
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
		:param allFreqBands: All the frequency band to be simulated
		:type allFreqBands: interval (list or tuple of 2 elements)
			Provide min and max frequencies.
			Default is [0.5,40].
		:param exp_alpha: The exponent to be used when calculating the factor:
			pow(frequency, exp_alpha)
			that multiplies to the amplitude, for obtaining the background noise.
		:type exp_alpha: float -2 <= exp_alpha <= 2.
			Default is -2.
		:param amplitudeScalingFactor: A scaling factor for the amplitude.
			Amplitude of the individual fundamental frequencies is random,
			but normalized to [0 1] by default. This scalign factor permits
			scaling amplitudes to [0 amplitudeScalingFactor].
			Optional. Default is 1.
		:type amplitudeScalingFactor: float (positive)
		:param frequencyResolutionStep: The step for generating evenly spaced values
			within the interval of frequencies of the band to be simulated.
			Optional. Default is 0.1.
		:type frequencyResolutionStep: float (positive)

		:return: None
		:rtype: NoneType
		'''

        # Check parameters
        if type(channelsList) is not list:
            msg = self.getClassName() + ':addBackgroundNoise: Unexpected parameter type for parameter ''channelList''.'
            raise ValueError(msg)
        for elem in channelsList:
            if type(elem) is not int:
                msg = self.getClassName() + ':addBackgroundNoise: Unexpected parameter value for parameter ''channelList''.'
                raise ValueError(msg)
            if elem < 0 or elem >= self.nChannels:  # Ensure the nChannels exist
                msg = self.getClassName() + ':addBackgroundNoise: Unexpected parameter value for parameter ''channelList''.'
                raise ValueError(msg)
        if type(initSample) is not int:
            msg = self.getClassName() + ':addBackgroundNoise: Unexpected parameter type for parameter ''initSample''.'
            raise ValueError(msg)
        if initSample < 0 or initSample >= self.nSamples:  # Ensure the nSamples exist
            msg = self.getClassName() + ':addBackgroundNoise: Unexpected parameter value for parameter ''initSample''.'
            raise ValueError(msg)
        if type(endSample) is not int:
            msg = self.getClassName() + ':addBackgroundNoise: Unexpected parameter type for parameter ''endSample''.'
            raise ValueError(msg)
        if endSample < -1 or endSample >= self.nSamples:  # Ensure the nSamples exist
            msg = self.getClassName() + ':addBackgroundNoise: Unexpected parameter value for parameter ''endSample''.'
            raise ValueError(msg)
        if endSample == -1:  # If -1, substitute by the maximum last sample
            endSample = self.nSamples - 1
        if endSample <= initSample:  # Ensure the endSample is posterior to the initSample
            msg = self.getClassName() + ':addBackgroundNoise: Unexpected parameter value for parameter ''endSample''.'
            raise ValueError(msg)
        if type(allFreqBands) is tuple:
            allFreqBands = list(allFreqBands)
        if type(allFreqBands) is not list:
            msg = self.getClassName() + ':addBackgroundNoise: Unexpected parameter type for parameter ''allFreqBands''.'
            raise ValueError(msg)
        if len(allFreqBands) != 2:
            msg = self.getClassName() + ':addBackgroundNoise: Unexpected parameter value for parameter ''allFreqBands''.'
            raise ValueError(msg)
        if type(exp_alpha) is int:
            exp_alpha = float(exp_alpha)
        if type(exp_alpha) is not float:
            msg = self.getClassName() + ':addBackgroundNoise: Unexpected parameter type for parameter ''exp_alpha''.'
            raise ValueError(msg)
        if exp_alpha < -2 or exp_alpha > 2:
            msg = self.getClassName() + ':addBackgroundNoise: Unexpected parameter value for parameter ''exp_alpha''.'
            raise ValueError(msg)
        if type(amplitudeScalingFactor) is int:
            amplitudeScalingFactor = float(amplitudeScalingFactor)
        if type(amplitudeScalingFactor) is not float:
            msg = self.getClassName() + ':addBackgroundNoise: Unexpected parameter type for parameter ''amplitudeScalingFactor''.'
            raise ValueError(msg)
        if amplitudeScalingFactor < 0:
            msg = self.getClassName() + ':addBackgroundNoise: Unexpected parameter value for parameter ''amplitudeScalingFactor''.'
            raise ValueError(msg)
        if type(frequencyResolutionStep) is not float:
            msg = self.getClassName() + ':addBackgroundNoise: Unexpected parameter type for parameter ''frequencyResolutionStep''.'
            raise ValueError(msg)
        if frequencyResolutionStep <= 0:
            msg = self.getClassName() + ':addBackgroundNoise: Unexpected parameter value for parameter ''frequencyResolutionStep''.'
            raise ValueError(msg)

        channelsList = list(set(channelsList))  # Unique and sort elements
        nChannels = len(channelsList)
        nSamples = endSample - initSample

        allFreqBands.sort()  # Ensure the min frequency is the first element.
        timestamps = np.arange(0, nSamples / self.samplingRate, \
                               1 / self.samplingRate, dtype=float)
        timestamps = timestamps.reshape(-1, 1)  # Reshape to column vector
        timestamps = np.tile(timestamps, nChannels)

        frequencySet = np.arange(allFreqBands[0], allFreqBands[1] + frequencyResolutionStep, \
                                 frequencyResolutionStep, dtype=float)
        for freq in frequencySet:
            # Amplitude. One random amplitude per channel
            A = amplitudeScalingFactor * np.random.rand(1, nChannels)
            A = np.tile(A, [nSamples, 1])
            # Phase [rad]. One random phase per channel
            theta = 2 * math.pi * np.random.rand(1, nChannels) - math.pi
            theta = np.tile(theta, [nSamples, 1])
            # Generate the fundamental signal
            tmpSin = A * np.sin(2 * math.pi * freq * timestamps + theta)
            # Elment-wise multiplication with the amplitude
            # NOTE: In python NumPy, a*b among ndarrays is the
            # element-wise product. For matrix multiplication, one
            # need to do np.matmul(a,b)

            # Add the EEG background noise
            self.__data[initSample:endSample, channelsList, 0] = \
                self.__data[initSample:endSample, channelsList, 0] + pow(freq, exp_alpha) * tmpSin

        return
    # end addBackgroundNoise(self, channelsList=list(), initSample=0, endSample=-1, ..., frequencyResolutionStep = 0.1):

    def execute(self):
        '''
		Generates the synthetic EEG data from the properties
		information.

		:return: A 3D data tensor
		:rtype: numpy.ndarray
		'''

        self.addBackgroundNoise(channelsList=list(range(0, self.nChannels)), \
								initSample=0, endSample=-1, \
								allFreqBands=[0.5,40], exp_alpha=-2, amplitudeScalingFactor=1, \
								frequencyResolutionStep=0.1)

        #for freqBandValue in self.frequency_bands:
        #    self.addFrequencyBand(channelsList=list(range(0, self.nChannels)), \
        #                          initSample=0, endSample=-1, \
        #                          freqBand='alpha')

        self.addFrequencyBand(channelsList=list(range(0, self.nChannels)), \
        				  initSample=0, endSample=-1, \
        				  freqBand='alpha')
        self.addFrequencyBand(channelsList=[0, 1, 2, 3], \
        				  initSample=round(self.nSamples / 2), endSample=-1, \
        				  freqBand='theta')
        self.addFrequencyBand(channelsList=[0, 1, 2, 3], \
        				  initSample=round(self.nSamples / 4), \
        				  endSample=round(3 * self.nSamples / 4), \
        				  freqBand='delta')
        self.addFrequencyBand(channelsList=[0, 1, 2, 3], \
        				  initSample=158, \
        				  endSample=846, \
        				  freqBand='gamma', \
        				  amplitudeScalingFactor=2.2)

        return copy.deepcopy(self.data)
    # end execute(self)

#class EEGSignalGenerator


def plotSyntheticEEG(tensor):
    '''
	Quick rendering of the synthetic EEG data tensor.
	'''

    nChannels = tensor.shape[1]
    for iCh in range(0, nChannels):
        plt.plot(tensor[:, iCh, 0] + 20 * iCh)
    plt.xlabel('Time [samples]')
    plt.ylabel('Channels [A.U.]')
    plt.show()

    return
# end plotSyntheticEEG(tensor)


def main():
    # Specifying the channel location map for the EEG signal
    newId = 2
    newDescription = 'ChannelLocationMap0002'
    newNChannels = 4
    newNOptodes = 4
    newChLocations = np.array([[1, 2, 0], [0, 1, 0], [2, 1, 0], [1, 0, 0]])
    newOptodesLocations = np.array([[0, 2, 0], [2, 2, 0], [0, 0, 0], [2, 0, 0]])
    newOptodesTypes = np.array([1, 2, 2, 1])  # Remember {0: Unknown, 1: Emission or source, 2: Detector}
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

    # A channelLocationMap for the EEG signals
    sg = EEGSignalGenerator(nSamples=3000, nChannels=newNChannels, chLocations=newChLocations, referencePoints=newReferencePoints,
                            chSurfacePositions=newChSurfacePositions, chOptodeArrays=newChOptodeArrays)
    sg.showAttributesValues()
    sg.execute()
    # print(sg.data)
    plotSyntheticEEG(sg.data)
# end main()


if __name__ == '__main__':
    main()
