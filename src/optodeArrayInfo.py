# -*- coding: utf-8 -*-
#
#File: optodeArrayInfo.py
#
'''
Created on Mon Dec 11 21:36:00 2020

Module ***optodeArrayInfo***

This module implements the class :class:`optodeArrayInfo <optodeArrayInfo>`.

:Log:

+-------------+---------+------------------------------------------------------+
| Date        | Authors | Description                                          |
+=============+=========+======================================================+
| 11-Dic-2020 |   FOE   | - Class :class:`optodeArrayInfo` created but    |
|             |   JJR   |   unfinished.                                        |
+-------------+--------+------------------------------------------------------+


.. sectionauthor:: Felipe Orihuela-Espina <f.orihuela-espina@inaoep.mx> and Jesús Joel Rivas <jrivas@inaoep.mx>
.. codeauthor::    Felipe Orihuela-Espina <f.orihuela-espina@inaoep.mx> and Jesús Joel Rivas <jrivas@inaoep.mx>

'''

import copy
import warnings

import numpy as np

import matplotlib.pyplot as plt


class optodeArrayInfo:
	'''
	Description:
	A basic class to register the optode array information.

	A optodeArrayInfo captures the spatial positioning of channels and optodes
	for a neuroimage in Real world coordinates (e.g. <X,Y,Z>)

	The real world coordinates are cartesian coordinates.

	Optode Arrays:

	Near Infrared Spectroscopy optodes in neuroimage are commonly
	disposed in optode arrays coupling light emitters and detectors
	to form channels at specific places.

	The size and type of the optode array and hence the number
	of channels that the optode array can accommodate and their spatial
	disposition (topological arrangement) vary with every device. This
	information is encoded in this class, which hold
	the information about the topological arrangement of the channels
	for the optode array.

	Following, a few examples of the topological disposition of channels
	for some known optode arrays are illustrated. All the information
	regarding the optode array configuration is part of the attribute
	oaInfo (optode array Info).

	Example: HITACHI ETG-4000 3x3 optode array

	S - Light Source           S---1---D---2---S
	D - Light Detector         |       |       |
	1,..,12 - Channel          3       4       5
	                           |       |       |
	                           D---6---S---7---D
	                           |       |       |
	                           8       9      10
	                           |       |       |
	                           S--11---D--12---S

	Example: HITACHI ETG-4000 4x4 optode array

	S - Light Source           S---1---D---2---S---3---D
	D - Light Detector         |       |       |       |
	1,..,24 - Channel          4       5       6       7
	                           |       |       |       |
	                           D---8---S---9---D--10---S
	                           |       |       |       |
	                          11      12      13      14
	                           |       |       |       |
	                           S--15---D--16---S--17---D
	                           |       |       |       |
	                          18      19      20      21
	                           |       |       |       |
	                           D--22---S--23---D--24---S

	Example: HITACHI ETG-4000 3x5 optode array

	S - Light Source        S---1---D---2---S---3---D---4---S
	D - Light Detector      |       |       |       |       |
	1,..,24 - Channel       5       6       7       8       9
	                        |       |       |       |       |
	                        D--10---S--11---D--12---S--13---D
	                        |       |       |       |       |
	                        14      15      16      17      18
	                        |       |       |       |       |
	                        S--19---D--20---S--21---D--22---S

	Properties:

		.nChannels - Number of channels supported by the optode array.

		.nOptodes - Number of optodes supported by the optode array.

		.mode - A string describing the optode array.
				Valid modes depend on the neuroimage type. Each neuroimage
				subclass should check the validity of the modes.

		.typeOptodeArray - A string describing whether the optode array is for
				adults, infants or neonates.

		.chTopoArrangement - A numpy.ndarray of nChannels x 3, for expressing the
				topographical arrangement (3D coordinates) of the channels
				within the optode array.

		.optodesTopoArrangement - A numpy.ndarray of nOptodes x 3, for expressing the
				topographical arrangement (3D coordinates) of the optodes
				within the optode array.

				The above two subfields are 3D coordinates which
				locate the channels and optodes respectively internally
				to the optode array. The
				XY plane is the surface plane (i.e. over the scalp)
				with arbitrary rotation and axis origin, and the Z
				coordinate indicates the depth (with Z=0 being the scalp
				plane and positive values indicating deeper layers into
				the head. Note how these coordinates differ from those
				of the real world in the class attributes .chLocations
				and optodesLocations.

			    The coordinates in this property are assigned to the
				channels and optodes associated to this optode array in
				order from the lowest channel or optode number (i.e. 1)
				to the highest. A default arrangement positions
				the channels and optodes along a straight line
				over the X axis.

				The number of locations in this topographical arrangement
				may not match the number of associated channels or optodes
				respectively. When associating new channels or optodes,
				if the number of associated channels or optodes
				surpasses the number of defined topographical
				locations, these latter will be automatically be
				generated by default (set along a line over the X axis).
				When the number of associated channels or optodes
				drop below the
				number of defined topographical locations, the remaining
				topographical locations will simply be ignored. However,
				beware that they will not be removed, and will remain
				latent and will be used again if ever new channels or
				optodes are associated to the optode array.

	'''

	def __init__(self, nChannels = 1, nOptodes  = 1, mode = '', typeOptodeArray = '', \
				 chTopoArrangement = np.array([[np.NaN, np.NaN, np.NaN]]), \
				 optodesTopoArrangement = np.array([[np.NaN, np.NaN, np.NaN]])):
		'''
		Class constructor.
		
		
		:Properties:
		
		nChannels: Number of channels supported by the optode array.

		nOptodes: Number of optodes supported by the optode array.

		mode: A string describing the optode array.

		typeOptodeArray: A string describing whether the optode array is for adults, infants or neonates.

		chTopoArrangement: A numpy.ndarray of nChannels x 3, for expressing the
				topographical arrangement (3D coordinates) of the channels
				within the optode array.

		optodesTopoArrangement: A numpy.ndarray of nOptodes x 3, for expressing the
				topographical arrangement (3D coordinates) of the optodes
				within the optode array.


		:Parameters:
		
		:param nChannels: Number of channels supported by the optode array.
			Default is 1.
		:type nChannels: int (positive)

		:param nOptodes: Number of optodes supported by the optode array.
			Default is 1.
		:type nOptodes: int (positive)

		:param mode: A string describing the optode array.
			Optional. Default is ''.
		:type mode: str

		:param typeOptodeArray: A string describing whether the optode array is for adults, infants or neonates.
			Optional. Default is ''.
		:type typeOptodeArray: str

		:param chTopoArrangement: A numpy.ndarray of nChannels x 3, for expressing the
				topographical arrangement (3D coordinates) of the channels
				within the optode array.
			Optional. Default is np.array([[np.NaN, np.NaN, np.NaN]]).
		:type chTopoArrangement: np.ndarray [nChannels x 3]

		:param optodesTopoArrangement: A numpy.ndarray of nOptodes x 3, for expressing the
				topographical arrangement (3D coordinates) of the optodes
				within the optode array.
			Optional. Default is np.array([[np.NaN, np.NaN, np.NaN]]).
		:type optodesTopoArrangement: np.ndarray [nOptodes x 3]
		'''
		
		#Check parameters
		if type(nChannels) is not int:
			msg = self.getClassName() + ':__init__: Unexpected parameter type for parameter ''nChannels''.'
			raise ValueError(msg)
		if nChannels < 1:
			msg = self.getClassName() + ':__init__: Unexpected parameter value for parameter ''nChannels''.'
			raise ValueError(msg)

		if type(nOptodes) is not int:
			msg = self.getClassName() + ':__init__: Unexpected parameter type for parameter ''nOptodes''.'
			raise ValueError(msg)
		if nOptodes < 1:
			msg = self.getClassName() + ':__init__: Unexpected parameter value for parameter ''nOptodes''.'
			raise ValueError(msg)

		if type(mode) is not str:
			msg = self.getClassName() + ':__init__: Unexpected parameter type for parameter ''mode''.'
			raise ValueError(msg)

		if type(typeOptodeArray) is not str:
			msg = self.getClassName() + ':__init__: Unexpected parameter type for parameter ''typeOptodeArray''.'
			raise ValueError(msg)

		if type(chTopoArrangement) is not np.ndarray:
			msg = self.getClassName() + ':__init__: Unexpected parameter type for parameter ''chTopoArrangement''.'
			raise ValueError(msg)
		if chTopoArrangement.ndim != 2:
			msg = self.getClassName() + ':__init__: Unexpected attribute value for parameter ''chTopoArrangement''. ' \
				  + 'The topographical arrangement (3D coordinates) of the channels must be stored on 2D [nChannels x 3]'
			raise ValueError(msg)
		if chTopoArrangement.shape[0] != nChannels:
			msg = self.getClassName() + ':__init__: Unexpected attribute value for parameter ''chTopoArrangement''. ' \
				  + 'Number of rows must be nChannels.'
			raise ValueError(msg)
		if chTopoArrangement.shape[1] != 3:
			msg = self.getClassName() + ':__init__: Unexpected attribute value for parameter ''chTopoArrangement''. ' \
				  + 'Number of columns must be 3.'
			raise ValueError(msg)

		if type(optodesTopoArrangement) is not np.ndarray:
			msg = self.getClassName() + ':__init__: Unexpected parameter type for parameter ''optodesTopoArrangement''.'
			raise ValueError(msg)
		if optodesTopoArrangement.ndim != 2:
			msg = self.getClassName() + ':__init__: Unexpected attribute value for parameter ''optodesTopoArrangement''. ' \
				  + 'The topographical arrangement (3D coordinates) of the optodes must be stored on 2D [nOptodes x 3]'
			raise ValueError(msg)
		if optodesTopoArrangement.shape[0] != nOptodes:
			msg = self.getClassName() + ':__init__: Unexpected attribute value for parameter ''optodesTopoArrangement''. ' \
				  + 'Number of rows must be nOptodes.'
			raise ValueError(msg)
		if optodesTopoArrangement.shape[1] != 3:
			msg = self.getClassName() + ':__init__: Unexpected attribute value for parameter ''optodesTopoArrangement''. ' \
				  + 'Number of columns must be 3.'
			raise ValueError(msg)

		#Initialize
		self.__nChannels 	= nChannels		# Number of channels supported by the optode array.
		self.__nOptodes 	= nOptodes		# Number of optodes supported by the optode array.
		self.mode 			= mode 			# String describing the optode array.
		self.typeOptodeArray = typeOptodeArray	# String describing whether the optode array is for adults, infants or neonates
		self.chTopoArrangement 	 	= copy.deepcopy(chTopoArrangement)	# Topographical arrangement of the channels within the optode array
															# 	this is expressed through the 3D coordinates of each channel
															# 	within the optode array

		self.optodesTopoArrangement = copy.deepcopy(optodesTopoArrangement)	# Topographical arrangement of the optodes within the optode array
															# 	this is expressed through the 3D coordinates of each optode
															# 	within the optode array

		return
	#end __init__(self, nChannels = 1, nOptodes  = 1, ..., optodesTopoArrangement = np.array([[np.NaN, np.NaN, np.NaN]]))


	#Properties getters/setters
	#
	# Remember: Sphinx ignores docstrings on property setters so all
	#documentation for a property must be on the @property method

	@property
	def nChannels(self):  # nChannels getter
		'''
		Number of channels supported by the optode array.

		When setting the number of channels:

		* if the number of channels is smaller than
		the current number of channels, a warning is issued
		and the channels indexed highest (on the bottom) in the
			chTopoArrangement		array [nChannels x 3] of float
		will be removed.
		* if the number of channels is greater than
		the current number of channels, the new channels will
		be filled with np.NaN value in chTopoArrangement


		:getter: Gets the number of channels supported by the optode array.
		:setter: Sets the number of channels supported by the optode array.
		:type: int
		'''

		return self.__nChannels

	# end nChannels(self)

	@nChannels.setter
	def nChannels(self, newNChannels):  # nChannels setter

		# Check parameters
		if type(newNChannels) is not int:
			msg = self.getClassName() + ':nChannels: Unexpected parameter type.'
			raise ValueError(msg)
		if newNChannels < 1:
			msg = self.getClassName() + ':nChannels: Unexpected attribute value. Number of channels must be greater or equal than 1.'
			raise ValueError(msg)

		if newNChannels < self.__nChannels:
			msg = self.getClassName() + ':nChannels: New number of channels is smaller than current number of channels. Some data will be lost.'
			warnings.warn(msg, RuntimeWarning)

			# Modifying chTopoArrangement			array [nChannels x 3] of float
			self.chTopoArrangement = copy.deepcopy(self.chTopoArrangement[0:newNChannels, :])

		elif newNChannels > self.__nChannels:
			# Add channels with np.NaN value in chTopoArrangement
			tmpNChannels = newNChannels - self.__nChannels

			# Modifying chTopoArrangement
			#tmpChTopoArrangement   = np.zeros((tmpNChannels, 3), dtype=float)
			tmpChTopoArrangement = np.empty((tmpNChannels, 3))
			tmpChTopoArrangement[:] = np.NaN
			self.chTopoArrangement = np.concatenate((self.chTopoArrangement, tmpChTopoArrangement), axis=0)

		self.__nChannels = newNChannels

		#return None

	# end nChannels(self,newNChannels)


	@property
	def nOptodes(self):  # nOptodes getter
		'''
		Number of optodes supported by the optode array.

		When setting the number of optodes:

		* if the number of optodes is smaller than
		the current number of optodes, a warning is issued
		and the optodes indexed highest (on the bottom) in the
			optodesTopoArrangement	array [nOptodes x 3] of float
		will be removed.
		* if the number of optodes is greater than
		the current number of optodes, the new optodes will
		be filled with np.NaN value in optodesTopoArrangement


		:getter: Gets the number of optodes supported by the optode array.
		:setter: Sets the number of optodes supported by the optode array.
		:type: int
		'''

		return self.__nOptodes

	# end nOptodes(self)

	@nOptodes.setter
	def nOptodes(self, newNOptodes):  # nOptodes setter

		# Check parameters
		if type(newNOptodes) is not int:
			msg = self.getClassName() + ':nOptodes: Unexpected parameter type.'
			raise ValueError(msg)
		if newNOptodes < 1:
			msg = self.getClassName() + ':nOptodes: Unexpected attribute value. Number of optodes must be greater or equal than 1.'
			raise ValueError(msg)

		if newNOptodes < self.__nOptodes:
			msg = self.getClassName() + ':nOptodes: New number of optodes is smaller than current number of optodes. Some data will be lost.'
			warnings.warn(msg, RuntimeWarning)

			# Modifying optodesTopoArrangement			array [nOptodes x 3] of float
			self.optodesTopoArrangement = copy.deepcopy(self.optodesTopoArrangement[0:newNOptodes, :])

		elif newNOptodes > self.__nOptodes:
			# Add optodes with np.NaN value in optodesTopoArrangement
			tmpNOptodes = newNOptodes - self.__nOptodes

			# Modifying optodesTopoArrangement
			#tmpOptodesTopoArrangement   = np.zeros((tmpNOptodes, 3), dtype=float)
			tmpOptodesTopoArrangement = np.empty((tmpNOptodes, 3))
			tmpOptodesTopoArrangement[:] = np.NaN
			self.optodesTopoArrangement = np.concatenate((self.optodesTopoArrangement, tmpOptodesTopoArrangement), axis=0)

		self.__nOptodes = newNOptodes

		#return None

	# end nOptodes(self,newNOptodes)


	@property
	def mode(self): # mode getter
		'''
		A string describing the optode array.
		Valid modes depend on the neuroimage type.
		Each neuroimage subclass should check the validity of the modes.
		
		:getter: Gets the description of the optode array.
		:setter: Sets the description of the optode array.
		:type: str
		'''

		return self.__mode
	#end mode(self)

	@mode.setter
	def mode(self, newMode): # mode setter

		#Check parameters
		if type(newMode) is not str:
			msg = self.getClassName() + ':mode: Unexpected parameter type.'
			raise ValueError(msg)

		self.__mode = newMode

		#return None
	#end mode(self, newMode)


	@property
	def typeOptodeArray(self):  # typeOptodeArray getter
		'''
		A string describing whether the optode array is for adults, infants or neonates.

		:getter: Gets the description of what the optode array is for.
		:setter: Sets the description of what the optode array is for.
		:type: str
		'''

		return self.__typeOptodeArray

	# end typeOptodeArray(self)

	@typeOptodeArray.setter
	def typeOptodeArray(self, newTypeOptodeArray):  # typeOptodeArray setter

		# Check parameters
		if type(newTypeOptodeArray) is not str:
			msg = self.getClassName() + ':typeOptodeArray: Unexpected parameter type.'
			raise ValueError(msg)

		self.__typeOptodeArray = newTypeOptodeArray

		#return None

	# end typeOptodeArray(self, newTypeOptodeArray)


	@property
	def chTopoArrangement(self):  # chTopoArrangement getter
		'''
		The topographical arrangement of the channels within the optode array.
		A numpy.ndarray of nChannels x 3 (matrix nChannels x 3) is used for expressing the
			topographical arrangement (3D coordinates) of the channels
			within the optode array.

		The matrix chTopoArrangement has the following 2 dimensions:

		* nChannels (number of channels)
		* 3 (3D coordinates of the channels)

		:getter: Gets the topographical arrangement of the channels within the optode array.
		:setter: Sets the topographical arrangement of the channels within the optode array.
		:type: numpy.ndarray
		'''

		return copy.deepcopy(self.__chTopoArrangement)

	# end chTopoArrangement(self)

	@chTopoArrangement.setter
	def chTopoArrangement(self, newChTopoArrangement):  # chTopoArrangement setter

		# Check parameters
		if type(newChTopoArrangement) is not np.ndarray:
			msg = self.getClassName() + ':chTopoArrangement: Unexpected parameter type.'
			raise ValueError(msg)
		if newChTopoArrangement.ndim != 2:
			msg = self.getClassName() + ':chTopoArrangement: Unexpected attribute value. ' \
				  + 'The topographical arrangement (3D coordinates) of the channels must be stored on 2D [nChannels x 3]'
			raise ValueError(msg)
		#if newChTopoArrangement.shape[0] != self.nChannels:
		#	msg = self.getClassName() + ':chTopoArrangement: Unexpected attribute value. ' \
		#		  + 'Number of rows must be nChannels.'
		#	raise ValueError(msg)
		if newChTopoArrangement.shape[1] != 3:
			msg = self.getClassName() + ':chTopoArrangement: Unexpected attribute value. ' \
				  + 'Number of columns must be 3.'
			raise ValueError(msg)

		self.__chTopoArrangement = copy.deepcopy(newChTopoArrangement)

		#return None

	# end chTopoArrangement(self, newChTopoArrangement)


	@property
	def optodesTopoArrangement(self):  # optodesTopoArrangement getter
		'''
		The topographical arrangement of the optodes within the optode array.
		A numpy.ndarray of nOptodes x 3 (matrix nOptodes x 3) is used for expressing the
			topographical arrangement (3D coordinates) of the optodes
			within the optode array.

		The matrix optodesTopoArrangement has the following 2 dimensions:

		* nOptodes (number of optodes)
		* 3 (3D coordinates of the optodes)

		:getter: Gets the topographical arrangement of the optodes within the optode array.
		:setter: Sets the topographical arrangement of the optodes within the optode array.
		:type: numpy.ndarray
		'''

		return copy.deepcopy(self.__optodesTopoArrangement)

	# end optodesTopoArrangement(self)

	@optodesTopoArrangement.setter
	def optodesTopoArrangement(self, newOptodesTopoArrangement):  # optodesTopoArrangement setter

		# Check parameters
		if type(newOptodesTopoArrangement) is not np.ndarray:
			msg = self.getClassName() + ':optodesTopoArrangement: Unexpected parameter type.'
			raise ValueError(msg)
		if newOptodesTopoArrangement.ndim != 2:
			msg = self.getClassName() + ':optodesTopoArrangement: Unexpected attribute value. ' \
				  + 'The topographical arrangement (3D coordinates) of the optodes must be stored on 2D [nOptodes x 3]'
			raise ValueError(msg)
		#if newOptodesTopoArrangement.shape[0] != self.nOptodes:
		#	msg = self.getClassName() + ':optodesTopoArrangement: Unexpected attribute value. ' \
		#		  + 'Number of rows must be nOptodes.'
		#	raise ValueError(msg)
		if newOptodesTopoArrangement.shape[1] != 3:
			msg = self.getClassName() + ':optodesTopoArrangement: Unexpected attribute value. ' \
				  + 'Number of columns must be 3.'
			raise ValueError(msg)

		self.__optodesTopoArrangement = copy.deepcopy(newOptodesTopoArrangement)

		#return None

	# end optodesTopoArrangement(self, newOptodesTopoArrangement)


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

#class optodeArrayInfo

def plotOptodeArray(oaInfo):
	'''
	Quick rendering of the optode array.
	'''

	nOptodes  = oaInfo.optodesTopoArrangement.shape[0]
	nChannels = oaInfo.chTopoArrangement.shape[0]
	iCh = 0
	for iOp in range(0,nOptodes):
		plt.plot(oaInfo.optodesTopoArrangement[iOp,:], color='blue')
		plt.plot(oaInfo.chTopoArrangement[iCh,:], color='red')
		iCh += 1
	plt.show()

	return
#end plotSyntheticfNIRS(tensor)


def main():
	NewChTopoArrangement  		= np.array([[1, 2, 0], [0, 1, 0], [2, 1, 0], [1, 0, 0]])
	NewOptodesTopoArrangement 	= np.array([[0, 2, 0], [2, 2, 0], [0, 0, 0], [2, 0, 0]])

	oaInfo = optodeArrayInfo(nChannels = 4, nOptodes  = 4, \
							 mode = 'HITACHI ETG-4000 2x2 optode array', typeOptodeArray = 'adult', \
							 chTopoArrangement = NewChTopoArrangement, \
							 optodesTopoArrangement = NewOptodesTopoArrangement)

	print("nChannels: ", oaInfo.nChannels)
	oaInfo.nChannels = 5
	print("nOptodes:  ", oaInfo.nOptodes)
	oaInfo.nOptodes  = 3
	print("mode:      ", oaInfo.mode)
	print("type:      ", oaInfo.typeOptodeArray)
	print("")

	print("Topographical arrangement of the channels within the optode array")
	print("-----------------------------------------------------------------")
	print(oaInfo.chTopoArrangement)
	print("")

	print("Topographical arrangement of the optodes within the optode array")
	print("-----------------------------------------------------------------")
	print(oaInfo.optodesTopoArrangement)

	#plotOptodeArray(oaInfo)
#end main()


if __name__ == '__main__':
	main()


