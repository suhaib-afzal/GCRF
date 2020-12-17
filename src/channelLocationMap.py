# -*- coding: utf-8 -*-
#
#File: channelLocationMap.py
#
'''
Created on Mon Nov 16 20:15:00 2020

Module ***channelLocationMap***

This module implements the class :class:`channelLocationMap <channelLocationMap>`.

:Log:

+-------------+---------+------------------------------------------------------+
| Date        | Authors | Description                                          |
+=============+=========+======================================================+
| 16-Nov-2020 |   FOE   | - Class :class:`channelLocationMap` created but    |
|             |   JJR   |   unfinished.                                        |
+-------------+--------+------------------------------------------------------+


.. sectionauthor:: Felipe Orihuela-Espina <f.orihuela-espina@inaoep.mx> and Jesús Joel Rivas <jrivas@inaoep.mx>
.. codeauthor::    Felipe Orihuela-Espina <f.orihuela-espina@inaoep.mx> and Jesús Joel Rivas <jrivas@inaoep.mx>

'''

import copy
import warnings

import numpy as np
#import random
#import math

#import matplotlib.pyplot as plt

from optodeArrayInfo import optodeArrayInfo

class channelLocationMap:
	'''
	Description:
	A basic class to permit the channel list to be expressed by standard positioning systems.

	A channelLocationMap captures the spatial positioning of channels
	for a neuroimage in different ways:
	   * Real world coordinates (e.g. <X,Y,Z>)
	   * Channel surface location (e.g. standard international system 10/20).
	           Note this may be intended, rather than real

	This class represents the superclass for the classes EEGSignalGenerator and fNIRSSignalGenerator, thereby
	the class provides support for electrodes (EEG) and optodes (fNIRS)

	It further keeps track of the optode arrangement by storing:
		* Optodes real world coordinates (e.g. <X,Y,Z>)
		* Optodes surface location (e.g. standard international system 10/20)
			Note this may be intended, rather than real
		* Optodes types; whether emissor or receptor. See class constants.
		* Optodes pairings (to conform the channels)

	The real world coordinates are cartesian coordinates.

	In addition to the above representations of the channel locations,
	the map allows to allocate the different optodes and channels to
	a physical holder a.k.a. optode array.

	IMPORTANT: The channelLocationMap class does NOT hold any image data!
	It only keeps track of the channel locations!

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

	.id - A numerical identifier of the channelLocationMap

	.description - A short description of the channelLocationMap

	.nChannels - Number of channels supported by the channelLocationMap

	.nOptodes - Number of optodes supported by the channelLocationMap

	.chLocations - The "real-world" 3D locations of the channels
		(numpy.ndarray of nChannels x 3). Perhaps measured with a Polhemus.

	.optodesLocations - The "real-world" 3D locations of the optodes
		(numpy.ndarray of nOptodes x 3). Perhaps measured with a Polhemus.

	.optodesTypes - The type of optodes (numpy.ndarray of nOptodes x 1).
		Optodes can be of one of the following types:
			* 0: Unknown
			* 1: Emission or source
			* 2: Detector
		See constants below

	.referencePoints - The "real-world" 3D locations of the reference
		points; e.g. left ear, right ear, inion, nasion, top (Cz).
		A dictionary is used for registering the reference points.
		The keys are the names and the values are the corresponding locations
			.name - A string with the reference point name. In
				particular the following locations are expected to
				have standard names:
					+ 'nasion' or 'Nz': dent at the upper root of the
						nose bridge;
					+ 'inion' or 'Iz': external occipital protuberance;
					+ 'leftear' or 'LPA': left preauricular point, an
						anterior root of the center of the peak
						region of the tragus;
					+ 'rightear' or 'RPA': right preauricular point
					+ 'top' or 'Cz': Midpoint between Nz and Iz
			.location - A numpy.ndarray of 3 positions for registering
				the "real-world" 3D locations of the reference point

	.surfacePositioningSystem - A string indicating the surface positioning
		system used for reference. Both, channels and optodes
		must be expressed in the same positioning system.
			Currently, only the 10/20 and
		UI 10/10 (default) systems [JurcakV2007] are supported.

	.chSurfacePositions - A standard surface position for each channel
		(a tuple of length nChannels, with a position per channel as a string
		e.g. 'C3'). Unset positions are indicated with an empty string ''.
		Refer to property .surfacePositioningSystem for
		currently supported positioning systems.

	.optodesSurfacePositions - A standard surface position for each optode
		(a tuple of length nOptodes, with a position per optode as a string
		e.g. 'C3'). Unset positions are indicated with an empty string ''.
		Refer to property .surfacePositioningSystem for
		currently supported positioning systems.

	IT IS NOT IMPLEMENTED -> .stereotacticPositioningSystem - A string indicating the stereotactic
		positioning system used for reference. Currently, only the MNI
		(default) and Talairach systems are supported.
			SUPPORT FOR STEREOTACTIC INFORMATION IS LIMITED.

	IT IS NOT IMPLEMENTED -> .chStereotacticPositions - A standard stereotactic position for each
		channel (a nChannelsx3 array with a position per channel).
		Refer to property .stereotacticPositioningSystem for
		currently supported positioning systems.
			SUPPORT FOR STEREOTACTIC INFORMATION IS LIMITED.

	.chOptodeArrays - A numpy.ndarray of nChannels x 1 (column vector) storing the identifier of the
		associated optode array for each channel. Each optode array
		has an associated information (see property .optodeArrays)

	+==================================================+
	| NOTE: optodes numbers, like channels numbers,    |
	|are unique.                                       |
	+==================================================+


	.optodesOptodeArrays - A numpy.ndarray of nOptodes x 1 (column vector) storing the identifier of the
		associated optode array for optode. Each optode array
		has an associated information (see property .optodeArrays)

	.pairings - A numpy.ndarray of nChannels x 2, storing the
		identifiers of the associated optodes conforming each channel.
		Each pairing must hold a light source and a light detector (or nan if
		unknown). The identifier of the source is always stored on the first column
		and the identifier of the detector on the second column.

			NOTE: The constant OPTODE_TYPE_UNKNOWN applies to
				property .optodesTypes


	.optodeArrays - An array of struct holding the information for the
		m different optode arrays. The number of optode arrays (m)
		is at least the maximum in the property chOptodeArrays.
		The struct has the following fields:
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


	==Constants
	.OPTODE_TYPE_UNKNOWN - Unknown optode type.
	.OPTODE_TYPE_EMISOR - Light sources.
	.OPTODE_TYPE_DETECTOR - Light detector.

	NOTE: These constants apply for property .optodesTypes

	'''

	def __init__(self, id = 1, description = 'ChannelLocationMap0001', nChannels = 1, nOptodes  = 1, \
				 chLocations = np.array([[np.NaN, np.NaN, np.NaN]]), optodesLocations = np.array([[np.NaN, np.NaN, np.NaN]]), \
				 optodesTypes = np.array([np.NaN]), referencePoints = dict(), surfacePositioningSystem = 'UI 10/20', \
				 chSurfacePositions = tuple(('',)), optodesSurfacePositions = tuple(('',)), chOptodeArrays = np.array([np.NaN]), \
				 optodesOptodeArrays = np.array([np.NaN]), pairings = np.array([[np.NaN, np.NaN]]), \
				 optodeArrays = np.array([optodeArrayInfo()])):
		'''
		Class constructor.
		
		
		:Properties:
		
		id: A numerical identifier of the channelLocationMap.

		description: A short description of the channelLocationMap.

		nChannels: Number of channels supported by the channelLocationMap.

		nOptodes: Number of optodes supported by the channelLocationMap.

		chLocations: The "real-world" 3D locations of the channels
			(numpy.ndarray of nChannels x 3). Perhaps measured with a Polhemus.

		optodesLocations: The "real-world" 3D locations of the optodes
			(numpy.ndarray of nOptodes x 3). Perhaps measured with a Polhemus.

		optodesTypes: The type of optodes (numpy.ndarray of nOptodes x 1).
			Optodes can be of one of the following types:
				* 0: Unknown
				* 1: Emission or source
				* 2: Detector
			See constants below

		referencePoints: The "real-world" 3D locations of the reference points
			A dictionary is used for registering the reference points.
			The keys are the names (a string with the reference point name) and
			the values are the corresponding locations (a numpy.ndarray of 3 positions
				for registering the "real-world" 3D locations of the reference point).
			Therefore, the pairs key:value must correspond to pairs 'name':(x, y, z)

		surfacePositioningSystem: A string indicating the surface positioning system
			used for reference.

		chSurfacePositions: A standard surface position for each channel
			(a tuple of length nChannels, with a position per channel as a string e.g. 'C3').

		optodesSurfacePositions: A standard surface position for each optode
			(a tuple of length nOptodes, with a position per optode as a string e.g. 'C3').

		chOptodeArrays: A numpy.ndarray of nChannels x 1 (column vector) storing the identifier of the
			associated optode array for each channel.

		optodesOptodeArrays: A numpy.ndarray of nOptodes x 1 (column vector) storing the identifier of the
			associated optode array for optode.

		pairings: A numpy.ndarray of nChannels x 2, storing the identifiers of the associated optodes conforming each channel.
			Each pairing must hold a light source and a light detector (or nan if unknown).
			The identifier of the source is always stored on the first column and the identifier of the detector on the second column.

		optodeArrays: An array of optodeArrayInfo object holding the information for the
			m different optode arrays.


		:Parameters:
		
		:param id: Number for identifying the channelLocationMap.
			Optional. Default is 1.
		:type id: int (positive)

		:param description: A short description of the channelLocationMap.
			Optional. Default is 'ChannelLocationMap0001'.
		:type description: str

		:param nChannels: Number of channels supported by the channelLocationMap.
			Default is 1.
		:type nChannels: int (positive)

		:param nOptodes: Number of optodes supported by the channelLocationMap.
			Default is 1.
		:type nOptodes: int (positive)

		:param chLocations: The "real-world" 3D locations of the channels.
			Optional. Default is np.array([[np.NaN, np.NaN, np.NaN]])
		:type chLocations: numpy.ndarray [nChannels x 3]

		:param optodesLocations: The "real-world" 3D locations of the optodes.
			Optional. Default is np.array([[np.NaN, np.NaN, np.NaN]])
		:type optodesLocations: numpy.ndarray [nOptodes x 3]

		:param optodesTypes: The type of optodes {0: Unknown, 1: Emission or source, 2: Detector}.
			Optional. Default is np.array([np.NaN])
		:type optodesTypes: numpy.ndarray [nOptodes x 1]

		:param referencePoints: The "real-world" 3D locations of the reference points.
			Default is dict()
		:type referencePoints: dict {name1: 3D location1, name2: 3D location2, ..., nameR: 3D locationR}

		:param surfacePositioningSystem: A string indicating the surface positioning system
			used for reference.
			Optional. Default is 'UI 10/20'
		:type surfacePositioningSystem: str

		:param chSurfacePositions: A standard surface position for each channel.
			A tuple is used, with a position per channel as a string e.g. 'C3'.
			Optional. Default is tuple(('',))
		:type chSurfacePositions: tuple (nChannels x 1)

		:param optodesSurfacePositions: A standard surface position for each optode.
			A tuple is used, with a position per optode as a string e.g. 'C3'.
			Optional. Default is tuple(('',))
		:type optodesSurfacePositions: tuple (nOptodes x 1)

		:param chOptodeArrays: An array for storing the identifier of the associated optode array for each channel.
		 	Optional. Default is np.array([np.NaN])
		:type chOptodeArrays: numpy.ndarray [nChannels x 1]

		:param optodesOptodeArrays: An array for storing the identifier of the associated optode array for each optode.
		 	Optional. Default is np.array([np.NaN])
		:type optodesOptodeArrays: numpy.ndarray [nOptodes x 1]

		:param pairings: An array for storing the identifiers of the associated optodes conforming each channel.
		 	Optional. Default is np.array([[np.NaN, np.NaN]])
		:type pairings: numpy.ndarray [nChannels x 2]

		:param optodeArrays: An array of optodeArrayInfo objects holding the information of the
			m different optode arrays.
			Optional. Default is np.array([optodeArrayInfo()])
		:type optodeArrays: numpy.ndarray [m x 1]
		'''
		
		#Check parameters
		if type(id) is not int:
			msg = self.getClassName() + ':__init__: Unexpected parameter type for parameter ''id''.'
			raise ValueError(msg)
		if id < 1:
			msg = self.getClassName() + ':__init__: Unexpected parameter value for parameter ''id''.'
			raise ValueError(msg)

		if type(description) is not str:
			msg = self.getClassName() + ':__init__: Unexpected parameter type for parameter ''description''.'
			raise ValueError(msg)

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

		if type(chLocations) is not np.ndarray:
			msg = self.getClassName() + ':__init__: Unexpected parameter type for parameter ''chLocations''.'
			raise ValueError(msg)
		if chLocations.ndim != 2:
			msg = self.getClassName() + ':__init__: Unexpected attribute value for parameter ''chLocations''. ' \
				  + 'The ''real-world'' 3D locations of the channels must be stored on 2D [nChannels x 3]'
			raise ValueError(msg)
		if chLocations.shape[0] != nChannels:
			msg = self.getClassName() + ':__init__: Unexpected attribute value for parameter ''chLocations''. ' \
				  + 'Number of rows must be nChannels.'
			raise ValueError(msg)
		if chLocations.shape[1] != 3:
			msg = self.getClassName() + ':__init__: Unexpected attribute value for parameter ''chLocations''. ' \
				  + 'Number of columns must be 3.'
			raise ValueError(msg)

		if type(optodesLocations) is not np.ndarray:
			msg = self.getClassName() + ':__init__: Unexpected parameter type for parameter ''optodesLocations''.'
			raise ValueError(msg)
		if optodesLocations.ndim != 2:
			msg = self.getClassName() + ':__init__: Unexpected attribute value for parameter ''optodesLocations''. ' \
				  + 'The ''real-world'' 3D locations of the optodes must be stored on 2D [nOptodes x 3]'
			raise ValueError(msg)
		if optodesLocations.shape[0] != nOptodes:
			msg = self.getClassName() + ':__init__: Unexpected attribute value for parameter ''optodesLocations''. ' \
				  + 'Number of rows must be nOptodes.'
			raise ValueError(msg)
		if optodesLocations.shape[1] != 3:
			msg = self.getClassName() + ':__init__: Unexpected attribute value for parameter ''optodesLocations''. ' \
				  + 'Number of columns must be 3.'
			raise ValueError(msg)

		if type(optodesTypes) is not np.ndarray:
			msg = self.getClassName() + ':__init__: Unexpected parameter type for parameter ''optodesTypes''.'
			raise ValueError(msg)
		if optodesTypes.ndim != 1:
			msg = self.getClassName() + ':__init__: Unexpected attribute value for parameter ''optodesTypes''. ' \
				  + 'The information of the optodes types must be stored on 1D [nOptodes x 1] array'
			raise ValueError(msg)
		if optodesTypes.shape[0] != nOptodes:
			msg = self.getClassName() + ':__init__: Unexpected attribute value for parameter ''optodesTypes''. ' \
				  + 'Number of elements must be nOptodes.'
			raise ValueError(msg)
		for elem in optodesTypes:
			if elem not in {0, 1, 2} and not np.isnan(elem):
				msg = self.getClassName() + ':__init__: Unexpected parameter value for parameter ''optodesTypes''.'
				raise ValueError(msg)

		if type(referencePoints) is not dict:
			msg = self.getClassName() + ':__init__: Unexpected parameter type for parameter ''referencePoints''.'
			raise ValueError(msg)
		for key in referencePoints.keys():
			if type(key) is not str:
				msg = self.getClassName() + ':__init__: Unexpected type of a key for parameter ''referencePoints''.'
				raise ValueError(msg)
		for value in referencePoints.values():
			if type(value) is not np.ndarray:
				msg = self.getClassName() + ':__init__: Unexpected type of a value for parameter ''referencePoints''.'
				raise ValueError(msg)
			if value.ndim != 1:
				msg = self.getClassName() + ':__init__: Unexpected attribute value of a value for parameter ''referencePoints''. ' \
					  + 'The reference point 3D location must be stored on an array of one dimension 1D'
				raise ValueError(msg)
			if value.shape[0] != 3:
				msg = self.getClassName() + ':__init__: Unexpected attribute value of a value for parameter ''referencePoints''. ' \
					  + 'The reference point 3D location must be stored on an array of 3 components'
				raise ValueError(msg)

		if type(surfacePositioningSystem) is not str:
			msg = self.getClassName() + ':__init__: Unexpected parameter type for parameter ''surfacePositioningSystem''.'
			raise ValueError(msg)

		if type(chSurfacePositions) is not tuple:
			msg = self.getClassName() + ':__init__: Unexpected parameter type for parameter ''chSurfacePositions''.'
			raise ValueError(msg)
		if len(chSurfacePositions) != nChannels:
			msg = self.getClassName() + ':__init__: Unexpected attribute value for parameter ''chSurfacePositions''. ' \
				  + 'Number of positions must be nChannels.'
			raise ValueError(msg)
		for elem in chSurfacePositions:
			if type(elem) is not str:
				msg = self.getClassName() + ':__init__: Unexpected parameter value for parameter ''chSurfacePositions''.'
				raise ValueError(msg)

		if type(optodesSurfacePositions) is not tuple:
			msg = self.getClassName() + ':__init__: Unexpected parameter type for parameter ''optodesSurfacePositions''.'
			raise ValueError(msg)
		if len(optodesSurfacePositions) != nOptodes:
			msg = self.getClassName() + ':__init__: Unexpected attribute value for parameter ''optodesSurfacePositions''. ' \
				  + 'Number of positions must be nOptodes.'
			raise ValueError(msg)
		for elem in optodesSurfacePositions:
			if type(elem) is not str:
				msg = self.getClassName() + ':__init__: Unexpected parameter value for parameter ''optodesSurfacePositions''.'
				raise ValueError(msg)

		if type(chOptodeArrays) is not np.ndarray:
			msg = self.getClassName() + ':__init__: Unexpected parameter type for parameter ''chOptodeArrays''.'
			raise ValueError(msg)
		if chOptodeArrays.ndim != 1:
			msg = self.getClassName() + ':__init__: Unexpected attribute value for parameter ''chOptodeArrays''. ' \
				  + 'The identifier of the associated optode array for each channel must be stored on 1D [nChannels x 1] array'
			raise ValueError(msg)
		if chOptodeArrays.shape[0] != nChannels and nOptodes > 1:	# Another alternative is:  if len(chOptodeArrays) != nChannels:
			msg = self.getClassName() + ':__init__: Unexpected attribute value for parameter ''chOptodeArrays''. ' \
				  + 'Number of elements must be nChannels.'
			raise ValueError(msg)
		for elem in chOptodeArrays:
			if type(elem) not in {np.int32, np.int64, np.float64}:   # np.float64 is the type of np.NaN, hence it must be included
				msg = self.getClassName() + ':__init__: Unexpected parameter value for parameter ''chOptodeArrays''.'
				raise ValueError(msg)
			if elem < 0:
				msg = self.getClassName() + ':__init__: Unexpected parameter value for parameter ''chOptodeArrays''.'
				raise ValueError(msg)

		if type(optodesOptodeArrays) is not np.ndarray:
			msg = self.getClassName() + ':__init__: Unexpected parameter type for parameter ''optodesOptodeArrays''.'
			raise ValueError(msg)
		if optodesOptodeArrays.ndim != 1:
			msg = self.getClassName() + ':__init__: Unexpected attribute value for parameter ''optodesOptodeArrays''. ' \
				  + 'The identifier of the associated optode array for each optode must be stored on 1D [nOptodes x 1] array'
			raise ValueError(msg)
		if optodesOptodeArrays.shape[0] != nOptodes:
			msg = self.getClassName() + ':__init__: Unexpected attribute value for parameter ''optodesOptodeArrays''. ' \
				  + 'Number of elements must be nOptodes.'
			raise ValueError(msg)
		for elem in optodesOptodeArrays:
			if type(elem) not in {np.int32, np.int64, np.float64}:   # np.float64 is the type of np.NaN, hence it must be included
				msg = self.getClassName() + ':__init__: Unexpected parameter value for parameter ''optodesOptodeArrays''.'
				raise ValueError(msg)
			if elem < 0:
				msg = self.getClassName() + ':__init__: Unexpected parameter value for parameter ''optodesOptodeArrays''.'
				raise ValueError(msg)

		if type(pairings) is not np.ndarray:
			msg = self.getClassName() + ':__init__: Unexpected parameter type for parameter ''pairings''.'
			raise ValueError(msg)
		if pairings.ndim != 2:
			msg = self.getClassName() + ':__init__: Unexpected attribute value for parameter ''pairings''. ' \
				  + 'The identifiers of the associated optodes conforming each channel must be stored on 2D [nChannels x 2] array'
			raise ValueError(msg)
		if pairings.shape[0] != nChannels and nOptodes > 1:
			msg = self.getClassName() + ':__init__: Unexpected attribute value for parameter ''pairings''. ' \
				  + 'Number of rows must be nChannels.'
			raise ValueError(msg)
		for elem in pairings:	# elem is an array of 2 components [a0, a1]
			for i in {0, 1}:
				if type(elem[i]) not in {np.int32, np.int64, np.float64}:  # np.float64 is the type of np.NaN, hence it must be included
					msg = self.getClassName() + ':__init__: Unexpected parameter value for parameter ''pairings''.'
					raise ValueError(msg)
				if elem[i] < 0:
					msg = self.getClassName() + ':__init__: Unexpected parameter value for parameter ''pairings''.'
					raise ValueError(msg)

		if type(optodeArrays) is not np.ndarray:
			msg = self.getClassName() + ':__init__: Unexpected parameter type for parameter ''optodeArrays''.'
			raise ValueError(msg)
		if optodeArrays.ndim != 1:
			msg = self.getClassName() + ':__init__: Unexpected attribute value for parameter ''optodeArrays''. ' \
				  + 'The optodeArrayInfo objects holding the information of the m different optode arrays must be stored on 1D [m x 1] array'
			raise ValueError(msg)
		for elem in optodeArrays:
			if type(elem) is not optodeArrayInfo:
				msg = self.getClassName() + ':__init__: Unexpected parameter value for parameter ''optodeArrays''.'
				raise ValueError(msg)

		#Initialize
		self.id 			= id 			# Numerical identifier of the channelLocationMap
		self.description 	= description	# String for a short description of the channelLocationMap
		self.__nChannels 	= nChannels		# Number of channels supported by the channelLocationMap
		self.__nOptodes  	= nOptodes		# Number of optodes supported by the channelLocationMap

		self.chLocations      = copy.deepcopy(chLocations)			# An array [nChannels x 3] for 3D "real world" location of the channels.
		self.optodesLocations = copy.deepcopy(optodesLocations) 	# An array [nOptodes x 3] for 3D "real world" location of the optodes.
		self.optodesTypes     = copy.deepcopy(optodesTypes)			# An array [nOptodes x 1] for Optodes types;
																	# 	(0) Unknown, (1) Emission, (2) Detector

		self.referencePoints = referencePoints		# It is a dict(), the pairs key:value must correspond to pairs 'name':(x, y, z)

		self.surfacePositioningSystem = surfacePositioningSystem # string indicating the surface positioning system used for reference
		self.chSurfacePositions 	  = chSurfacePositions 		 # A tuple (nChannels x 1) for Channels surface positions
		self.optodesSurfacePositions  = optodesSurfacePositions  # A tuple (nOptodes x 1) for Optodes surface positions

		#self.stereotacticPositioningSystem 	= 'MNI'
		#self.chStereotacticPositions 		= (math.nan, math.nan, math.nan) # 3D MNI location of the channels.

		self.chOptodeArrays 	 = copy.deepcopy(chOptodeArrays) 	 # An array [nChannels x 1] column vector for storing
																	 # 	the identifier of the associated optode array for each channel.
																	 # 	Each optode array is identified by a numerical value from 0, 1, 2, ...

		self.optodesOptodeArrays = copy.deepcopy(optodesOptodeArrays) # An array [nOptodes x 1] column vector for storing
																	  #  the identifier of the associated optode array for each optode.
																	  #  Each optode array is identified by a numerical value from 0, 1, 2, ...

		self.pairings = copy.deepcopy(pairings) 	# An array [nChannels x 2] for indicating the pair of associated optodes
													# 	conforming each channel.

		self.optodeArrays = copy.deepcopy(optodeArrays)	# An array [m x 1] of optodeArrayInfo objects holding the information
														# 	of the m different optode arrays.

		# Create constant OPTODE_TYPE_UNKNOWN = 0, constant OPTODE_TYPE_EMISOR = 1, and constant OPTODE_TYPE_DETECTOR = 2
		self.__OPTODE_TYPE_UNKNOWN  = 0
		self.__OPTODE_TYPE_EMISOR   = 1
		self.__OPTODE_TYPE_DETECTOR = 2

		return
	#end __init__(self, id = 1, ..., optodeArrays = np.array([optodeArrayInfo()]))


	#Properties getters/setters
	#
	# Remember: Sphinx ignores docstrings on property setters so all
	#documentation for a property must be on the @property method

	# Note that python does not have constants nor static constants,
	# so in order to have a constant, a new property is defined
	# with only a getter method and the setter method raises an error message.
	@property
	def OPTODE_TYPE_UNKNOWN(self):  # OPTODE_TYPE_UNKNOWN getter
		'''
		Constant OPTODE_TYPE_UNKNOWN = 0

		:getter: Gets constant OPTODE_TYPE_UNKNOWN.
		:type: int
		'''

		return 0
	# end OPTODE_TYPE_UNKNOWN(self)

	@OPTODE_TYPE_UNKNOWN.setter
	def OPTODE_TYPE_UNKNOWN(self, value):  # OPTODE_TYPE_UNKNOWN setter
		'''
		Constant OPTODE_TYPE_UNKNOWN = 0

		:setter: Raise an error message because the value of constant OPTODE_TYPE_UNKNOWN is being tried to be changed.
		:type: int
		'''

		msg = self.getClassName() + ':OPTODE_TYPE_UNKNOWN: ConstantError: Can not rebind const.'
		raise ValueError(msg)
	# end OPTODE_TYPE_UNKNOWN(self, value)

	@property
	def OPTODE_TYPE_EMISOR(self):  # OPTODE_TYPE_EMISOR getter
		'''
		Constant OPTODE_TYPE_EMISOR = 1

		:getter: Gets constant OPTODE_TYPE_EMISOR.
		:type: int
		'''

		return 1
	# end OPTODE_TYPE_EMISOR(self)

	@OPTODE_TYPE_EMISOR.setter
	def OPTODE_TYPE_EMISOR(self, value):  # OPTODE_TYPE_EMISOR setter
		'''
		Constant OPTODE_TYPE_EMISOR = 1

		:setter: Raise an error message because the value of constant OPTODE_TYPE_EMISOR is being tried to be changed.
		:type: int
		'''

		msg = self.getClassName() + ':OPTODE_TYPE_EMISOR: ConstantError: Can not rebind const.'
		raise ValueError(msg)
	# end OPTODE_TYPE_EMISOR(self, value)

	@property
	def OPTODE_TYPE_DETECTOR(self):  # OPTODE_TYPE_DETECTOR getter
		'''
		Constant OPTODE_TYPE_DETECTOR = 2

		:getter: Gets constant OPTODE_TYPE_DETECTOR.
		:type: int
		'''

		return 2
	# end OPTODE_TYPE_DETECTOR(self)

	@OPTODE_TYPE_DETECTOR.setter
	def OPTODE_TYPE_DETECTOR(self, value):  # OPTODE_TYPE_DETECTOR setter
		'''
		Constant OPTODE_TYPE_DETECTOR = 2

		:setter: Raise an error message because the value of constant OPTODE_TYPE_DETECTOR is being tried to be changed.
		:type: int
		'''

		msg = self.getClassName() + ':OPTODE_TYPE_DETECTOR: ConstantError: Can not rebind const.'
		raise ValueError(msg)
	# end OPTODE_TYPE_DETECTOR(self, value)


	@property
	def id(self):  # id getter
		'''
		A numerical identifier of the channelLocationMap.

		:getter: Gets the numerical identifier of the channelLocationMap.
		:setter: Sets the numerical identifier of the channelLocationMap.
		:type: int
		'''

		return self.__id

	# end id(self)

	@id.setter
	def id(self, newId):  # id setter

		# Check parameters
		if type(newId) is not int:
			msg = self.getClassName() + ':id: Unexpected parameter type.'
			raise ValueError(msg)

		self.__id = newId

	# return None
	# end id(self, newId)


	@property
	def description(self):  # description getter
		'''
		A short description of the channelLocationMap.

		:getter: Gets the short description of the channelLocationMap.
		:setter: Sets the short description of the channelLocationMap.
		:type: str
		'''

		return self.__description

	# end description(self)

	@description.setter
	def description(self, newDescription):  # description setter

		# Check parameters
		if type(newDescription) is not str:
			msg = self.getClassName() + ':description: Unexpected parameter type.'
			raise ValueError(msg)

		self.__description = newDescription

	# return None
	# end description(self, newDescription)


	@property
	def nChannels(self):  # nChannels getter
		'''
		Number of channels supported by the channelLocationMap.

		When setting the number of channels:

		* if the number of channels is smaller than
		the current number of channels, a warning is issued
		and the channels indexed rightmost in the
			* chSurfacePositions 	tuple (nChannels x 1) of str,
		and the channels indexed highest (on the bottom) in the
			* chLocations 			array [nChannels x 3] of float,
			* chOptodeArrays 		array [nChannels x 1] of int,
			* pairings 				array [nChannels x 2] of int,
			* chTopoArrangement		array [nChannels x 3] of float <- in object  optodeArrayInfo
		will be removed.
		* if the number of channels is greater than
		the current number of channels, the new channels will
		be filled with
			'' in chSurfacePositions,
		and with
			np.NaN value in chLocations, chOptodeArrays, pairings, and chTopoArrangement


		:getter: Gets the number of channels supported by the channelLocationMap.
		:setter: Sets the number of channels supported by the channelLocationMap.
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

			# Modifying chSurfacePositions 	tuple (nChannels x 1) of str
			self.chSurfacePositions = self.chSurfacePositions[0:newNChannels]

			# Modifying chLocations			array [nChannels x 3] of float
			self.chLocations = copy.deepcopy(self.chLocations[0:newNChannels, :])

			# Modifying chOptodeArrays 		array [nChannels x 1] of int
			self.chOptodeArrays = copy.deepcopy(self.chOptodeArrays[0:newNChannels])

			# Modifying pairings			array [nChannels x 2] of int
			self.pairings = copy.deepcopy(self.pairings[0:newNChannels, :])

		elif newNChannels > self.__nChannels:
			# Add channels with '' in chSurfacePositions  and
			# 	add channels with np.NaN value in chLocations, chOptodeArrays, pairings, and  chTopoArrangement
			tmpNChannels = newNChannels - self.nChannels

			# Modifying chSurfacePositions
			tmpList = list()
			for i in range(0,tmpNChannels):
				tmpList.append('')
			self.chSurfacePositions = self.chSurfacePositions + tuple(tmpList)

			# Modifying chLocations
			#tmpChLocations   = np.zeros((tmpNChannels, 3), dtype=float)
			tmpChLocations = np.empty((tmpNChannels, 3))
			tmpChLocations[:] = np.NaN
			self.chLocations = np.concatenate((self.chLocations, tmpChLocations), axis=0)

			# Modifying chOptodeArrays
			#tmpChOptodeArrays   = np.zeros((tmpNChannels, 1), dtype=int)
			tmpChOptodeArrays = np.empty((tmpNChannels, 1))
			tmpChOptodeArrays[:] = np.NaN
			self.chOptodeArrays = np.concatenate((self.chOptodeArrays, tmpChOptodeArrays), axis=None)

			# Modifying pairings
			tmpPairings = np.empty((tmpNChannels, 2))
			tmpPairings[:] = np.NaN
			self.pairings = np.concatenate((self.pairings, tmpPairings), axis=0)

		# Modifying chTopoArrangement in object  optodeArrayInfo
		for elem in self.optodeArrays:  # Each element in the array 'self.optodeArrays' [m x 1]  is an object optodeArrayInfo
			elem.nChannels = newNChannels  	# type(elem) is optodeArrayInfo.
											# 	Setting a new value to nChannels property of the object optodeArrayInfo will cause
											# 	the invocation of the method  @nChannels.setter  in optodeArrayInfo, and this method
											#   will deal with the new value of nChannels:
											# 		* if newNChannels < self.nChannels: it will raise a warning, and
											# 			the channels indexed highest (on the bottom) in the chTopoArrangement property
											# 			of optodeArrayInfo will be removed.
											# 		* elif newNChannels > self.nChannels: it will add channels with np.NaN value in
											# 			the chTopoArrangement property of optodeArrayInfo

		self.__nChannels = newNChannels

		#return None

	# end nChannels(self, newNChannels)


	@property
	def nOptodes(self):  # nOptodes getter
		'''
		Number of optodes supported by the channelLocationMap.

		When setting the number of optodes:

		* if the number of optodes is smaller than
		the current number of optodes, a warning is issued
		and the optodes indexed rightmost in the
			* optodesSurfacePositions 	tuple (nOptodes x 1) of str,
		and the optodes indexed highest (on the bottom) in the
			* optodesLocations			array [nOptodes x 3] of float,
			* optodesTypes 				array [nOptodes x 1] of int,
			* optodesOptodeArrays		array [nOptodes x 1] of int,
			* optodesTopoArrangement	array [nOptodes x 3] of float <- in object  optodeArrayInfo
		will be removed.
		* if the number of optodes is greater than
		the current number of optodes, the new optodes will
		be filled with
			'' in optodesSurfacePositions,
		and with
			np.NaN value in optodesLocations, optodesTypes, optodesOptodeArrays, and optodesTopoArrangement


		:getter: Gets the number of optodes supported by the channelLocationMap.
		:setter: Sets the number of optodes supported by the channelLocationMap.
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

			# Modifying optodesSurfacePositions 	tuple (nOptodes x 1) of str
			self.optodesSurfacePositions = self.optodesSurfacePositions[0:newNOptodes]

			# Modifying optodesLocations			array [nOptodes x 3] of float
			self.optodesLocations = copy.deepcopy(self.optodesLocations[0:newNOptodes, :])

			# Modifying optodesTypes 				array [nOptodes x 1] of int
			self.optodesTypes = copy.deepcopy(self.optodesTypes[0:newNOptodes])

			# Modifying optodesOptodeArrays			array [nOptodes x 1] of int
			self.optodesOptodeArrays = copy.deepcopy(self.optodesOptodeArrays[0:newNOptodes])

		elif newNOptodes > self.__nOptodes:
			# Add optodes with '' in optodesSurfacePositions  and
			# 	add optodes with np.NaN value in optodesLocations, optodesTypes, optodesOptodeArrays, and  optodesTopoArrangement
			tmpNOptodes = newNOptodes - self.nOptodes

			# Modifying optodesSurfacePositions
			tmpList = list()
			for i in range(0,tmpNOptodes):
				tmpList.append('')
			self.optodesSurfacePositions = self.optodesSurfacePositions + tuple(tmpList)

			# Modifying optodesLocations
			#tmpOptodesLocations   = np.zeros((tmpNOptodes, 3), dtype=float)
			tmpOptodesLocations = np.empty((tmpNOptodes, 3))
			tmpOptodesLocations[:] = np.NaN
			self.optodesLocations = np.concatenate((self.optodesLocations, tmpOptodesLocations), axis=0)

			# Modifying optodesTypes
			#tmpOptodesTypes   = np.zeros((tmpNOptodes, 1), dtype=int)
			tmpOptodesTypes = np.empty((tmpNOptodes, 1))
			tmpOptodesTypes[:] = np.NaN
			self.optodesTypes = np.concatenate((self.optodesTypes, tmpOptodesTypes), axis=None)

			# Modifying optodesOptodeArrays
			tmpOptodesOptodeArrays = np.empty((tmpNOptodes, 1))
			tmpOptodesOptodeArrays[:] = np.NaN
			self.optodesOptodeArrays = np.concatenate((self.optodesOptodeArrays, tmpOptodesOptodeArrays), axis=None)

		# Modifying optodesTopoArrangement in object  optodeArrayInfo
		for elem in self.optodeArrays:  # Each element in the array 'self.optodeArrays' [m x 1]  is an object optodeArrayInfo
			elem.nOptodes = newNOptodes  	# type(elem) is optodeArrayInfo.
											# 	Setting a new value to nOptodes property of the object optodeArrayInfo will cause
											# 	the invocation of the method  @nOptodes.setter  in optodeArrayInfo, and this method
											#   will deal with the new value of nOptodes:
											# 		* if newNOptodes < self.nOptodes: it will raise a warning, and
											# 			the optodes indexed highest (on the bottom) in the optodesTopoArrangement property
											# 			of optodeArrayInfo will be removed.
											# 		* elif newNOptodes > self.nOptodes: it will add optodes with np.NaN value in
											# 			the optodesTopoArrangement property of optodeArrayInfo

		self.__nOptodes = newNOptodes

		#return None

	# end nOptodes(self, newNOptodes)


	@property
	def chLocations(self):  # chLocations getter
		'''
		The "real-world" 3D locations of the channels.

		The matrix chLocations has 2 dimensions, namely:

		* nChannels (number of channels)
		* 3 (3D coordinates of the channels)

		:getter: Gets the 3D locations of the channels.
		:setter: Sets the 3D locations of the channels.
		:type: np.ndarray [nChannels x 3]
		'''

		return copy.deepcopy(self.__chLocations)

	# end chLocations(self)

	@chLocations.setter
	def chLocations(self, newChLocations):  # chLocations setter

		# Check parameters
		if type(newChLocations) is not np.ndarray:
			msg = self.getClassName() + ':chLocations: Unexpected parameter type.'
			raise ValueError(msg)
		if newChLocations.ndim != 2:
			msg = self.getClassName() + ':chLocations: Unexpected attribute value. ' \
				  + 'The ''real-world'' 3D locations of the channels must be stored on 2D [nChannels x 3]'
			raise ValueError(msg)
		#if newChLocations.shape[0] != self.nChannels:
		#	msg = self.getClassName() + ':chLocations: Unexpected attribute value. ' \
		#		  + 'Number of rows must be nChannels.'
		#	raise ValueError(msg)
		if newChLocations.shape[1] != 3:
			msg = self.getClassName() + ':chLocations: Unexpected attribute value. ' \
				  + 'Number of columns must be 3.'
			raise ValueError(msg)

		self.__chLocations = copy.deepcopy(newChLocations)

		#return None

	# end chLocations(self, newChLocations)


	@property
	def optodesLocations(self):  # optodesLocations getter
		'''
		The "real-world" 3D locations of the optodes.

		The matrix optodesLocations has 2 dimensions, namely:

		* nOptodes (number of optodes)
		* 3 (3D coordinates of the optodes)

		:getter: Gets the 3D locations of the optodes.
		:setter: Sets the 3D locations of the optodes.
		:type: np.ndarray [nOptodes x 3]
		'''

		return copy.deepcopy(self.__optodesLocations)

	# end optodesLocations(self)

	@optodesLocations.setter
	def optodesLocations(self, newOptodesLocations):  # optodesLocations setter

		# Check parameters
		if type(newOptodesLocations) is not np.ndarray:
			msg = self.getClassName() + ':optodesLocations: Unexpected parameter type.'
			raise ValueError(msg)
		if newOptodesLocations.ndim != 2:
			msg = self.getClassName() + ':optodesLocations: Unexpected attribute value. ' \
				  + 'The ''real-world'' 3D locations of the optodes must be stored on 2D [nOptodes x 3]'
			raise ValueError(msg)
		#if newOptodesLocations.shape[0] != self.nOptodes:
		#	msg = self.getClassName() + ':optodesLocations: Unexpected attribute value. ' \
		#		  + 'Number of rows must be nOptodes.'
		#	raise ValueError(msg)
		if newOptodesLocations.shape[1] != 3:
			msg = self.getClassName() + ':optodesLocations: Unexpected attribute value. ' \
				  + 'Number of columns must be 3.'
			raise ValueError(msg)

		self.__optodesLocations = copy.deepcopy(newOptodesLocations)

		#return None

	# end optodesLocations(self, newOptodesLocations)


	@property
	def optodesTypes(self):  # optodesTypes getter
		'''
		The type of optodes {0: Unknown, 1: Emission or source, 2: Detector}.

		:getter: Gets the types of the optodes.
		:setter: Sets the types of the optodes.
		:type: np.ndarray [nOptodes x 1]
		'''

		return copy.deepcopy(self.__optodesTypes)

	# end optodesTypes(self)

	@optodesTypes.setter
	def optodesTypes(self, newOptodesTypes):  # optodesTypes setter

		# Check parameters
		if type(newOptodesTypes) is not np.ndarray:
			msg = self.getClassName() + ':optodesTypes: Unexpected parameter type.'
			raise ValueError(msg)
		if newOptodesTypes.ndim != 1:
			msg = self.getClassName() + ':optodesTypes: Unexpected attribute value. ' \
				  + 'The information of the optodes types must be stored on 1D [nOptodes x 1] array'
			raise ValueError(msg)
		#if newOptodesTypes.shape[0] != self.nOptodes:
		#	msg = self.getClassName() + ':optodesTypes: Unexpected attribute value. ' \
		#		  + 'Number of elements must be nOptodes.'
		#	raise ValueError(msg)
		for elem in newOptodesTypes:
			if elem not in {0, 1, 2} and not np.isnan(elem):
				msg = self.getClassName() + ':optodesTypes: Unexpected parameter value.'
				raise ValueError(msg)

		self.__optodesTypes = copy.deepcopy(newOptodesTypes)

		#return None

	# end optodesTypes(self, newOptodesTypes)


	@property
	def referencePoints(self):  # referencePoints getter
		'''
		The "real-world" 3D locations of the reference points.

		A dictionary is used for registering the reference points.
			* keys are the names (a string with the reference point name) and
			* values are the corresponding locations (a numpy.ndarray of 3 positions
				for registering the "real-world" 3D locations of the reference point).
			Therefore, the pairs key:value must correspond to pairs 'name':(x, y, z)

		:getter: Gets the 3D locations of the reference points.
		:setter: Sets the 3D locations of the reference points.
		:type: dict {name1: 3D location1, name2: 3D location2, ..., nameR: 3D locationR}
		'''

		return self.__referencePoints

	# end referencePoints(self)

	@referencePoints.setter
	def referencePoints(self, newReferencePoints):  # referencePoints setter

		# Check parameters
		if type(newReferencePoints) is not dict:
			msg = self.getClassName() + ':referencePoints: Unexpected parameter type.'
			raise ValueError(msg)
		for key in newReferencePoints.keys():
			if type(key) is not str:
				msg = self.getClassName() + ':referencePoints: Unexpected type of a key.'
				raise ValueError(msg)
		for value in newReferencePoints.values():
			if type(value) is not np.ndarray:
				msg = self.getClassName() + ':referencePoints: Unexpected type of a value.'
				raise ValueError(msg)
			if value.ndim != 1:
				msg = self.getClassName() + ':referencePoints: Unexpected attribute value of a value. ' \
					  + 'The reference point 3D location must be stored on an array of one dimension 1D'
				raise ValueError(msg)
			if value.shape[0] != 3:
				msg = self.getClassName() + ':referencePoints: Unexpected attribute value of a value. ' \
					  + 'The reference point 3D location must be stored on an array of 3 components'
				raise ValueError(msg)

		self.__referencePoints = newReferencePoints

		#return None

	# end referencePoints(self, newReferencePoints)


	@property
	def surfacePositioningSystem(self):  # surfacePositioningSystem getter
		'''
		A string indicating the surface positioning system used for reference.

		:getter: Gets the string indicating the surface positioning system used for reference.
		:setter: Sets the string indicating the surface positioning system used for reference.
		:type: str
		'''

		return self.__surfacePositioningSystem

	# end surfacePositioningSystem(self)

	@surfacePositioningSystem.setter
	def surfacePositioningSystem(self, newSurfacePositioningSystem):  # surfacePositioningSystem setter

		# Check parameters
		if type(newSurfacePositioningSystem) is not str:
			msg = self.getClassName() + ':surfacePositioningSystem: Unexpected parameter type.'
			raise ValueError(msg)

		self.__surfacePositioningSystem = newSurfacePositioningSystem

	# return None
	# end surfacePositioningSystem(self, newSurfacePositioningSystem)


	@property
	def chSurfacePositions(self):  # chSurfacePositions getter
		'''
		A standard surface position for each channel.
		A tuple is used, with a position per channel as a string e.g. 'C3'.

		:getter: Gets the standard surface position for each channel.
		:setter: Sets the standard surface position for each channel.
		:type: tuple (nChannels x 1)
		'''

		return self.__chSurfacePositions

	# end chSurfacePositions(self)

	@chSurfacePositions.setter
	def chSurfacePositions(self, newChSurfacePositions):  # chSurfacePositions setter

		# Check parameters
		if type(newChSurfacePositions) is not tuple:
			msg = self.getClassName() + ':chSurfacePositions: Unexpected parameter type.'
			raise ValueError(msg)
		#if len(newChSurfacePositions) != self.nChannels:
		#	msg = self.getClassName() + ':chSurfacePositions: Unexpected attribute value. ' \
		#		  + 'Number of positions must be nChannels.'
		#	raise ValueError(msg)
		for elem in newChSurfacePositions:
			if type(elem) is not str:
				msg = self.getClassName() + ':chSurfacePositions: Unexpected attribute value.'
				raise ValueError(msg)

		self.__chSurfacePositions = newChSurfacePositions

		#return None

	# end chSurfacePositions(self, newChSurfacePositions)


	@property
	def optodesSurfacePositions(self):  # optodesSurfacePositions getter
		'''
		A standard surface position for each optode.
		A tuple is used, with a position per optode as a string e.g. 'C3'.

		:getter: Gets the standard surface position for each optode.
		:setter: Sets the standard surface position for each optode.
		:type: tuple (nOptodes x 1)
		'''

		return self.__optodesSurfacePositions

	# end optodesSurfacePositions(self)

	@optodesSurfacePositions.setter
	def optodesSurfacePositions(self, newOptodesSurfacePositions):  # optodesSurfacePositions setter

		# Check parameters
		if type(newOptodesSurfacePositions) is not tuple:
			msg = self.getClassName() + ':optodesSurfacePositions: Unexpected parameter type.'
			raise ValueError(msg)
		#if len(newOptodesSurfacePositions) != self.nOptodes:
		#	msg = self.getClassName() + ':optodesSurfacePositions: Unexpected attribute value. ' \
		#		  + 'Number of positions must be nOptodes.'
		#	raise ValueError(msg)
		for elem in newOptodesSurfacePositions:
			if type(elem) is not str:
				msg = self.getClassName() + ':optodesSurfacePositions: Unexpected parameter value.'
				raise ValueError(msg)

		self.__optodesSurfacePositions = newOptodesSurfacePositions

		#return None

	# end optodesSurfacePositions(self, newOptodesSurfacePositions)


	@property
	def chOptodeArrays(self):  # chOptodeArrays getter
		'''
		An array for storing the identifier of the associated optode array for each channel.

		:getter: Gets the array that stores the identifier of the associated optode array for each channel.
		:setter: Sets the array that stores the identifier of the associated optode array for each channel.
		:type: np.ndarray [nChannels x 1]
		'''

		return copy.deepcopy(self.__chOptodeArrays)

	# end chOptodeArrays(self)

	@chOptodeArrays.setter
	def chOptodeArrays(self, newChOptodeArrays):  # chOptodeArrays setter

		# Check parameters
		if type(newChOptodeArrays) is not np.ndarray:
			msg = self.getClassName() + ':chOptodeArrays: Unexpected parameter type.'
			raise ValueError(msg)
		if newChOptodeArrays.ndim != 1:
			msg = self.getClassName() + ':chOptodeArrays: Unexpected attribute value. ' \
				  + 'The identifier of the associated optode array for each channel must be stored on 1D [nChannels x 1] array'
			raise ValueError(msg)
		#if newChOptodeArrays.shape[0] != self.nChannels and self.nOptodes > 1:	# Another alternative is:  if len(newChOptodeArrays) != self.nChannels:
		#	msg = self.getClassName() + ':chOptodeArrays: Unexpected attribute value. ' \
		#		  + 'Number of elements must be nChannels.'
		#	raise ValueError(msg)
		for elem in newChOptodeArrays:
			if type(elem) not in {np.int32, np.int64, np.float64}:   # np.float64 is the type of np.NaN, hence it must be included
				msg = self.getClassName() + ':chOptodeArrays: Unexpected parameter value.'
				raise ValueError(msg)
			if elem < 0:
				msg = self.getClassName() + ':chOptodeArrays: Unexpected parameter value.'
				raise ValueError(msg)

		self.__chOptodeArrays = copy.deepcopy(newChOptodeArrays)

		#return None

	# end chOptodeArrays(self, newChOptodeArrays)


	@property
	def optodesOptodeArrays(self):  # optodesOptodeArrays getter
		'''
		An array for storing the identifier of the associated optode array for each optode.

		:getter: Gets the array that stores the identifier of the associated optode array for each optode.
		:setter: Sets the array that stores the identifier of the associated optode array for each optode.
		:type: np.ndarray [nOptodes x 1]
		'''

		return copy.deepcopy(self.__optodesOptodeArrays)

	# end optodesOptodeArrays(self)

	@optodesOptodeArrays.setter
	def optodesOptodeArrays(self, newOptodesOptodeArrays):  # optodesOptodeArrays setter

		# Check parameters
		if type(newOptodesOptodeArrays) is not np.ndarray:
			msg = self.getClassName() + ':optodesOptodeArrays: Unexpected parameter type.'
			raise ValueError(msg)
		if newOptodesOptodeArrays.ndim != 1:
			msg = self.getClassName() + ':optodesOptodeArrays: Unexpected attribute value. ' \
				  + 'The identifier of the associated optode array for each optode must be stored on 1D [nOptodes x 1] array'
			raise ValueError(msg)
		#if newOptodesOptodeArrays.shape[0] != self.nOptodes:
		#	msg = self.getClassName() + ':optodesOptodeArrays: Unexpected attribute value. ' \
		#		  + 'Number of elements must be nOptodes.'
		#	raise ValueError(msg)
		for elem in newOptodesOptodeArrays:
			if type(elem) not in {np.int32, np.int64, np.float64}:   # np.float64 is the type of np.NaN, hence it must be included
				msg = self.getClassName() + ':optodesOptodeArrays: Unexpected parameter value.'
				raise ValueError(msg)
			if elem < 0:
				msg = self.getClassName() + ':optodesOptodeArrays: Unexpected parameter value.'
				raise ValueError(msg)

		self.__optodesOptodeArrays = copy.deepcopy(newOptodesOptodeArrays)

		#return None

	# end optodesOptodeArrays(self, newOptodesOptodeArrays)


	@property
	def pairings(self):  # pairings getter
		'''
		An array for storing the identifiers of the associated optodes conforming each channel.

		:getter: Gets the array that stores the identifiers of the associated optodes conforming each channel.
		:setter: Sets the array that stores the identifiers of the associated optodes conforming each channel.
		:type: np.ndarray [nChannels x 2]
		'''

		return copy.deepcopy(self.__pairings)

	# end pairings(self)

	@pairings.setter
	def pairings(self, newPairings):  # pairings setter

		# Check parameters
		if type(newPairings) is not np.ndarray:
			msg = self.getClassName() + ':pairings: Unexpected parameter type.'
			raise ValueError(msg)
		if newPairings.ndim != 2:
			msg = self.getClassName() + ':pairings: Unexpected attribute value. ' \
				  + 'The identifiers of the associated optodes conforming each channel must be stored on 2D [nChannels x 2] array'
			raise ValueError(msg)
		#if newPairings.shape[0] != self.nChannels and self.nOptodes > 1:
		#	msg = self.getClassName() + ':pairings: Unexpected attribute value. ' \
		#		  + 'Number of rows must be nChannels.'
		#	raise ValueError(msg)
		for elem in newPairings:	# elem is an array of 2 components [a0, a1]
			for i in {0, 1}:
				if type(elem[i]) not in {np.int32, np.int64, np.float64}:  # np.float64 is the type of np.NaN, hence it must be included
					msg = self.getClassName() + ':pairings: Unexpected parameter value.'
					raise ValueError(msg)
				if elem[i] < 0:
					msg = self.getClassName() + ':pairings: Unexpected parameter value.'
					raise ValueError(msg)

		self.__pairings = copy.deepcopy(newPairings)

		#return None

	# end pairings(self, newPairings)


	@property
	def optodeArrays(self):  # optodeArrays getter
		'''
		An array of optodeArrayInfo objects holding the information of the m different optode arrays.

		:getter: Gets the array of optodeArrayInfo objects holding the information of the m different optode arrays.
		:setter: Sets the array of optodeArrayInfo objects holding the information of the m different optode arrays.
		:type: np.ndarray [m x 1]
		'''

		return copy.deepcopy(self.__optodeArrays)

	# end optodeArrays(self)

	@optodeArrays.setter
	def optodeArrays(self, newOptodeArrays):  # optodeArrays setter

		# Check parameters
		if type(newOptodeArrays) is not np.ndarray:
			msg = self.getClassName() + ':optodeArrays: Unexpected parameter type.'
			raise ValueError(msg)
		if newOptodeArrays.ndim != 1:
			msg = self.getClassName() + ':optodeArrays: Unexpected attribute value. ' \
				  + 'The optodeArrayInfo objects holding the information of the m different optode arrays must be stored on 1D [m x 1] array'
			raise ValueError(msg)
		for elem in newOptodeArrays:
			if type(elem) is not optodeArrayInfo:
				msg = self.getClassName() + ':optodeArrays: Unexpected parameter value.'
				raise ValueError(msg)

		self.__optodeArrays = copy.deepcopy(newOptodeArrays)

		#return None

	# end optodeArrays(self, newOptodeArrays)


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

	def showAttributesValues(self):
		'''
		Print the properties information.

		:return: None
		'''

		print("-------------------------------------------------------------------------------------------------------------")

		print("id:          ", self.id)
		print("description: ", self.description)
		print("nChannels:   ", self.nChannels)
		print("nOptodes:    ", self.nOptodes)
		print("")

		print("[chLocations] The ''real-world'' 3D locations of the channels.")
		print("--------------------------------------------------------------")
		print(self.chLocations)
		print("")

		print("[optodesLocations] The ''real-world'' 3D locations of the optodes.")
		print("------------------------------------------------------------------")
		print(self.optodesLocations)
		print("")

		print("[optodesTypes] The type of optodes {0: Unknown, 1: Emission or source, 2: Detector}.")
		print("------------------------------------------------------------------------------------")
		print(self.optodesTypes)
		print("")

		print("[referencePoints] The ''real-world'' 3D locations of the reference points.")
		print("--------------------------------------------------------------------------")
		print(self.referencePoints)
		print("")

		print("surfacePositioningSystem: ", self.surfacePositioningSystem)
		print("")

		print("[chSurfacePositions] The standard surface position for each channel.")
		print("--------------------------------------------------------------------")
		print(self.chSurfacePositions)
		print("")

		print("[optodesSurfacePositions] The standard surface position for each optode.")
		print("------------------------------------------------------------------------")
		print(self.optodesSurfacePositions)
		print("")

		print("[chOptodeArrays] The array for storing the identifier of the associated optode array for each channel.")
		print("------------------------------------------------------------------------------------------------------")
		print(self.chOptodeArrays)
		print("")

		print("[optodesOptodeArrays] The array for storing the identifier of the associated optode array for each optode.")
		print("----------------------------------------------------------------------------------------------------------")
		print(self.optodesOptodeArrays)
		print("")

		print("[pairings] The array for storing the identifiers of the associated optodes conforming each channel.")
		print("---------------------------------------------------------------------------------------------------")
		print(self.pairings)
		print("")

		print("[optodeArrays] The array of optodeArrayInfo objects holding the information of the m different optode arrays.")
		print("-------------------------------------------------------------------------------------------------------------")
		print(self.optodeArrays)
		print("")

		print("-------------------------------------------------------------------------------------------------------------")

	# end showAttributesValues(self)

#class channelLocationMap

def main():
	# Information of the data to be used:
	#	def __init__(self, id = 1, description = 'ChannelLocationMap0001', nChannels = 1, nOptodes  = 1, \
	#				 chLocations = np.array([[np.NaN, np.NaN, np.NaN]]), optodesLocations = np.array([[np.NaN, np.NaN, np.NaN]]), \
	#				 optodesTypes = np.array([np.NaN]), referencePoints = dict(), surfacePositioningSystem = 'UI 10/20', \
	#				 chSurfacePositions = tuple(('',)), optodesSurfacePositions = tuple(('',)), chOptodeArrays = np.array([np.NaN]), \
	#				 optodesOptodeArrays = np.array([np.NaN]), pairings = np.array([[np.NaN, np.NaN]]), \
	#				 optodeArrays = np.array([optodeArrayInfo()])):

	newId 			= 2
	newDescription 	= 'ChannelLocationMap0002'
	newNChannels 	= 4
	newNOptodes  	= 4
	newChLocations		= np.array([[1, 2, 0], [0, 1, 0], [2, 1, 0], [1, 0, 0]])
	newOptodesLocations	= np.array([[0, 2, 0], [2, 2, 0], [0, 0, 0], [2, 0, 0]])
	newOptodesTypes 	= np.array([1, 2, 2, 1])  # Remember {0: Unknown, 1: Emission or source, 2: Detector}
	newReferencePoints	= dict({'Nz': np.array([0, -18.5, 0]), 'Iz': np.array([0, 18.5, 0]),
								  'LPA': np.array([17.5, 0, 0]), 'RPA': np.array([-17.5, 0, 0]), 'Cz': np.array([0, 0, 0]) })
	newSurfacePositioningSystem	= 'UI 10/20'
	newChSurfacePositions 		= tuple(('Fz', 'C3', 'C4', 'Cz'))
	newOptodesSurfacePositions 	= tuple(('FC5', 'CP3', 'FC6', 'CP4'))
	newChOptodeArrays 			= np.array([0, 0, 0, 0])
	newOptodesOptodeArrays 		= np.array([0, 0, 0, 0])
	newPairings = np.array([[0, 1], [0, 2], [3, 1], [3, 2]])

	NewChTopoArrangement  		= np.array([[1, 2, 0], [0, 1, 0], [2, 1, 0], [1, 0, 0]])
	NewOptodesTopoArrangement 	= np.array([[0, 2, 0], [2, 2, 0], [0, 0, 0], [2, 0, 0]])

	oaInfo = optodeArrayInfo(nChannels = newNChannels, nOptodes  = newNOptodes, \
							 mode = 'HITACHI ETG-4000 2x2 optode array', typeOptodeArray = 'adult', \
							 chTopoArrangement = NewChTopoArrangement, \
							 optodesTopoArrangement = NewOptodesTopoArrangement)

	newOptodeArrays = np.array([oaInfo])

	# A channelLocationMap for only EEG signals
	chLM01 = channelLocationMap(nChannels = newNChannels, chLocations = newChLocations, referencePoints = newReferencePoints,
								chSurfacePositions = newChSurfacePositions, chOptodeArrays = newChOptodeArrays)
	chLM01.showAttributesValues()

	# A channelLocationMap for EEG and fNIRS signals
	chLM02 = channelLocationMap(id = newId, description = newDescription, nChannels = newNChannels, nOptodes  = newNOptodes,
								chLocations = newChLocations, optodesLocations	= newOptodesLocations,
								optodesTypes = newOptodesTypes, referencePoints = newReferencePoints,
								surfacePositioningSystem = newSurfacePositioningSystem,
								chSurfacePositions = newChSurfacePositions, optodesSurfacePositions = newOptodesSurfacePositions,
								chOptodeArrays = newChOptodeArrays, optodesOptodeArrays = newOptodesOptodeArrays,
								pairings = newPairings, optodeArrays = newOptodeArrays)
	chLM02.showAttributesValues()

#end main()


if __name__ == '__main__':
	main()


