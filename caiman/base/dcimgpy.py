# DCIMG reader interface for Python
# dcimgpy.py: 


import ctypes
from ctypes import c_int32, c_uint32, c_void_p, c_char_p
from ctypes import Structure, POINTER, sizeof, byref
import numpy as np
import caiman.base.dcimgapi as dcimg

class Dcimg:
	"""
	class for instance to access DCIMG file.
	"""
	# New instance
	def __init__(self, filepath = None):
		self._hdcimg = c_void_p(None)
		self._bOpened = False
		self._image_width = 0
		self._image_height = 0
		self._image_pixeltype = 0
		self._numberof_frame = 0
		self._dcimg_path = None
		self._n = 0

		if(filepath != None):
			b_filepath = filepath.encode('utf-8')
			openparam = dcimg.DCIMG_OPEN()
			openparam.path = b_filepath
			retval = dcimg.open(openparam)
			if retval != dcimg.DCIMG_ERR.SUCCESS:
				return

			self._hdcimg = openparam.hdcimg
			self._bOpened = True
			self._image_width = self.getParaml(dcimg.DCIMG_IDPARAML.IMAGE_WIDTH)
			self._image_height = self.getParaml(dcimg.DCIMG_IDPARAML.IMAGE_HEIGHT)
			self._image_pixeltype = self.getParaml(dcimg.DCIMG_IDPARAML.IMAGE_PIXELTYPE)
			self._numberof_frame = self.getParaml(dcimg.DCIMG_IDPARAML.NUMBEROF_FRAME)
			self._dcimg_path = filepath
		return

	def __len__(self):
		if(self._bOpened == True):
			return self._numberof_frame
		else:
			return 0

	def __repr__(self):
		return 'Dcimg({!r})'.format(self._dcimg_path)

	def __str__(self):
		return 'Dcimg class of {!r}'.format(self._dcimg_path)
	"""
	def __iter__(self):
		self._n = 0
		return self

	def __next__(self):
		if self._n < self._numberof_frame:
			result = self._n
			self._n += 1
			return result
		else:
			return StopIteration
	"""
	def open(self, filepath):
		"""
		open DCIMG file.

		Args:
			arg1 (str): file path and name for DCIMG.

		Returns:
			bool: True means DCIMG file was opened successfully.
				If False is returned, dcimg.__lasterr member indicates error.
		"""
		# close previous handle
		if self._bOpened:
			dcimg.close(self._hdcimg)

		b_filepath = filepath.encode('utf-8')
		openparam = dcimg.DCIMG_OPEN()
		openparam.path = b_filepath
		retval = dcimg.open(openparam)
		if retval != dcimg.DCIMG_ERR.SUCCESS:
			return False

		self._hdcimg = openparam.hdcimg
		self._bOpened = True
		self._image_width = self.getParaml(dcimg.DCIMG_IDPARAML.IMAGE_WIDTH)
		self._image_height = self.getParaml(dcimg.DCIMG_IDPARAML.IMAGE_HEIGHT)
		self._image_pixeltype = self.getParaml(dcimg.DCIMG_IDPARAML.IMAGE_PIXELTYPE)
		self._numberof_frame = self.getParaml(dcimg.DCIMG_IDPARAML.NUMBEROF_FRAME)
		self._dcimg_path = filepath
		return True

	def close(self):
		""" Close the file """
		if self._bOpened:
			dcimg.close(self._hdcimg)
		return True

	def getImageWidth(self):
		"""	get image width of the frames in the DCIMG file	"""
		return self._image_width
		
	def getImageHeight(self):
		"""	get image height of the frames in the DCIMG file """
		return self._image_height
		
	def getImagePixelType(self):
		"""	get pixel type of the frames in DCIMG file	"""
		return self._image_pixeltype

	def getTotalFrames(self):
		"""	get total frames in DCIMG file """
		return self._numberof_frame

	def getFilePath(self):
		"""	get file path of the DCIMG file """
		return self._dcimg_path


	def getParaml(self, idparaml):
		"""
		get information of DCIMG file

		Args:
			arg1 (DCIMG_IDPARAML): parameter id to get value

		Returns:
			int: value of parameter id.
				Return value can be 0 when calling is failed.
				To know the last calling status, check __lasterr member
				or call dcimg.failed() function.
		"""
		v = c_int32(0)
		if self._bOpened:
			self._lasterr = dcimg.getparaml(self._hdcimg, idparaml.value,	byref(v))
		return v.value


	def getImageData(self, frame):
		"""
		read a frame of DCIMG file

		Args:
			arg1 (int): frame index

		Returns:
			numpy.ndarray: image data if succeeded.
						If failed, None is returned.
		"""
		if (self._bOpened == False):
			return None
			
		frameparam = dcimg.DCIMG_FRAME()
		frameparam.iFrame = frame
		retval = dcimg.lockframe(self._hdcimg, frameparam)
		if retval != dcimg.DCIMG_ERR.SUCCESS:
			return None

		data_pointer = ctypes.cast(frameparam.buf, ctypes.POINTER(ctypes.c_int))
		buffrommem = ctypes.pythonapi.PyMemoryView_FromMemory
		buffrommem.restype = ctypes.py_object
		buffer = buffrommem(data_pointer, frameparam.width*frameparam.height*2, 0x100)
		data_array = np.frombuffer(buffer, dtype=np.uint16)
		return np.reshape(data_array, (-1, frameparam.width))

	def getImageTimestamp(self, frame):
		"""	get Timestamp value of the specified frame in milliseconds """
		if (self._bOpened == False):
			return None
			
		frameparam = dcimg.DCIMG_FRAME()
		frameparam.iFrame = frame
		retval = dcimg.lockframe(self._hdcimg, frameparam)
		if retval != dcimg.DCIMG_ERR.SUCCESS:
			return None

		sec = frameparam.timestamp.sec
		msec = frameparam.timestamp.microsec
		return (sec*1000000) + msec

