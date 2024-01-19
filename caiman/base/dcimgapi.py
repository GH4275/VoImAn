# direct DCIMGAPI wrapper for Python
# dcimgapi.py: 

"""
module for accessing DCIMG files via DCIMG-API.
"""

import platform
from enum import IntEnum
import ctypes
from ctypes import c_int32, c_uint32, c_void_p, c_char_p
from ctypes import Structure, POINTER, sizeof, byref


# ==== load shared library ====

# abosorb platform dependency

from ctypes import cdll
__dll = cdll.LoadLibrary('dcimgapi')

# ==== declare constants ====


class DCIMG_ERR(IntEnum):
    NOMEMORY 					= 0x80000203    # not enough memory
    INVALIDHANDLE 				= 0x80000807    # invalid dcimg value
    INVALIDPARAM 				= 0x80000808    # invalid parameter, e.g. parameter is NULL
    INVALIDVALUE 				= 0x80000821    # invalid parameter value
    INVALIDVIEW 				= 0x8000082a    # invalid view index
    INVALIDFRAMEINDEX 			= 0x80000833    # the frame index is invalid
    FILENOTOPENED 				= 0x80000835    # file is not opened at dcimg_open()
    UNKNOWNFILEFORMAT 			= 0x80000836    # opened file format is not supported
    NOTSUPPORT 					= 0x80000f03    # the function or property are not supportted
    FAILEDREADDATA 				= 0x84001004
    UNKNOWNSIGN 				= 0x84001801
    OLDERFILEVERSION 			= 0x84001802
    NEWERFILEVERSION 			= 0x84001803
    NOIMAGE 					= 0x84001804
    UNKNOWNIMAGEPROC 			= 0x84001805
    NOTSUPPORTIMAGEPROC 		= 0x84001806
    NODATA 						= 0x84001807
    IMAGE_UNKNOWNSIGNATURE 		= 0x84003001 # sigunature of image header is unknown or corrupted
    IMAGE_NEWRUNTIMEREQUIRED 	= 0x84003002 # the dcimg file uses new format so newer DCIMG runtime is necessary
    IMAGE_ERRORSTATUSEXIST 		= 0x84003003    # image header stands error status
    IMAGE_HEADERCORRUPTED 		= 0x84004004    # image header value is strange
    UNKNOWNCOMMAND 				= 0x80000801    # unknown command id
    UNKNOWNPARAMID 				= 0x80000803    # unkown parameter id
    UNREACH 					= 0x80000f01    # internal error
    INVALIDCODEPAGE 			= 0x8DC10001    # invalid DCIMG_OPEN::codepage
    SUCCESS 					= 1    # no error, general success code


class DCIMG_IDPARAML(IntEnum):
	NUMBEROF_TOTALFRAME 		= 0    # number of total frame in the file
	NUMBEROF_FRAME 				= 2    # number of frame
	SIZEOF_USERDATABIN_FILE 	= 5    # byte size of file binary USER META DATA.
	SIZEOF_USERDATATEXT_FILE 	= 8    # byte size of file text USER META DATA.
	IMAGE_WIDTH 				= 9    # image width
	IMAGE_HEIGHT 				= 10    # image height
	IMAGE_ROWBYTES 				= 11    # image rowbytes
	IMAGE_PIXELTYPE 			= 12    # image pixeltype
	MAXSIZE_USERDATABIN 		= 13    # maximum byte size of binary USER META DATA
	MAXSIZE_USERDATATEXT 		= 16    # maximum byte size of text USER META DATA
	NUMBEROF_VIEW 				= 20    # number of view
	FILEFORMAT_VERSION 			= 21    # file format version
	CAPABILITY_IMAGEPROC 		= 22    # capability of image processing


class DCIMG_PIXELTYPE(IntEnum):
	NONE 		= 0    # no pixeltype specified
	MONO8 		= 1    # B/W 8 bit
	MONO16 		= 2    # B/W 16 bit


class DCIMG_CODEPAGE(IntEnum):
	SHIFT_JIS 	= 932    # Shift JIS
	UTF16_LE 	= 1200    # UTF-16 (Little Endian)
	UTF16_BE 	= 1201    # UTF-16 (Big Endian)
	UTF7 		= 65000    # UTF-7 translation
	UTF8 		= 65001    # UTF-8 translation


# ==== declare structures for DCIMG-API functions ====


class DCIMG_INIT(Structure):
	_pack_ = 8
	_fields_ = [
		("size", c_int32),
		("reserved", c_int32),
		("guid", c_void_p)
	]

	def __init__(self):
		self.size = sizeof(DCIMG_INIT)


class DCIMG_OPEN(Structure):
	_pack_ = 8
	_fields_ = [
		("size", c_int32),
		("codepage", c_int32),
		("hdcimg", c_void_p),
		("path", c_char_p)
	]

	def __init__(self):
		self.size = sizeof(DCIMG_OPEN)


class DCIMG_TIMESTAMP(Structure):
	_pack_ = 8
	_fields_ = [
		("sec", c_uint32),
		("microsec", c_int32)
	]

	def __init__(self):
		self.sec = 0
		self.microsec = 0


class DCIMG_FRAME(Structure):
	_pack_ = 8
	_fields_ = [
		("size", c_int32),
		("iKind", c_int32),
		("option", c_int32),
		("iFrame", c_int32),
		("buf", c_void_p),
		("rowbytes", c_int32),
		("type", c_int32),    # DCIMG_PIXELTYPE
		("width", c_int32),
		("height", c_int32),
		("left", c_int32),
		("top", c_int32),
		("timestamp", DCIMG_TIMESTAMP),
		("framestamp", c_int32),
		("camerastamp", c_int32)
	]

	def __init__(self):
		self.size = sizeof(DCIMG_FRAME)
		self.iKind = 0
		self.option = 0
		self.iFrame = 0
		self.buf = 0
		self.rowbytes = 0
		self.type = DCIMG_PIXELTYPE.MONO16
		self.width = 0
		self.height = 0
		self.left = 0
		self.top = 0

def open(openparam):
	__platform_system = platform.system()
	if __platform_system == 'Windows':
		retval = __dll.dcimg_openA(byref(openparam))
	else:
		retval = __dll.dcimg_open(byref(openparam))
	return retval

def close(hdcimg):
	retval = __dll.dcimg_close(ctypes.c_void_p(hdcimg))
	return retval

def lockframe(hdcimg, frame):
	retval = __dll.dcimg_lockframe(ctypes.c_void_p(hdcimg), byref(frame))
	return retval

def copyframe(hdcimg, frame):
	retval = __dll.dcimg_copyframe(ctypes.c_void_p(hdcimg), byref(frame))
	return retval

def getparaml(hdcimg, index, paraml):
	retval = __dll.dcimg_getparaml(ctypes.c_void_p(hdcimg), index, paraml)
	return retval


