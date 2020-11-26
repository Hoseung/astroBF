#
# Written by Min-Su Shin
# Astrophysics, Department of Physics, University of Oxford (2012 - )
# Department of Astronomy, University of Michigan (2009 - 2012)
# Department of Astrophysical Sciences, Princeton University (2005 - 2009)
#
# You can freely use the code.
#

import numpy as np
import math

def sky_sig_clip(input_arr, sig_fract, percent_fract, max_iter=100, operator='median', low_cut=True, high_cut=True):
	"""Estimating a sky value for a given number of iterations

	@type input_arr: np array
	@param input_arr: image data array
	@type sig_fract: float
	@param sig_fract: fraction of sigma clipping
	@type percent_fract: float
	@param percent_fract: convergence fraction
	@type max_iter: integer
	@param max_iter: max. of iterations
	@type low_cut: boolean
	@param low_cut: cut out only low values
	@type high_cut: boolean
	@param high_cut: cut out only high values
	@rtype: tuple
	@return: (sky value, number of iterations)

	"""
	if operator == 'median':
		op = np.median
	elif operator == "mean":
		op = np.mean
	
	work_arr = np.ravel(input_arr)
	old_sky = np.median(work_arr)
	sig = work_arr.std()
	upper_limit = old_sky + sig_fract * sig
	lower_limit = old_sky - sig_fract * sig
	if low_cut and high_cut:
		indices = np.where((work_arr < upper_limit) & (work_arr > lower_limit))
	else:
		if low_cut:
			indices = np.where((work_arr > lower_limit))
		else:
			indices = np.where((work_arr < upper_limit))
	work_arr = work_arr[indices]
	new_sky = op(work_arr)
	iteration = 0
	while ((math.fabs(old_sky - new_sky)/new_sky) > percent_fract) and (iteration < max_iter) :
		iteration += 1
		old_sky = new_sky
		sig = work_arr.std()
		upper_limit = old_sky + sig_fract * sig
		lower_limit = old_sky - sig_fract * sig
		if low_cut and high_cut:
			indices = np.where((work_arr < upper_limit) & (work_arr > lower_limit))
		else:
			if low_cut:
				indices = np.where((work_arr > lower_limit))
			else:
				indices = np.where((work_arr < upper_limit))
		work_arr = work_arr[indices]
		new_sky = op(work_arr)
	return (new_sky, iteration)


def sky_median_sig_clip(input_arr, sig_fract, percent_fract, max_iter=100, low_cut=True, high_cut=True):
	return sky_sig_clip(input_arr, sig_fract, percent_fract,
	 		max_iter=max_iter, operator='median', low_cut=low_cut, high_cut=high_cut)


def sky_mean_sig_clip(input_arr, sig_fract, percent_fract, max_iter=100, low_cut=True, high_cut=True):
	return sky_sig_clip(input_arr, sig_fract, percent_fract,
	 		max_iter=max_iter, operator='mean', low_cut=low_cut, high_cut=high_cut)



def range_from_zscale(input_arr, contrast = 1.0, sig_fract = 3.0, percent_fract = 0.01, max_iter=100, low_cut=True, high_cut=True):
	"""Estimating ranges with the zscale algorithm

	@type input_arr: np array
	@param input_arr: image data array as sample pixels to derive z-ranges
	@type contrast: float
	@param contrast: zscale contrast which should be larger than 0.
	@type sig_fract: float
	@param sig_fract: fraction of sigma clipping
	@type percent_fract: float
	@param percent_fract: convergence fraction
	@type max_iter: integer
	@param max_iter: max. of iterations
	@type low_cut: boolean
	@param low_cut: cut out only low values
	@type high_cut: boolean
	@param high_cut: cut out only high values
	@rtype: tuple
	@return: (min. value, max. value, number of iterations)

	"""
	work_arr = np.ravel(input_arr)
	work_arr = np.sort(work_arr) # sorting is done.
	max_ind = len(work_arr) - 1
	midpoint_ind = int(len(work_arr)*0.5)
	I_midpoint = work_arr[midpoint_ind]
	print(".. midpoint index ", midpoint_ind, " I_midpoint ", I_midpoint)
	# initial estimation of the slope
	x = np.array(range(0, len(work_arr))) - midpoint_ind
	y = np.array(work_arr)
	temp = np.vstack([x, np.ones(len(x))]).T
	slope, intercept = np.linalg.lstsq(temp, y)[0]
	old_slope = slope
	print("... slope & intercept ", old_slope, " ", intercept)
	# initial clipping
	sig = y.std()
	upper_limit = I_midpoint + sig_fract * sig
	lower_limit = I_midpoint - sig_fract * sig
	if low_cut and high_cut:
		indices = np.where((work_arr < upper_limit) & (work_arr > lower_limit))
	else:
		if low_cut:
			indices = np.where((work_arr > lower_limit))
		else:
			indices = np.where((work_arr < upper_limit))
	# new estimation of the slope
	x = np.array(indices[0]) - midpoint_ind
	y = np.array(work_arr[indices])
	temp = np.vstack([x, np.ones(len(x))]).T
	slope, intercept = np.linalg.lstsq(temp, y)[0]
	new_slope = slope
	print("... slope & intercept ", new_slope, " ", intercept)
	iteration = 1
	# to run the iteration, we need more than 50% of the original input array
	while (((math.fabs(old_slope - new_slope)/new_slope) > percent_fract) and (iteration < max_iter)) and (len(y) >= midpoint_ind) :
		iteration += 1
		old_slope = new_slope
		# clipping
		sig = y.std()
		upper_limit = I_midpoint + sig_fract * sig
		lower_limit = I_midpoint - sig_fract * sig
		if low_cut and high_cut:
			indices = np.where((work_arr < upper_limit) & (work_arr > lower_limit))
		else:
			if low_cut:
				indices = np.where((work_arr > lower_limit))
			else:
				indices = np.where((work_arr < upper_limit))
		# new estimation of the slope
		x = np.array(indices[0]) - midpoint_ind
		y = work_arr[indices]
		temp = np.vstack([x, np.ones(len(x))]).T
		slope, intercept = np.linalg.lstsq(temp, y)[0]
		new_slope = slope
		print("... slope & intercept ", new_slope, " ", intercept)

	z1 = I_midpoint + (new_slope / contrast) * (0 - midpoint_ind)
	z2 = I_midpoint + (new_slope / contrast) * (max_ind - midpoint_ind)

	return (z1, z2, iteration)



def range_from_percentile(input_arr, low_cut=0.25, high_cut=0.25):
	"""Estimating ranges with given percentiles

	@type input_arr: np array
	@param input_arr: image data array as sample pixels to derive ranges
	@type low_cut: float
	@param low_cut: cut of low-value pixels
	@type high_cut: float
	@param high_cut: cut of high-value pixels
	@rtype: tuple
	@return: (min. value, max. value)

	"""
	work_arr = np.ravel(input_arr)
	work_arr = np.sort(work_arr) # sorting is done.
	size_arr = len(work_arr)
	low_size = int(size_arr * low_cut)
	high_size = int(size_arr * high_cut)
	
	z1 = work_arr[low_size]
	z2 = work_arr[size_arr - 1 - high_size]

	return (z1, z2)



def histeq(inputArray, scale_min=None, scale_max=None, num_bins=512):
	"""Performs histogram equalisation of the input np array.
    
	@type inputArray: np array
	@param inputArray: image data array
	@type scale_min: float
	@param scale_min: minimum data value
	@type scale_max: float
	@param scale_max: maximum data value
	@type num_bins: int
	@param num_bins: number of bins in which to perform the operation (e.g. 512)
	@rtype: np array
	@return: image data array
    
	"""		
    
	imageData=np.array(inputArray, copy=True)
	if scale_min == None:
		scale_min = imageData.min()
	if scale_max == None:
		scale_max = imageData.max()
	imageData.clip(min=scale_min, max=scale_max)
	imageData = (imageData -scale_min) / (scale_max - scale_min) # now between 0 and 1.
	indices = np.where(imageData < 0)
	imageData[indices] = 0.0
    
	# histogram equalisation: we want an equal number of pixels in each intensity range
	image_histogram, histogram_bins = np.histogram(imageData.flatten(), bins=num_bins, range=(0.0, 1.0), density=True)
	histogram_cdf = image_histogram.cumsum()
	histogram_cdf = histogram_cdf / histogram_cdf[-1] # normalization

	# mapping the image values to the histogram bins
	imageData_temp = np.interp(imageData.flatten(), histogram_bins[:-1], histogram_cdf)
	imageData = imageData_temp.reshape(imageData.shape)
       
	return imageData



def linear(inputArray, scale_min=None, scale_max=None):
	"""Performs linear scaling of the input np array.

	@type inputArray: np array
	@param inputArray: image data array
	@type scale_min: float
	@param scale_min: minimum data value
	@type scale_max: float
	@param scale_max: maximum data value
	@rtype: np array
	@return: image data array
	
	"""		
	print("img_scale : linear")
	imageData=np.array(inputArray, copy=True)
	
	if scale_min == None:
		scale_min = imageData.min()
	if scale_max == None:
		scale_max = imageData.max()

	imageData.clip(min=scale_min, max=scale_max)
	imageData = (imageData -scale_min) / (scale_max - scale_min)
	indices = np.where(imageData < 0)
	imageData[indices] = 0.0
	
	return imageData


def sqrt(inputArray, scale_min=None, scale_max=None):
	"""Performs sqrt scaling of the input np array.

	@type inputArray: np array
	@param inputArray: image data array
	@type scale_min: float
	@param scale_min: minimum data value
	@type scale_max: float
	@param scale_max: maximum data value
	@rtype: np array
	@return: image data array
	
	"""		
    
	print("img_scale : sqrt")
	imageData=np.array(inputArray, copy=True)
	
	if scale_min == None:
		scale_min = imageData.min()
	if scale_max == None:
		scale_max = imageData.max()

	imageData.clip(min=scale_min, max=scale_max)
	imageData = imageData - scale_min
	indices = np.where(imageData < 0)
	imageData[indices] = 0.0
	imageData = np.sqrt(imageData)
	imageData = imageData / math.sqrt(scale_max - scale_min)
	
	return imageData


def log(inputArray, scale_min=None, scale_max=None):
	"""Performs log10 scaling of the input np array.

	@type inputArray: np array
	@param inputArray: image data array
	@type scale_min: float
	@param scale_min: minimum data value
	@type scale_max: float
	@param scale_max: maximum data value
	@rtype: np array
	@return: image data array
	
	"""		
    
	print("img_scale : log")
	imageData=np.array(inputArray, copy=True)
	
	if scale_min == None:
		scale_min = imageData.min()
	if scale_max == None:
		scale_max = imageData.max()
	factor = math.log10(scale_max - scale_min)
	indices0 = np.where(imageData < scale_min)
	indices1 = np.where((imageData >= scale_min) & (imageData <= scale_max))
	indices2 = np.where(imageData > scale_max)
	imageData[indices0] = 0.0
	imageData[indices2] = 1.0
	try :
		imageData[indices1] = np.log10(imageData[indices1])/factor
	except :
		print("Error on math.log10 for ", (imageData[i][j] - scale_min))

	return imageData


def power(inputArray, power_index=3.0, scale_min=None, scale_max=None):
	"""Performs power scaling of the input np array.

	@type inputArray: np array
	@param inputArray: image data array
	@type power_index: float
	@param power_index: power index
	@type scale_min: float
	@param scale_min: minimum data value
	@type scale_max: float
	@param scale_max: maximum data value
	@rtype: np array
	@return: image data array
	
	"""		
    
	print("img_scale : power")
	imageData=np.array(inputArray, copy=True)
	
	if scale_min == None:
		scale_min = imageData.min()
	if scale_max == None:
		scale_max = imageData.max()
	factor = 1.0 / math.pow((scale_max - scale_min), power_index)
	indices0 = np.where(imageData < scale_min)
	indices1 = np.where((imageData >= scale_min) & (imageData <= scale_max))
	indices2 = np.where(imageData > scale_max)
	imageData[indices0] = 0.0
	imageData[indices2] = 1.0
	imageData[indices1] = np.power((imageData[indices1] - scale_min), power_index)*factor

	return imageData


def asinh(inputArray, scale_min=None, scale_max=None, non_linear=2.0):
	"""Performs asinh scaling of the input np array.

	@type inputArray: np array
	@param inputArray: image data array
	@type scale_min: float
	@param scale_min: minimum data value
	@type scale_max: float
	@param scale_max: maximum data value
	@type non_linear: float
	@param non_linear: non-linearity factor
	@rtype: np array
	@return: image data array
	
	"""		
    
	print("img_scale : asinh")
	imageData=np.array(inputArray, copy=True)
	
	if scale_min == None:
		scale_min = imageData.min()
	if scale_max == None:
		scale_max = imageData.max()
	factor = np.arcsinh((scale_max - scale_min)/non_linear)
	indices0 = np.where(imageData < scale_min)
	indices1 = np.where((imageData >= scale_min) & (imageData <= scale_max))
	indices2 = np.where(imageData > scale_max)
	imageData[indices0] = 0.0
	imageData[indices2] = 1.0
	imageData[indices1] = np.arcsinh((imageData[indices1] - scale_min)/non_linear)/factor

	return imageData


def logistic(inputArray, scale_min=None, scale_max=None, center=0.5, slope=1.0):
	"""Performs logistic scaling of the input np array.

	@type inputArray: np array
	@param inputArray: image data array
	@type scale_min: float
	@param scale_min: minimum data value
	@type scale_max: float
	@param scale_max: maximum data value
	@type center: float
	@param center: central value
	@type slope: float
	@param slope: slope
	@rtype: np array
	@return: image data array
	
	"""		
    
	print("img_scale : logistic")
	imageData=np.array(inputArray, copy=True)
	
	if scale_min == None:
		scale_min = imageData.min()
	if scale_max == None:
		scale_max = imageData.max()
	factor2 = 1.0/(1.0+1.0/math.exp((scale_max - center)/slope))
	factor2 = factor2 + 1.0/(1.0+1.0/math.exp((scale_min - center)/slope))
	factor2 = 1.0 / factor2
	factor1 = -1.0 * factor2 / (1.0+1.0/math.exp((scale_min - center)/slope))
	indices0 = np.where(imageData < scale_min)
	indices1 = np.where((imageData >= scale_min) & (imageData <= scale_max))
	indices2 = np.where(imageData > scale_max)
	imageData[indices0] = 0.0
	imageData[indices2] = 1.0
	imageData[indices1] = factor1 + factor2 / (1.0 + 1.0/np.exp((imageData[indices1] - center)/slope))

	return imageData
