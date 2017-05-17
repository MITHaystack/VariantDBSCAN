# The MIT License (MIT)
# Copyright (c) 2017 Massachusetts Institute of Technology
#
# Author: Cody Rude and Michael Gowanlock
# This software has been created in projects supported by the US National
# Science Foundation and NASA (PI: Pankratius, NSF ACI-1442997, NASA AIST-NNX15AG84G)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import numpy as np
import numpy.ctypeslib as npct
from ctypes import c_int, c_uint, c_double, c_bool
import os 



def VariantDBSCAN(eps_array,mp_array, x_array, y_array, mbbsize = 70, verbose=False):
    ''' 
    Runs variant DBScan on input arrays

    @param eps_array: numpy array containing epsilon values, must be same length as mp_array
    @param mp_array: numpy array containing minpoint values, must be same length as eps_array
    @param x_array: numpy array of x positions
    @param y_array: numpy array of y positions

    @return: 2-d array yielding cluster id for coordinate for each variant
    '''

    # Get library path
    # Use the following line if installed by pip
    lib_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'lib')
    # Otherwise, comment out the above line and 
    # uncomment the following line, adjusting path as necessary
    # lib_path = '/usr/local/lib/' 
    
    # Load variant dbscan shared library
    libvdbscan = npct.load_library('libSharedVDBSCAN', lib_path)

    # Create variables that define C interface
    array_1d_double = npct.ndpointer(dtype=c_double, ndim=1, flags='CONTIGUOUS')
    array_1d_unsigned = npct.ndpointer(dtype=c_uint, ndim=1, flags='CONTIGUOUS')


    # Set return and input variables for C interface. The interface
    # for the function call is as follows:
    # int libVDBSCAN(double * inputx, double * inputy, unsigned int
    # datasetSize, double * inputEpsilon, unsigned int * inputMinpts,
    # unsigned int numVariants, int MBBsize, unsigned int * retArr,
    # bool verbose);

    libvdbscan.libVDBSCAN.restype = c_int
    libvdbscan.libVDBSCAN.argtypes = [array_1d_double, array_1d_double, c_uint,
                                      array_1d_double, array_1d_unsigned, c_uint,
                                      c_int, array_1d_unsigned, c_bool]


    # This function converts an input numpy array into a different
    # data type and ensure that it is contigious.  
    def convert_type(in_array, new_dtype):

        ret_array = in_array
        
        if not isinstance(in_array, np.ndarray):
            ret_array = np.array(in_array, dtype=new_dtype)
        
        elif in_array.dtype != new_dtype:
            ret_array = np.array(ret_array, dtype=new_dtype)

        if ret_array.flags['C_CONTIGUOUS'] == False:
            ret_array = np.ascontiguousarray(ret_array)

        return ret_array

    # Convert all the input arrays to apprioprate type
    # required by the c function call
    c_eps_array = convert_type(eps_array, c_double)
    c_mp_array = convert_type(mp_array, c_uint)

    c_x_array = convert_type(x_array, c_double)
    c_y_array = convert_type(y_array, c_double)

    # Allocate array for results from function call
    results = np.zeros(len(x_array) * len(eps_array), dtype=c_uint)    

    # Call Variant DBScan
    res = libvdbscan.libVDBSCAN(c_x_array, c_y_array, len(c_x_array), c_eps_array, c_mp_array,
                                len(c_eps_array), mbbsize, results, verbose)

    # libVDBSCAN returns a 1d array.  If more than 1 epsilon value is
    # specified, reshape results into a 2d array
    if len(eps_array) > 1:
        results = results.reshape([len(eps_array), len(x_array)]).T

    return results
