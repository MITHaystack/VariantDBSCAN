//The MIT License (MIT)
//Copyright (c) 2016 Massachusetts Institute of Technology

//Authors: Mike Gowanlock
//This software has been created in projects supported by the US National
//Science Foundation and NASA (PI: Pankratius, NSF ACI-1442997, NASA AIST-NNX15AG84G)


//Permission is hereby granted, free of charge, to any person obtaining a copy
//of this software and associated documentation files (the "Software"), to deal
//in the Software without restriction, including without limitation the rights
//to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//copies of the Software, and to permit persons to whom the Software is
//furnished to do so, subject to the following conditions:
//The above copyright notice and this permission notice shall be included in
//all copies or substantial portions of the Software.
//THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
//THE SOFTWARE.

#include "structs.h"
__global__ void kernelBruteForce(unsigned int *N, unsigned int *debug1, unsigned int *debug2, double *epsilon, 
	unsigned int * cnt, struct point * database, int * pointIDKey, int * pointInDistVal);

//used for both batched and non-batched executions
__global__ void kernelGridIndex(unsigned int *N, unsigned int *offset, unsigned int *batchNum, unsigned int *debug1, unsigned int *debug2, double *epsilon,  
	struct grid * index, double * gridMin_x, double * gridMin_y, int * gridNumXCells, int * gridNumYCells, int * lookupArr, 
	unsigned int * cnt, struct point * database, int * pointIDKey, int * pointInDistVal);

//DEPRECATED-- now is just kernelgridindex
// __global__ void kernelGridIndexKeyVal(unsigned int *N, unsigned int *debug1, unsigned int *debug2, double *epsilon, 
// 	struct grid * index, double * gridMin_x, double * gridMin_y, int * gridNumXCells, int * gridNumYCells, int * lookupArr, 
// 	unsigned int * cnt, struct point * database, int * pointIDKey, int * pointInDistVal);

__global__ void kernelGridIndexSMBlock(unsigned int *numthreads, unsigned int *N, unsigned int *debug1, unsigned int *debug2, double *epsilon, 
	struct grid * index, double * gridMin_x, double * gridMin_y, int * gridNumXCells, int * gridNumYCells, int * lookupArr, 
	unsigned int * cnt, struct point * database, unsigned int * schedule, int * pointIDKey, int * pointInDistVal);

__global__ void kernelGridIndexSMBlockDataAware(unsigned int *numthreads, unsigned int *N, unsigned int *debug1, unsigned int *debug2, double *epsilon, 
	struct grid * index, double * gridMin_x, double * gridMin_y, int * gridNumXCells, int * gridNumYCells, int * lookupArr, 
	unsigned int * cnt, struct point * database, const unsigned int * sharedMemElemSize, int * pointIDKey, int * pointInDistVal);


__global__ void calcNeighborsFromTableKernel(unsigned int *N, struct gpulookuptable * lookup, int * directNeighborArray, unsigned int * cnt, double * epsilon, struct point * database, int * pointIDKey, int * pointInDistVal);
__global__ void calcNeighborsFromTableKernelBatches(unsigned int *N, unsigned int *offset, unsigned int *batchNum, struct gpulookuptable * lookup, int * directNeighborArray, unsigned int * cnt, double * epsilon, struct point * database, int * pointIDKey, int * pointInDistVal);

__global__ void testkernel(unsigned int * cnt);


//used to estimate the number of batches for the batched implementation
__global__ void kernelGridIndexBatchEstimator(unsigned int *N, unsigned int *sampleOffset, unsigned int *debug1, unsigned int *debug2, double *epsilon, struct grid * index, double * gridMin_x, double * gridMin_y, int * gridNumXCells, int * gridNumYCells, int * lookupArr, unsigned int * cnt, struct point * database);