//The MIT License (MIT)
//Copyright (c) 2016 Massachusetts Institute of Technology

//Authors: Mike Gowanlock
//This software has been created in projects supported by the US National
//Science Foundation and NASA (PI: Pankratius)


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

//function prototypes:
void copyDatabaseToGPU(std::vector<struct dataElem> * dataPoints, struct point * dev_database);

bool compResults(structresults const& lhs, structresults const& rhs);

void setKernelParams(unsigned int * dev_N, unsigned int * N, unsigned int  * dev_debug1, unsigned int  * dev_debug2, unsigned int *dev_cnt, double * dev_epsilon, double * epsilon);

void makeDistanceTableGPUBruteForce(std::vector<struct dataElem> * dataPoints, double * epsilon, struct table * neighborTable, int * totalNeighbors);


																												
void makeDistanceTableGPUGridIndex(std::vector<struct dataElem> * dataPoints, double * epsilon, struct grid * index, double * gridMin_x, double * gridMin_y, int * gridNumXCells, int * gridNumYCells, unsigned int * lookupArr, struct table * neighborTable, int * totalNeighbors);
void makeDistanceTableGPUGridIndexWithSMBlockDataOblivious(std::vector<struct dataElem> * dataPoints, double * epsilon, struct grid * index, int * numNonEmptyCells, double * gridMin_x, double * gridMin_y, int * gridNumXCells, int * gridNumYCells, unsigned int * lookupArr, struct table * neighborTable, int * totalNeighbors);
void makeDistanceTableGPUGridIndexWithSMBlockDataAware(std::vector<struct dataElem> * dataPoints, double * epsilon, struct grid * index, double * gridMin_x, double * gridMin_y, int * gridNumXCells, int * gridNumYCells, int * lookupArr, struct table * neighborTable, int * totalNeighbors, unsigned int maxNumSMDataItems);

void makeDistanceTableGPUGridIndexBatches(std::vector<struct dataElem> * dataPoints, double * epsilon, struct grid * index, double * gridMin_x, double * gridMin_y, int * gridNumXCells, int * gridNumYCells, unsigned int * lookupArr, struct table * neighborTable, unsigned int * totalNeighbors);



//void makeDistanceTableGPUGridIndexBatchesAlternateTest(std::vector<struct dataElem> * dataPoints, double * epsilon, struct grid * index, double * gridMin_x, double * gridMin_y, int * gridNumXCells, int * gridNumYCells, unsigned int * lookupArr, struct neighborTableLookup * neighborTable, std::vector<int *> * pointersToNeighbors, unsigned int * totalNeighbors);
void makeDistanceTableGPUGridIndexBatchesAlternateTest(std::vector<struct dataElem> * dataPoints, double * epsilon, struct grid * index, double * gridMin_x, double * gridMin_y, int * gridNumXCells, int * gridNumYCells, unsigned int * lookupArr, struct neighborTableLookup * neighborTable, std::vector<struct neighborDataPtrs> * pointersToNeighbors, unsigned int * totalNeighbors);


void constructNeighborTableKeyValueAlternateTest(int * pointIDKey, int * pointInDistValue, struct neighborTableLookup * neighborTable, int * pointersToNeighbors, unsigned int * cnt);
//void constructNeighborTableKeyValueAlternateTest(int * pointIDKey, int * pointInDistValue, struct neighborTableLookup * neighborTable, struct neighborDataPtrs * pointersToNeighbors, unsigned int * cnt);

void generateNeighborArrayForGPUAlternative(unsigned int numPoints, struct neighborTableLookup * inputNeighborTable, int * directNeighborArray, struct gpulookuptable * gpuLookupArray);

bool generateDistanceTableFromPreviousTableBatchesAlternate(std::vector<struct dataElem> * dataPoints, struct gpulookuptable * gpuLookupArray, int * directNeighborArray, unsigned int * totalDirectNeighbors, double * epsilon, double * previousEpsilon,  struct neighborTableLookup * neighborTable, std::vector<struct neighborDataPtrs> * pointersToNeighbors, unsigned int * totalNeighbors);


//estimates the total result size by sampling the dataset
void EstimateBatchesGPUGridIndex(std::vector<struct dataElem> * dataPoints, double * epsilon, struct grid * index, double * gridMin_x, double * gridMin_y, int * gridNumXCells, int * gridNumYCells, int * lookupArr, int * totalNeighbors);

void constructNeighborTable(thrust::host_vector<structresults> * hVectResults, struct table * neighborTable, unsigned int * cnt);
void constructNeighborTableKeyValue(int * pointIDKey, int * pointInDistValue, struct table * neighborTable, unsigned int * cnt);



void generateNeighborArrayForGPU(unsigned int numPoints, struct table * inputNeighborTable, int * directNeighborArray, struct gpulookuptable * gpuLookupArray);

void allocateResultSet(struct structresults * dev_results, struct structresults * results);

void generateDistanceTableFromPreviousTable(std::vector<struct dataElem> * dataPoints, struct gpulookuptable * gpuLookupArray, int * directNeighborArray, int * totalDirectNeighbors, double * epsilon, struct table * neighborTable);
bool generateDistanceTableFromPreviousTableBatches(std::vector<struct dataElem> * dataPoints, struct gpulookuptable * gpuLookupArray, int * directNeighborArray, unsigned int * totalDirectNeighbors, double * epsilon,  double * previousEpsilon, struct table * neighborTable, unsigned int * totalNeighbors);
//Kernel:
//__global__ void kernel( unsigned int *debug1, unsigned int *debug2, double *epsilon, unsigned int * cnt, struct point * database, struct structresults * results);