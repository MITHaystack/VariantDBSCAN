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
#include <vector>
//list of function prototypes across files to be included by all files.

void importDataset(std::vector<dataElem> *dataPoints, char * fname);
void randData(std::vector<dataElem> *dataPoints, int numElem);
void randDataInClusters(std::vector<dataElem> *dataPoints, int numElem, int numClusters);
//void createEntryMBBs(std::vector<dataElem> *dataPoints, Rect * dataRects);
//void BruteForceFindPoints(std::vector<dataElem> *dataPoints, std::vector<int> *candidateSet);
//bool MySearchCallback(int id, void* arg);
//bool DBSCANmySearchCallback(int id, void* arg);
//bool DBSCANmySearchCallbackSequential(int id, void* arg);
//bool DBSCANmySearchCallbackParallel(int id, void* arg);
bool compareDataElemStructFunc(const dataElem &elem1, const dataElem &elem2);
//void createEntryMBBMultiplePoints(std::vector<dataElem> *dataPoints, std::vector<std::vector<int> > *MPB_ids, MPBRect * dataRectsMPB);
void importDBScanInstances(std::vector<struct experiment> * exper, char * fname);

//bool TESTCALLBACK(int id, void* arg);