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

//list of function prototypes (mostly for the shared library)


///Imports the 2-D dataset
void importDataset(std::vector<dataElem> *dataPoints, char * fname);


///Generates MBBs for the R-tree
void createEntryMBBs(std::vector<dataElem> *dataPoints, Rect * dataRects);


///Callback function for the R-tree
bool DBSCANmySearchCallbackParallel(int id, void* arg);


///Comparison function for sorting
bool compareDataElemStructFunc(const dataElem &elem1, const dataElem &elem2);


///Generates MBBs for for the R-tree when indexing multiple points per MBB
void createEntryMBBMultiplePoints(std::vector<dataElem> *dataPoints, std::vector<std::vector<int> > *MPB_ids, MPBRect * dataRectsMPB, int MBBsize);


///Imports the list of DBSCAN instances (not used in the shared library version)
void importDBScanInstances(std::vector<struct experiment> * exper, char * fname);


///Comparison function for sorting
bool compareByEpsilon(const experiment &a, const experiment &b);


///Used for binning the input dataset
int bin_x(double x);


///Used for binning the input dataset
int bin_y(double x);


///Comparison function for sorting
bool compareDataElemStructFunc(const dataElem &elem1, const dataElem &elem2);


///Bins the 2-D input dataset, and keeps track of where the points in space were mapped to the original input dataset.
void binDataset(std::vector<dataElem> *dataPoints, int numBins, std::vector<int> *mapping, bool verbose);