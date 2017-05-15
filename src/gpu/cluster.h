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

void dbscan(struct table * neighborTable, unsigned int numPoints, int minPts);

void dbscanWithFilter(std::vector<struct dataElem> *dataPoints, struct table * neighborTable, double epsilon, int minPts);

void dbscanAlternate(struct neighborTableLookup * neighborTable, unsigned int numPoints, int minPts);

void copyVect(std::vector<int> * dest, std::vector<int> * source);

int filterCandidates(int pointID, std::vector<struct dataElem> * dataPoints, std::vector<int> * candidateSet, double distance, std::vector<int> * setIDsInDistPtr);

double EuclidianDistance(struct dataElem * point1, struct dataElem * point2);