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

#ifndef STRUCTS_H
#define STRUCTS_H
#include <vector>
#include <stdio.h>
#include <iostream>
//thrust
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>




//need to pass in the neighbortable thats an array of the dataset size.
//carry around a pointer to the array that has the points within epsilon though
struct neighborTableLookup
{
	int pointID;
	int indexmin;
	int indexmax;
	int * dataPtr;
};

//a struct that points to the arrays of individual data points within epsilon
//and the size of each of these arrays (needed to construct a subsequent neighbor table)
//will be used inside a vector.
struct neighborDataPtrs{
	int * dataPtr;
	int sizeOfDataArr;
};


//struct that outlines the parameters for performance evaluation experiments
//multiple instances of DBScan, each one described per struct 
struct experiment{
	double epsilon;
	int minpts;
};

	//keeps an ordering of the experiments/variants, whether it should be clustered from scratch, and if it has started
	struct schedInfo{
		bool clusterScratch; //if it should be clustered from scratch 0-no 1-yes
		bool status; //whether it has been started 0-not started, 1-started
	};



//holding all of the data including the value and time
struct dataElem{
	double x; //latitude
	double y; //longitude
	//double val;
	//double time;
	
};



//holding just the x and y values:
struct point{
double x;
double y;
};

//the result set:
struct structresults{
int pointID;
int pointInDist;
};

//the lookup table.  The index is the point ID, each contains a vector
struct table{
int pointID;
std::vector<int> neighbors;
};

//index lookup table for the GPU. Contains the indices for each point in an array
//where the array stores the direct neighbours of all of the points
struct gpulookuptable{
int indexmin;
int indexmax;
};

struct grid{	
int indexmin; //Contains the indices for each point in an array where the array stores the ids of the points in the grid
int indexmax;
};






struct compareThrust
{
  __host__ __device__
  bool operator()(structresults const& lhs, structresults const& rhs)
  {
    if (lhs.pointID != rhs.pointID)
    {
        return (lhs.pointID < rhs.pointID);
    }
        return (lhs.pointInDist < rhs.pointInDist);
  }
};


#endif
