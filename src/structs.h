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


///Struct that outlines the parameters for multiple instances of DBScan, each one described per struct. 
///
///
struct experiment{
	double epsilon;
	int minpts;
	unsigned int variantID;
};

	//keeps an ordering of the experiments/variants, whether it should be clustered from scratch, and if it has started
	struct schedInfo{
		bool clusterScratch; //if it should be clustered from scratch 0-no 1-yes
		bool status; //whether it has been started 0-not started, 1-started
	};


///2-D data struct
///
///
struct dataElem{
	double x; 
	double y; 
};



///Struct used to order the clusters that will get reused
///
///
struct densityStruct{
		int clusterID;
		double density;
	};


///Used for the index of point objects - multiple points per MBB.
///They make the MBBs that are inserted into the tree (2-D MBBs only)
///
struct MPBRect
{

	  double MBB_min[2]; //MBB min
	  double MBB_max[2]; //MBB max
	  int pid; //MBB id

	  //takes as input a pointer to the data points and a vector of the ids 
	  //inside the data points to index together 
  	void CreateMBB(std::vector<struct dataElem> * dataPoints, std::vector<int>* multiplePointsToIndex){
		
  		

		int id=(*multiplePointsToIndex)[0];
		
		MBB_min[0]=(*dataPoints)[id].x;
		MBB_max[0]=(*dataPoints)[id].x;
		MBB_min[1]=(*dataPoints)[id].y;
		MBB_max[1]=(*dataPoints)[id].y;

		for (int i=0; i<multiplePointsToIndex->size(); i++)
		{
				id=(*multiplePointsToIndex)[i];
				
			
				
				if (((*dataPoints)[id].x)<MBB_min[0])
				{
					MBB_min[0]=(*dataPoints)[id].x;
				}
				if (((*dataPoints)[id].x)>MBB_max[0])
				{
					MBB_max[0]=(*dataPoints)[id].x;
				}

				if (((*dataPoints)[id].y)<MBB_min[1])
				{
					MBB_min[1]=(*dataPoints)[id].y;
				}
				if (((*dataPoints)[id].y)>MBB_max[1])
				{
					MBB_max[1]=(*dataPoints)[id].y;
				}

			
		}

				
	

	} //end of function CreateMBB

	


};

///Used for the index of point objects. 
///They make the MBBs that are inserted into the tree (2-D MBBs)
///
struct Rect
{
	Rect()  {}


	  double P1[2];//point
	  double MBB_min[2]; //MBB min
	  double MBB_max[2]; //MBB max
	  int pid; //point id

  	void CreateMBB(){
		MBB_min[0]=P1[0];
		MBB_max[0]=P1[0];
		MBB_min[1]=P1[1];
		MBB_max[1]=P1[1];
			
	} //end of function CreateMBB




};


///MBB for querying the R-tree
///
///
struct QueryRect
{
  QueryRect()  {}
  
  //two points defining the MBB: assume the two points can be given in any order:
  double P1[2];
  double P2[2];

  //MBB min and max points
  double MBB_min[2]; //MBB min
  double MBB_max[2]; //MBB max
  

  	void CreateMBB()
 	{
		
		for (int i=0; i<2; i++)
		{	
			if (P1[i]<P2[i])
			{
				MBB_min[i]=P1[i];
				MBB_max[i]=P2[i];
			}
			else
			{
				MBB_max[i]=P1[i];
				MBB_min[i]=P2[i];
			}
		}
		
	} //end of function CreateMBB


};

#endif
