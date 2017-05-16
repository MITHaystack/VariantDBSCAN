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

#include <stdio.h>
#include <vector>
#include <fstream>
#include <iostream>
#include <string.h>
#include <cstdlib>
#include "prototypes.h"
#include <algorithm>
//for random number generation
#include "randomc.h"
#include "rand.h"



//comparison function to sort vector
//sort in increasing order with epsilon and decreasing order with minpts for a given epsilon
bool compareByEpsilon(const experiment &a, const experiment &b)
{
	if (a.epsilon==b.epsilon)
	{
		return a.minpts < b.minpts;
	}
	
	return a.epsilon>b.epsilon;
}



//reads in the data file:
//the file is a csv, reads in the data in the following order:
//longitude
//latitude
//TEC

//the longitude is in the range 0 to 360
//the latitude is in the range from +90 to -90.
//We need to transform the latitude to be from 0 to 180. Thus: 90 becomes 0, and -90 becomes 180.

void importDataset(std::vector<dataElem> *dataPoints, char * fname)
{
	struct dataElem tmpStruct;

	char in_line[400];
	char data[4][20]; //read in 4 values (the last value, time has not been included yet)

	FILE *fileInput = fopen(fname,"r");
	
	//loop over each line in the file, until the end of the file (fgets will return NULL)
	while(fgets(in_line, 400, fileInput)!=NULL)
	{
	//fgets(in_line, 400, fileInput);

	sscanf(in_line,"%[^,],%[^,],%[^,]",data[0], data[1],data[2]);

	// std::cout <<"\nLine: "<<in_line;
	// std::cout <<"\nval1: "<<data[0];
	// std::cout <<"\nval2: "<<data[1];
	// std::cout <<"\nval3: "<<data[2];

	//set the values in the temp struct
	
	//the latitude needs to be adjusted to be in the range between 0-180 degrees
	tmpStruct.x=atof(data[1]); //latitude

	//if positive: transform into the range 0-90
	if (tmpStruct.x>0.0)
	{
		double deltaD=(90.0-tmpStruct.x)/90.0;
		tmpStruct.x=deltaD*90.0;
	}
	//if negative transform into the range 90-180
	else
	{
		double deltaD=(-tmpStruct.x)/90.0;
		tmpStruct.x=(deltaD*90.0)+90.0;
	}


	//longitude
	tmpStruct.y=atof(data[0]); //longitude

	//TEC
	//actual TEC:
	//tmpStruct.val=atof(data[2]); //TEC
	//for testing, set TEC to 50, because when we do MSB, we need a constant TEC
	//printf("\nWARNING: THE TEC VALUE IMPORTED FROM FILE HAS BEEN ARTIFICALLY SET TO 50 FOR TESTING");	
	

	//time set to 0 for now
	//tmpStruct.time=0;

	//add tje temp struct to the vector of datapoints

	(*dataPoints).push_back(tmpStruct);


	} //end of while loop


	//Test output data 

	// for (int i=0; i<(*dataPoints).size(); i++)
	// {
	// 	printf("\n%f,%f,%f,%f",(*dataPoints)[i].x,(*dataPoints)[i].y,(*dataPoints)[i].val,(*dataPoints)[i].time);
	// }




}




//generates random data in the region: latitude between 0 and 180, longitude between 0 and 360
/*
void randData(std::vector<dataElem> *dataPoints, int numElem)
{
	struct dataElem tmpStruct;
	
	for (int i=0; i<numElem; i++){

			tmpStruct.x=rg.Random()*180; //latitude
			tmpStruct.y=rg.Random()*360; //longitude
			
			tmpStruct.time=0;
			//tmpStruct.time=rg.Random()*100; 
			
			//tmpStruct.val=rg.Random()*100;
			tmpStruct.val=50.0; //make the value constant, or else wont find anything in the dataset
			
			(*dataPoints).push_back(tmpStruct);
		}

		printf("SIZE test data:%zu", dataPoints[0].size());
}

//generates random data in certain clusters to confirm that the algorithm works properly
void randDataInClusters(std::vector<dataElem> *dataPoints, int numElem, int numClusters)
{

	double clusterCenter[numClusters][2];	
	double latOffset=5;
	double longOffset=5;

	//first randomly generate a number of cluster centers
	// for (int i=0; i<numClusters; i++)
	// {
	// 	clusterCenter[i][0]=latOffset+((rg.Random()*(180-latOffset))); //latitude
	// 	clusterCenter[i][1]=longOffset+((rg.Random()*(360-longOffset))); //longitude
	// 	printf("\nCluster center: %f,%f", clusterCenter[i][0],clusterCenter[i][1]);
	// }

	//manually create cluster centers:
	clusterCenter[0][0]=20;
	clusterCenter[0][1]=20;
	
	clusterCenter[1][0]=20;
	clusterCenter[1][1]=345;
	
	clusterCenter[2][0]=90;
	clusterCenter[2][1]=180;
	
	clusterCenter[3][0]=160;
	clusterCenter[3][1]=20;
	
	clusterCenter[4][0]=160;
	clusterCenter[4][1]=345;

	//cluster on the border
	clusterCenter[5][0]=90;
	clusterCenter[5][1]=355;

	//cluster on the border
	clusterCenter[6][0]=90;
	clusterCenter[6][1]=5;
	




	 int elemPerCluster=numElem/numClusters;

	 struct dataElem tmpStruct;
	

	// //make clusters within 15 units from the cluster center
	for (int i=0; i<numClusters; i++)
	{
		for (int j=0; j<elemPerCluster; j++){

				tmpStruct.x=(clusterCenter[i][0]-latOffset)+(rg.Random()*(2*latOffset)); //latitude
				tmpStruct.y=(clusterCenter[i][1]-longOffset)+(rg.Random()*(2*longOffset)); //longitude
				tmpStruct.time=0; 
				tmpStruct.val=50.0; //make the value constant, or else wont find anything in the dataset

				(*dataPoints).push_back(tmpStruct);
			}
	}

		printf("SIZE test data:%zu", dataPoints[0].size());
}

*/



void importDBScanInstances(std::vector<struct experiment> * exper, char * fname)
{
	struct experiment tmpStruct;
	int count=0;
	char in_line[400];
	char data[2][20]; //read in 2 values (epsilon and minpts)

	FILE *fileInput = fopen(fname,"r");
	
	//loop over each line in the file, until the end of the file (fgets will return NULL)
	while(fgets(in_line, 400, fileInput)!=NULL)
	{
		sscanf(in_line,"%[^,],%[^,]",data[0], data[1]);
		tmpStruct.epsilon=atof(data[0]);
		tmpStruct.minpts=atoi(data[1]);
		exper->push_back(tmpStruct);
		count++;
	} //end of while loop	


	//sort list of experiments
	std::sort(exper->begin(),exper->end(),compareByEpsilon);

	//output the sorted experiments
	printf("\nImported information for the following number of DBSCAN instances (sorted by epsilon in decreasing order): %d",count);
	for (int i=0; i<exper->size();i++)
	{
		printf("\nEps: %f, minpts: %d", (*exper)[i].epsilon, (*exper)[i].minpts);
	}

}








