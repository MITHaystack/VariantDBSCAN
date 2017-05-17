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

#include <vector>
#include <stdio.h>
#include <string.h>
#include <cstdlib>
//for opening shared library
#include <dlfcn.h>
//definition of the data is in structs.h
#include "structs.h"
//header file for the prototype of the shared library
#include "c_test_prog.h"


using namespace std;


////////////////////////////////////////
//Test program for the c interface to the c++ shared VDBSCAN library
//libSharedVDBSCAN.so
////////////////////////////////////////
int main()
{

	//open shared library:
	void *handle;
    char *error;
 
    handle = dlopen ("libSharedVDBSCAN.so", RTLD_LAZY);
    if (!handle) {
        fputs (dlerror(), stderr);
        exit(1);
    }

    //2D dataset:
	double * inputx;
	double * inputy;

	//epsilon and minpts values
	double * inputEps;
	unsigned int * inputMinpts;
	
	//Minimum bounding box size for the index (70 default)	
	int MBBsize=70;

	//result array from the clustering across all variants
	unsigned int * retArr;
	
	
	//verbose mode: outputs optional information about the clustering, can be turned off.
	bool verbose=true;

	//datapoints
	std::vector<dataElem> dataPoints;


	/////////////////////////////////////
	//import dataset:
	/////////////////////////////////////

	struct dataElem tmpStruct;

	char in_line[400];
	char data[2][20];
	char dataset[]="test_dataset.csv";
	FILE *fileInput = fopen(dataset,"r");
	
	printf("\nImporting dataset: %s\n", dataset);

	//loop over each line in the file, until the end of the file (fgets will return NULL)
	while(fgets(in_line, 400, fileInput)!=NULL)
	{
		sscanf(in_line,"%[^,],%[^,]",data[0], data[1]);

		//set the values in the temp struct
		tmpStruct.x=atof(data[0]); 
		tmpStruct.y=atof(data[1]); 

		//append to the vector
		dataPoints.push_back(tmpStruct);

	} //end of while loop

	/////////////////////////////////////
	//end import dataset
	/////////////////////////////////////

	//size of dataset
	unsigned int datasetSize=dataPoints.size();
	
	//store the dataset in 2 arrays for the x and y values 
	inputx=new double[datasetSize];
	inputy=new double[datasetSize];

	//Copy the vector data into the input x and y:
	for (int i=0; i<datasetSize; i++)
	{
		inputx[i]=dataPoints[i].x;
		inputy[i]=dataPoints[i].y;
	}

	//Create two variants to cluster:
	unsigned int numVariants=2;
	inputEps=new double[numVariants];
	inputMinpts=new unsigned int[numVariants];

	//example variant1:
	inputEps[0]=1.0;
	inputMinpts[0]=10;
	//example variant2:
	inputEps[1]=0.5;
	inputMinpts[1]=12;


	//the result array is the dataset size * the number of variants in one big array.
	retArr=new unsigned int [datasetSize*numVariants];

	//Call VariantDBSCAN
	libVDBSCAN(inputx, inputy, datasetSize, inputEps, inputMinpts, numVariants, MBBsize, retArr, verbose);


	//Example output the assignment of each data point to their cluster ID for each variant.
	for (int i=0; i<numVariants; i++)
	{
		printf("\n\nVariant: %i, Epsilon: %f, minpts: %d",i, inputEps[i], inputMinpts[i]);
		for (int j=0; j<datasetSize; j++)
		{
			//point id, x,y,cluster id
			printf("\n%d, %f, %f, %d",j,inputx[j],inputy[j],retArr[(datasetSize*i)+j]); 
		}
	}
	printf("\n");

	delete [] inputx;
	delete [] inputy;
	delete [] inputEps;
	delete [] inputMinpts;
	delete [] retArr;

	return 0;
}