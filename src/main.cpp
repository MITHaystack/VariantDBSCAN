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

#include <math.h>
#include <cstdlib>
#include <stdio.h>
#include "prototypes.h"
#include "globals.h"
#include "RTree.h"
#include "omp.h"
#include "DBScan.h"
#include "schedule.h"
#include <algorithm> 
#include <string.h>
#include <fstream>
#include <iostream>
#include <string>
#include <limits>

using namespace std;




//returns all of the cluster ids for all of the points for every variant (retArr)
//one big 1-D array of |D|*|V| (|D| the dataset size, |V| the number of variants)
//Also returns the variant ordering as epsilon and minpts, because it gets sorted as part of the scheduling
//retEps, retMinpts
extern "C" int libVDBSCAN(double * inputx, double * inputy, unsigned int datasetSize, double * inputEpsilon, unsigned int * inputMinpts, unsigned int numVariants, int MBBsize, unsigned int * retArr, bool verbose)
{
	if (verbose==true)	
	printf("Verbose mode on");	

	/////////////////////////
	//OpenMP
	/////////////////////////
	omp_set_num_threads(NSEARCHTHREADS);


	/////////////////////////
	//import variants
	/////////////////////////
	//make a vector that stores the experiments/variants
	std::vector<struct experiment> experimentList;
	

	//store the variant data in a vector struct from the passed in arrays
	for (int i=0; i<numVariants; i++){
		experiment tmp;
		tmp.epsilon=inputEpsilon[i];
		tmp.minpts=inputMinpts[i];
		tmp.variantID=i;
		experimentList.push_back(tmp);
	}	

	//Greedy scheduling: sort the experiment list from lowest to highest epsilon and highest to lowest minpts for the same epsilon value
	//Pg 5 in the paper pdf.
	std::sort(experimentList.begin(),experimentList.end(),compareByEpsilon);


	


	//////////////////////////////
	//import the dataset from data passed in as as arrays
	/////////////////////////////

	std::vector<struct dataElem> dataPoints;
	for (int i=0; i<datasetSize; i++){
		dataElem tmp;
		tmp.x=inputx[i];
		tmp.y=inputy[i];
		dataPoints.push_back(tmp);

	}




	//Bin the data points so that nearby points in a confined spatial region are nearby each other in memory.
	//This ensures that the MBBs in the R-tree don't become enormous, thus creating massive candidate sets.
	//And this also benefits locality
	std::vector<int> datasetMapping;
	//300 is the number of bins (300x300), dataset mapping keeps track of where the original point ids went in the bins, so that they can be mapped back
	//to the input dataset
	binDataset(&dataPoints, 300, &datasetMapping, verbose);


	//create R-tree index (MPB, multiple points per box)
	RTree<int,double,2,float> treeMPB;

	////////////////////////////////
	//One index for searching the tree when clustering
	////////////////////////////////
	
	//the number of rects inserted into the R-tree
	unsigned int numMPBs=ceil((dataPoints.size()*1.0)/(MBBsize*1.0));
	
	//struct that holds the data MBB rectangles for the R-tree
	MPBRect * dataRectsMPB;
	dataRectsMPB= new MPBRect[numMPBs];

	if (verbose==true)
	printf("\nNumber of MBB rects allocated: %d, number of points per box: %d", numMPBs, MBBsize); 
	

	
	//have a vector of ids that store the ids of the datapoints that are in a given MBB (MSB)
	//its an array of vectors, one for each MSB.  Make this a vector of vectors because
	//we might modify the scheme to allow a variable number of points per MBB
	std::vector<std::vector<int> > dataLookupMPB;



	//create MBBs that contain multiple point objects
	createEntryMBBMultiplePoints(&dataPoints, &dataLookupMPB, dataRectsMPB,MBBsize);

	

	//insert data into the R-tree
	for (int i=0; i<numMPBs; i++){
		treeMPB.Insert(dataRectsMPB[i].MBB_min,dataRectsMPB[i].MBB_max, i);	
	}	

	////////////////////////////////
	//Another index for discerning if points will belong to a cluster in a new instance. Use high resolution tree.	
	////////////////////////////////
	RTree<int,double,2,float> treeHighRes;
	

	//Create MBBs
	//struct to store MBB data
	//this is for 1 point per MBB
	
	Rect * dataRects;
	dataRects= new Rect[dataPoints.size()];
	createEntryMBBs(&dataPoints, dataRects);

	
	//insert data into the R-tree
	for (int i=0; i<dataPoints.size(); i++){
		//actual:
		treeHighRes.Insert(dataRects[i].MBB_min,dataRects[i].MBB_max, dataRects[i].pid);	
	}



	//instance vector:
	std::vector <DBScan *>instanceVect;		
	
	//make array of objects for each variant
	DBScan * clusterReusePreviousClusters[experimentList.size()];
	
	

	//initialize DBSCAN instances	
	for (int i=0; i<experimentList.size();i++)
	{
		//The first instance uses the sequential DBScan alg, and subsequent experiments use the cluster reuse algorithm.
		if (i==0){
			clusterReusePreviousClusters[i]=new DBScan(&dataPoints, experimentList[i].epsilon, experimentList[i].minpts, &treeMPB, &dataLookupMPB, &treeHighRes);
		}
		else{
		clusterReusePreviousClusters[i]=new DBScan(&dataPoints, experimentList[i].epsilon, experimentList[i].minpts, &treeMPB, &dataLookupMPB, &treeHighRes);
		}
		//we keep a pointer to all DBScan objects
		instanceVect.push_back(clusterReusePreviousClusters[i]);
	}



	//Allocate a vector of pointers to all of the DBScan objects
	//and then send these pointers to each object
	for (int i=0; i<experimentList.size(); i++){
		clusterReusePreviousClusters[i]->instanceVect=instanceVect;
	}


	////////////////////////////////
	//Create scheduler object that determines which experiments reuse results and which ones don't
	////////////////////////////////
	Schedule experimentSchedule(&experimentList);
	

	////////////////////////////////////////////
	//Run algorithm below:
	//Note that the scheduler determines whether a variant is clustered from scratch or reuses a previous variant
	//The scheduler also assigns the variants to threads, and hence, not a static schedule by loop iteration
	/////////////////////////////////////////////	

	//We multiply the number of iterations by 2 to fake a thread pool to prevent load imbalance when 
	//concurrently clustering more variants than threads.
	//A job queue is checked, and will return false if there are no more variants to cluster
	#pragma omp parallel for schedule (static,1)
	for (int i=0; i<experimentList.size()*2; i++)
	{
		int tid=omp_get_thread_num();
		if (verbose==true)
		printf("\ntid: %d -- starting loop iteration: %d",tid,i);
		int instanceIDReuse=0; // the instance ID that we reuse
		int instanceIDCluster=0; // the instance ID that we cluster
		//if this is true, there is a variant to cluster, if false, there are none left
		if(experimentSchedule.determineReuse(i,&instanceIDReuse, &instanceIDCluster)){

			//if the ID to reuse is the same as the ID to cluster, then that means we cluster from scratch.
			if(instanceIDReuse==instanceIDCluster){
			if (verbose==true)	
			printf("\ntid: %d variant: %d, building cluster from scratch",tid,instanceIDCluster);
			clusterReusePreviousClusters[instanceIDCluster]->algDBScanParallel(); 
			experimentSchedule.finishedList[instanceIDCluster]=true;
			if (verbose==true)
			printf("\nVariant completed. tid: %d cluster variant: %d, Number of clusters (including noise cluster): %d",tid,instanceIDReuse,clusterReusePreviousClusters[instanceIDCluster]->getDBScanNumClusters());
			}
			
			//can reuse variants
			else{
			if (verbose==true)	
			printf("\ntid: %d reuse instance: %d, cluster variant: %d",tid,instanceIDReuse,instanceIDCluster);
			clusterReusePreviousClusters[instanceIDCluster]->assignPointsToPredefinedCluster(instanceIDReuse);		
			experimentSchedule.finishedList[instanceIDCluster]=true;
			if (verbose==true)
			printf("\nVariant completed. tid: %d reuse instance: %d, cluster variant: %d, Number of clusters (including noise cluster): %d",tid,instanceIDReuse,instanceIDCluster,clusterReusePreviousClusters[instanceIDCluster]->getDBScanNumClusters());
			}
		

		
		} //end of if statement
		
		
	} //end for loop


	//reorder the data point ids based on the mapping from the mapping into bins
	for (int i=0; i<experimentList.size(); i++){
		std::vector<int>clusterIDsTmp=clusterReusePreviousClusters[i]->clusterIDs;
		for (int j=0; j<dataPoints.size(); j++){
		int idx=datasetMapping[j];	
		int clusterId=clusterIDsTmp[j];
		clusterReusePreviousClusters[i]->clusterIDs[idx]=clusterId;
		}
	}

	////////////////////////////////////////////////////////////
	//Return the data as 1 big 1D array across all variants
	////////////////////////////////////////////////////////////

	//The ordering of the variants will have changed based on the Greedy Schedule ordering
	//Thus we need to use an appropriate offset into the array based on the original experiment ID
	if (verbose==true)
	printf("\nMapping from initial variant id to order that variants were processed:");	

	for (int i=0; i<experimentList.size(); i++)
	{
		if (verbose==true)
		printf("\nEnum var id: %d, initial var id: %d",i,experimentList[i].variantID);
		unsigned int offset=dataPoints.size()*experimentList[i].variantID;
		unsigned int cnt=0;
		for (int j=offset; j<offset+dataPoints.size(); j++)
		{
			retArr[j]=clusterReusePreviousClusters[i]->clusterIDs[cnt];
			cnt++;
		}
	}



//free memory
for (int i=0; i<experimentList.size(); i++){
delete clusterReusePreviousClusters[i];
}

delete [] dataRectsMPB;
delete [] dataRects;

	
return 0;

} //end of lib


//Test program (use the c_test_prog instead):

// int main(int argc, char *argv[])
// {
	
	
// 	double * inputx;
// 	double * inputy;
// 	double * inputEpsilon;
// 	unsigned int * inputMinpts;
// 	unsigned int numVariants=1;
// 	int MBBsize=70;
// 	unsigned int * retArr;
	
// 	double * retEps;
// 	unsigned int * retMinpts;
// 	retEps=new double[numVariants];
// 	retMinpts=new unsigned int [numVariants];
// 	bool verbose=true;

// 	std::vector<dataElem> dataPoints;

// 	//////////////////////////////
// 	//import dataset:

// 	struct dataElem tmpStruct;

// 	char in_line[400];
// 	char data[4][20]; //read in 4 values (the last value, time has not been included yet)

// 	char inputFname[500];
// 	strcpy(inputFname,"/home/mgowanlock/Geospace/data/geodata/iono_20min_2Mpts.txt");
// 	FILE *fileInput = fopen(inputFname,"r");
	
// 	printf("\nImporting dataset: %s\n", inputFname);

// 	//loop over each line in the file, until the end of the file (fgets will return NULL)
// 	while(fgets(in_line, 400, fileInput)!=NULL)
// 	{
// 	//fgets(in_line, 400, fileInput);

// 	sscanf(in_line,"%[^,],%[^,],%[^,]",data[0], data[1],data[2]);

// 	//set the values in the temp struct
// 	tmpStruct.x=atof(data[1]); //latitude

// 	//longitude
// 	tmpStruct.y=atof(data[0]); //longitude

// 	dataPoints.push_back(tmpStruct);


// 	} //end of while loop




// 	//end import dataset
// 	/////////////////////////////////////

// 	unsigned int datasetSize=dataPoints.size();
// 	inputx=new double[datasetSize];
// 	inputy=new double[datasetSize];
// 	inputEpsilon=new double[numVariants];
// 	inputMinpts=new unsigned int[numVariants];
// 	retArr=new unsigned int [datasetSize*numVariants];

// 	inputEpsilon[0]=0.2;
// 	inputMinpts[0]=4;


// 	//put the vector data into the input x and y:
// 	for (int i=0; i<datasetSize; i++)
// 	{
// 		inputx[i]=dataPoints[i].x;
// 		inputy[i]=dataPoints[i].y;
// 	}

	

// 	libVDBSCAN(inputx, inputy, datasetSize, inputEpsilon, inputMinpts, numVariants, MBBsize, retArr, verbose);



// 	//test print result
// 	int cnt=0;
// 	for (int i=0; i<numVariants; i++)
// 	{
// 		printf("\n\n****Variant: %d (sorted though)\n",i);
// 		for (int j=0; j<datasetSize; j++)
// 		{
// 			printf("%d\n",retArr[cnt]);
// 			cnt++;
// 		}
// 	}
// 	return 0;
// }	



//multiple points per MBB:
//called this a multiple point box (MPB)
void createEntryMBBMultiplePoints(std::vector<dataElem> *dataPoints, std::vector<std::vector<int> > *MPB_ids, MPBRect * dataRectsMPB, int MBBSize)
{
	int MPB_cnt=0;
	int dataPointCnt=0;
	for (int i=0; i<(*dataPoints).size(); i+=MBBSize){
		
		//create new space in the MSB vector	
		(*MPB_ids).push_back(vector<int>());
		for (int j=0; j<MBBSize; j++)
		{

			//don't want to go over the size of the number of datapoints.	
			//the last MBB might have less than MBBSIZE number of points in it	
			//insert the dataElem ID into the vector of vectors
			if (((MPB_cnt*MBBSize)+j)<(*dataPoints).size()){	
			(*MPB_ids)[MPB_cnt].push_back((MPB_cnt*MBBSize)+j);
			
			//printf("\nMBB count: %d, data point cnt: %d",MPB_cnt, dataPointCnt);
			dataPointCnt++;
			}
		}
		
		
		//create a new MBB for the point(s)
		//&(*MPB_ids)[MPB_cnt] is confusing, but it's a pointer to a vector inside the vector that stores the IDs
		//of the data points that are within each MPB
		dataRectsMPB[MPB_cnt].CreateMBB(dataPoints, &(*MPB_ids)[MPB_cnt]);
		
		MPB_cnt++;
	} //end main for loop


}	





//create MBBs for R-tree
void createEntryMBBs(std::vector<dataElem> *dataPoints, Rect * dataRects)
{
	for (int i=0; i<(*dataPoints).size(); i++){
		dataRects[i].P1[0]=(*dataPoints)[i].x;
		dataRects[i].P1[1]=(*dataPoints)[i].y;
		dataRects[i].pid=i;
		dataRects[i].CreateMBB();
	}

}	



//the data needs to be binned, such that massive MBBs do not occur when indexing multiple points per box
//numBins is the number of bins in the x and y dimension
//The mapping just keeps track of where the original data points passed into the function went
void binDataset(std::vector<dataElem> *dataPoints, int numBins, std::vector<int> *mapping, bool verbose)
{

	

	//calculate the min/max value of the dataset for both the x and y dimensions
	double min_X=std::numeric_limits<double>::max();
	double min_Y=std::numeric_limits<double>::max();
	double max_X=std::numeric_limits<double>::min();
	double max_Y=std::numeric_limits<double>::min();


	//find the min/max values in both dimensions, x and y
	for (int i = 0; i < dataPoints->size(); i++){
		if ((*dataPoints)[i].x<min_X){
			min_X=(*dataPoints)[i].x;
		}

		if ((*dataPoints)[i].x>max_X){
			max_X=(*dataPoints)[i].x;
		}

		if ((*dataPoints)[i].y<min_Y){
			min_Y=(*dataPoints)[i].y;
		}

		if ((*dataPoints)[i].y>max_Y){
			max_Y=(*dataPoints)[i].y;
		}
	}

	
	double x_width=(max_X-min_X)*1.01;//multiply by a small amount to ensure no fp rounding errors into the wrong bin 
	double y_width=(max_Y-min_Y)*1.01;//multiply by a small amount to ensure no fp rounding errors into the wrong bin

	//need to keep track of where the data points are in the bins so we can map them back to the original data
	struct binnedData
	{
		vector<int> idsInBin;
	};


	//make a temporary struct
	//binnedData bins[numBins][numBins]; //on stack

	binnedData ** bins;
	bins=new binnedData*[numBins];
	for (int i=0; i<numBins; i++)
	{	
		bins[i]=new binnedData[numBins];
	}
	

	//calculate offsets for the binned data so that we get the correct number of bins and can deal with negative values
	double xoffset=0;
	double yoffset=0;
	if (min_X>0){
		xoffset=-min_X;
	}
	else{
		xoffset=fabs(min_X);
	}

	if (min_Y>0){
		yoffset=-min_Y;
	}
	else{
		yoffset=fabs(min_Y);
	}


	
	//bin the dataset
	int minXBinNum=std::numeric_limits<int>::max();;
	int minYBinNum=std::numeric_limits<int>::max();;
	int maxXBinNum=std::numeric_limits<int>::min();;
	int maxYBinNum=std::numeric_limits<int>::min();;


	for (int i = 0; i < dataPoints->size(); i++){
	int xbin=floor((((*dataPoints)[i].x+xoffset)/x_width)*numBins); //need to offset by the minimum value to start the bins at 0
	int ybin=floor((((*dataPoints)[i].y+yoffset)/y_width)*numBins); //need to offset by the minimum value to start the bins at 0
	
	//for some stats:
	if(xbin<minXBinNum){
		minXBinNum=xbin;
	}
	if(xbin>maxXBinNum){
		maxXBinNum=xbin;
	}
	if(ybin<minYBinNum){
		minYBinNum=ybin;
	}
	if(ybin>maxYBinNum){
		maxYBinNum=ybin;
	}



	//Make sure that the bin ids are never out of range	
	if (xbin>(numBins-1) || ybin >(numBins-1) || xbin<0 || ybin <0)
	{
		printf("\nError, bins out of range, xbin: %d, ybin:  %d",xbin,ybin);
		return;
	}
	//insert into the temporary structure	
	bins[xbin][ybin].idsInBin.push_back(i);
	}

	if (verbose==true)
	printf("\nMin bin ids: x: %d, y:%d, Max bin ids: x: %d, y:%d", minXBinNum,minYBinNum, maxXBinNum, maxYBinNum);

	

	//make a copy of the dataset
	std::vector<struct dataElem> tmpDataset;
	for (int i = 0; i < dataPoints->size(); i++){
		tmpDataset.push_back((*dataPoints)[i]);

	}

	//now reorder the data based on how the data was binned
	int cnt=0;
	for (int i = 0; i < numBins; i++){
		for (int j = 0; j < numBins; j++){
			
			for (int k=0; k<bins[i][j].idsInBin.size(); k++){
			int idx=bins[i][j].idsInBin[k];
			dataElem tmp;
			tmp.x=tmpDataset[idx].x;
			tmp.y=tmpDataset[idx].y;
			(*dataPoints)[cnt]=tmp; //reordering the input dataset
			mapping->push_back(idx); //record where the data went
			cnt++;	
			}
		}
	}


	for (int i=0; i<numBins; i++)
	{
		delete[] bins[i];
	}

	delete [] bins;

}




//comparison function to sort vector
//sort in increasing order with epsilon and decreasing order with minpts for a given epsilon
bool compareByEpsilon(const experiment &a, const experiment &b)
{
	if (a.epsilon==b.epsilon)
	{
		return a.minpts > b.minpts;
	}
	
	return a.epsilon<b.epsilon;
}






//Sort the data
//The points are sorted based on a binned granularity of 1 degree (can modify this later) 
//For a given x value, the y values will be ordered together
bool compareDataElemStructFunc(const dataElem &elem1, const dataElem &elem2)
{
	//compare based on the x-coodinate
   if ( bin_x(elem1.x) < bin_x(elem2.x)){
    return true;
	}
   else if (bin_x(elem1.x) > bin_x(elem2.x)){
      return false;
  	}
   //if the x-coordinates are equal, compare on the y-coordinate
   else if ( bin_y(elem1.y) < bin_y(elem2.y)){
   	return true;
   	}
   else if (bin_y(elem1.y) > bin_y(elem2.y)){
   	return false;
	}
   	//if they are equal
   	else{
     return true;
	}
}


//calculate the bin for a point, 100 bins
int bin_x(double x)
{
double width_dim=100; //Put in function signature later	
int num_bins=100;
return (ceil((x/width_dim)*num_bins));
}

//calculate the bin for a point, 100 bins
int bin_y(double x)
{
double width_dim=100; //Put in function signature later
int num_bins=100;
return (ceil((x/width_dim)*num_bins));
}








