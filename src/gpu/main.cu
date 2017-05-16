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

#include <cstdlib>
#include <stdio.h>
#include <random>
#include "prototypes.h"
#include "globals.h"
#include "omp.h"
#include "DBScan.h"
#include "schedule.h"
#include <algorithm> 
#include <string.h>
#include <fstream>
#include <iostream>
#include <string>
#include "GPU.h"
#include "kernel.h"
#include "cluster.h"
#include <math.h>
#include <queue>
#include <iomanip>

//#include "structs.h"
using namespace std;





void generateGridDimensions(std::vector<struct dataElem>* dataPoints, double epsilon, double * gridMin_x, double * gridMin_y, int * gridNumXCells, int * gridNumYCells);
void populateGridIndexAndLookupArray(std::vector<struct dataElem>* dataPoints, double epsilon, struct grid * index, unsigned int * lookupArr, double gridMin_x, double gridMin_y, int gridNumXCells, int gridNumYCells, int * numFullCells);
void calcLinearID(struct dataElem * point, struct grid * index, double epsilon, double gridMin_x, double gridMin_y, int gridNumXCells, int gridNumYCells, int * outGridCellIDs, int * outNumGridCells);

//calculates the maximum number of points that will be required by any originating grid cell
unsigned int calcMaxSharedMemDataAware(struct grid * index, int gridNumXCells, int gridNumYCells);


//NOTE:
//IN ALL IMPLEMENTATIONS, WE ONLY INDEX DATA BASED ON 2 DIMENSIONS, LAT AND LONG. HOWEVER, THE TIME, AND TEC VALUES ARE STILL IMPORTED AS DATA ELEMENTS FOR POST-PROCESSING

int main(int argc, char *argv[])
{
	


	/////////////////////////
	// Get information from command line
	//1) the dataset, 2) list of parameter values for experiments.
	//The dataset name is stored in: inputFname, and the experiment instances are stored in a struct containing epsilon and minpts (experimentList)
	/////////////////////////

	//Read in parameters from file:
	//dataset filename and cluster instance file
	if (argc!=3)
	{
	cout <<"\n\nIncorrect number of input parameters.  \nShould be dataset file, and DBScan instance file (outlines the experiments)\n";
	return 0;
	}
	
	//copy parameters from commandline:
	//char inputFname[]="data/test_data_removed_nan.txt";	
	char inputFname[500];
	char inputInstanceName[500];

	strcpy(inputFname,argv[1]);
	strcpy(inputInstanceName,argv[2]);

	printf("\nDataset file: %s",inputFname);
	printf("\nExperiment file: %s",inputInstanceName);

	

	//make a vector that stores the experiments
	std::vector<struct experiment> experimentList;
	importDBScanInstances(&experimentList, inputInstanceName);








	//////////////////////////////
	//import the dataset:
	/////////////////////////////
	std::vector<struct dataElem> dataPoints;
	

	//char inputFname[]="data/test_data_removed_nan.txt";
	importDataset(&dataPoints, inputFname);
	
	


	//sort the data in the following order: x,y,TEC value, time
	//qsort((void *) &dataPoints,dataPoints.size(),sizeof(struct dataElem),(compfn)compareDataElemStructFunc); 
	std::sort(dataPoints.begin(),dataPoints.end(),compareDataElemStructFunc);

	

	
	//experiment parameters (temporary)
	// double * epsilon;
	// epsilon=(double*)malloc(sizeof(double));
	// *epsilon=experimentList[0].epsilon;

	// int * minpts;
	// minpts=(int*)malloc(sizeof(int));
	// *minpts=experimentList[0].minpts;

	



	//////////////////////////////////////
	//GENERATE THE GRID INDEX FOR THE GPU FOR EACH VALUE OF EPSILON
	//////////////////////////////////////

	


	printf("\n\nCOMMENTED ORIGINAL GRID INDEX!!!\n\n");

	/*
	//double GPUGridTstart=omp_get_wtime();
	double gridMin_x=0;
	double gridMin_y=0; 
	int gridNumXCells=0; 
	int gridNumYCells=0;

	//generate grid dimensions:
	generateGridDimensions(&dataPoints, *epsilon, &gridMin_x, &gridMin_y, &gridNumXCells, &gridNumYCells);
	// printf("\n In main: Min x: %f, Min y: %f",gridMin_x,gridMin_y);
	printf("\n In main: Min x: %f, Min y: %f, Num X cells: %d, Num Y cells: %d",gridMin_x,gridMin_y,gridNumXCells,gridNumYCells);
	

	//Allocate grid index and lookup array
	struct grid * index;
	index=new grid[gridNumXCells*gridNumYCells];

	unsigned int * lookupArr;
	lookupArr=new unsigned int[dataPoints.size()];

	
	//number of full cells:
	int * numFullCells;
	numFullCells=(int*)malloc(sizeof(int));
	*numFullCells=0;


	//populate the index
	populateGridIndexAndLookupArray(&dataPoints, *epsilon, index, lookupArr, gridMin_x, gridMin_y, gridNumXCells, gridNumYCells, numFullCells);
	double GPUGridTend=omp_get_wtime();
	double totalGPUGridTtotal=GPUGridTend - GPUGridTstart;



	printf("\ntime to populate the grid and lookup array: %f",totalGPUGridTtotal);
	printf("\nIn main number of full cells: %d", *numFullCells);
	*/



	///////////////
	//BRUTE FORCE GPU
	//NO BATCHING
	///////////////
	#if SEARCHMODE==0


	double tstart_bruteforcegpu=omp_get_wtime();

	//neighbor table:
	table * neighborTable;
	neighborTable=new table[dataPoints.size()];
	int * totalNeighbors;
	totalNeighbors=(int*)malloc(sizeof(int));
	*totalNeighbors=0;

	printf("\nBrute force GPU (NO BATCHING):");
	

	double tstart=omp_get_wtime();
	makeDistanceTableGPUBruteForce(&dataPoints,epsilon, neighborTable, totalNeighbors);
	double tend=omp_get_wtime();
	printf("\nBRUTE FORCE Time on GPU: %f",tend-tstart);cout.flush();
	printf("\nTotal neighbours in table: %d", *totalNeighbors);

	double dbscantstartgpuindex=omp_get_wtime();
	dbscan(neighborTable, dataPoints.size(), *minpts); //5 is minpts
	double dbscantendgpuindex=omp_get_wtime();
	printf("\ntime to dbscan: %f",dbscantendgpuindex-dbscantstartgpuindex);

	double tend_bruteforcegpu=omp_get_wtime();
	printf("\nTotal time GPU brute force: %f", tend_bruteforcegpu - tstart_bruteforcegpu);

	printf("\n*********************************");
	
	#endif

	//////////////
	//END GPU BRUTE FORCE
	//NO BATCHING
	//////////////



	///////////////
	//GLOBAL MEMORY GRID KERNEL
	//NO BATCHING
	///////////////

	#if SEARCHMODE==1

	printf("\nSINGLE EXECUTION OF THE KERNEL (GLOBAL MEMORY) AND DBSCAN.  ONLY USED TO EVALUATE KERNEL PERFORMANCE USING THE PROFILER(no batching).");

	//SET UP THE INDEX:

	double gridMin_x=0;
	double gridMin_y=0; 
	int gridNumXCells=0; 
	int gridNumYCells=0;

	//generate grid dimensions:
	generateGridDimensions(&dataPoints, experimentList[0].epsilon, &gridMin_x, &gridMin_y, &gridNumXCells, &gridNumYCells);
	// printf("\n In main: Min x: %f, Min y: %f",gridMin_x,gridMin_y);
	printf("\n In main: Min x: %f, Min y: %f, Num X cells: %d, Num Y cells: %d",gridMin_x,gridMin_y,gridNumXCells,gridNumYCells);
	

	//Allocate grid index and lookup array
	struct grid * index;
	index=new grid[gridNumXCells*gridNumYCells];

	unsigned int * lookupArr;
	lookupArr=new unsigned int[dataPoints.size()];

	
	//number of full cells:
	int * numFullCells;
	numFullCells=(int*)malloc(sizeof(int));
	*numFullCells=0;


	//populate the index
	double GPUGridTstart=omp_get_wtime();
	populateGridIndexAndLookupArray(&dataPoints, experimentList[0].epsilon, index, lookupArr, gridMin_x, gridMin_y, gridNumXCells, gridNumYCells, numFullCells);
	double GPUGridTend=omp_get_wtime();
	double totalGPUGridTtotal=GPUGridTend - GPUGridTstart;



	printf("\ntime to populate the grid and lookup array: %f",totalGPUGridTtotal);
	printf("\nIn main number of full cells: %d", *numFullCells);


	//END SET UP THE INDEX


	//trials:

	for(int h=0; h<NUM_TRIALS; h++)
	{




		printf("\n\nTRIAL NUM: %d", h);


		double totalGPUTstart=omp_get_wtime();

		int * totalNeighborsGPU;
		totalNeighborsGPU=(int*)malloc(sizeof(int));
		*totalNeighborsGPU=0;		

		table * neighborTable;
		neighborTable=new table[dataPoints.size()];


		printf("\nCalling GPU grid kernel\n WITH NOSHARED MEMORY KERNEL\n Single batch\n");
		
		
		double gpuGridTstart=omp_get_wtime();
		makeDistanceTableGPUGridIndex(&dataPoints, &experimentList[0].epsilon, index, &gridMin_x, &gridMin_y, &gridNumXCells, &gridNumYCells, lookupArr, neighborTable, totalNeighborsGPU);	
		double gpuGridTend=omp_get_wtime();
		
		printf("\nTime to run the GPU Grid table implementation: %f", gpuGridTend - gpuGridTstart);
		
		printf("\nEnd calling GPU grid kernel");

		double dbscantstartgpuindex=omp_get_wtime();
		dbscan(neighborTable, dataPoints.size(), experimentList[0].minpts); //5 is minpts
		double dbscantendgpuindex=omp_get_wtime();
		printf("\ntime to dbscan: %f",dbscantendgpuindex-dbscantstartgpuindex);

		double totalGPUTend=omp_get_wtime();

		printf("\nTime excluding the grid construction and lookup array: %f", totalGPUTend-totalGPUTstart);
		printf("\nGrand total time gpu dbscan with indexing: %f", (totalGPUTend - totalGPUTstart) + totalGPUGridTtotal);

	}


	#endif

	///////////////
	//END GLOBAL MEMORY GRID KERNEL
	//NO BATCHING
	///////////////




	///////////////
	//SHARED MEMORY GRID KERNEL
	//DATA OBLIVIOUS- TILE THE COMPUTATION
	//NO BATCHING
	///////////////
	#if SEARCHMODE==2

	printf("\nSINGLE EXECUTION OF THE KERNEL (SHARED MEMORY) AND DBSCAN.  ONLY USED TO EVALUATE KERNEL PERFORMANCE USING THE PROFILER(no batching).");


	double GPUGridTstart=omp_get_wtime();
	double gridMin_x=0;
	double gridMin_y=0; 
	int gridNumXCells=0; 
	int gridNumYCells=0;

	//generate grid dimensions:
	generateGridDimensions(&dataPoints, experimentList[0].epsilon, &gridMin_x, &gridMin_y, &gridNumXCells, &gridNumYCells);
	// printf("\n In main: Min x: %f, Min y: %f",gridMin_x,gridMin_y);
	printf("\n In main: Min x: %f, Min y: %f, Num X cells: %d, Num Y cells: %d",gridMin_x,gridMin_y,gridNumXCells,gridNumYCells);
	

	//Allocate grid index and lookup array
	struct grid * index;
	index=new grid[gridNumXCells*gridNumYCells];

	unsigned int * lookupArr;
	lookupArr=new unsigned int[dataPoints.size()];

	
	//number of full cells:
	int * numFullCells;
	numFullCells=(int*)malloc(sizeof(int));
	*numFullCells=0;


	//populate the index
	populateGridIndexAndLookupArray(&dataPoints, experimentList[0].epsilon, index, lookupArr, gridMin_x, gridMin_y, gridNumXCells, gridNumYCells, numFullCells);
	double GPUGridTend=omp_get_wtime();
	double totalGPUGridTtotal=GPUGridTend - GPUGridTstart;



	printf("\ntime to populate the grid and lookup array: %f",totalGPUGridTtotal);
	printf("\nIn main number of full cells: %d", *numFullCells);




	

	//trials:

	for(int h=0; h<NUM_TRIALS; h++)
	{




		printf("\n\nTRIAL NUM: %d", h);


		double totalGPUTstart=omp_get_wtime();


		int * totalNeighborsGPU;
		totalNeighborsGPU=(int*)malloc(sizeof(int));
		*totalNeighborsGPU=0;

		table * neighborTable;
		neighborTable=new table[dataPoints.size()];


		printf("\nCalling GPU grid kernel With Shared Memory\n DATA OBLIVIOUS\nEach non-empty grid cell processed by a block!\nSingle Batch\n");
		double gpuGridTstart=omp_get_wtime();
		makeDistanceTableGPUGridIndexWithSMBlockDataOblivious(&dataPoints, &experimentList[0].epsilon, index, numFullCells, &gridMin_x, &gridMin_y, &gridNumXCells, &gridNumYCells, lookupArr, neighborTable, totalNeighborsGPU);
		double gpuGridTend=omp_get_wtime();
		


		printf("\nTime to run the GPU Grid table implementation: %f", gpuGridTend - gpuGridTstart);
		
		printf("\nEnd calling GPU grid kernel");




	double dbscantstartgpuindex=omp_get_wtime();
	dbscan(neighborTable, dataPoints.size(), experimentList[0].minpts); 
	double dbscantendgpuindex=omp_get_wtime();
	printf("\ntime to dbscan: %f",dbscantendgpuindex-dbscantstartgpuindex);


	double totalGPUTend=omp_get_wtime();

	printf("\nTime excluding the grid construction and lookup array: %f", totalGPUTend-totalGPUTstart);
	printf("\nGrand total Time gpu dbscan with indexing: %f", (totalGPUTend - totalGPUTstart) + totalGPUGridTtotal);
	
	}

	

	#endif

	///////////////
	//END SHARED MEMORY GRID KERNEL
	//DATA OBLIVIOUS- TILE THE COMPUTATION
	//NO BATCHING
	///////////////



	///////////////
	//GRID KERNEL GLOBAL MEMORY
	//BATCHING
	///////////////
	#if SEARCHMODE==3

	printf("\nCalling GPU grid kernel\n WITH NOSHARED MEMORY KERNEL\n");
	double totalGPUTstart=omp_get_wtime();

	int * totalNeighborsGPU;
	totalNeighborsGPU=(int*)malloc(sizeof(int));
	*totalNeighborsGPU=0;		

	table * neighborTable;
	neighborTable=new table[dataPoints.size()];

	
	double gpuGridTstart=omp_get_wtime();
	makeDistanceTableGPUGridIndexBatches(&dataPoints,epsilon, index, &gridMin_x, &gridMin_y, &gridNumXCells, &gridNumYCells, lookupArr, neighborTable, totalNeighborsGPU);
	double gpuGridTend=omp_get_wtime();
	
	printf("\nTime to run the GPU Grid table implementation: %f", gpuGridTend - gpuGridTstart);
	
	printf("\nEnd calling GPU grid kernel");

	// printf("\nEXITING early...");
	// return 0;

	double dbscantstartgpuindex=omp_get_wtime();
	dbscan(neighborTable, dataPoints.size(), *minpts); //5 is minpts
	double dbscantendgpuindex=omp_get_wtime();
	printf("\nTime to dbscan: %f",dbscantendgpuindex-dbscantstartgpuindex);

	double totalGPUTend=omp_get_wtime();

	printf("\nTime excluding the grid construction and lookup array: %f", totalGPUTend-totalGPUTstart);
	printf("\nGrand total Time gpu dbscan with indexing: %f", (totalGPUTend - totalGPUTstart) + totalGPUGridTtotal);


	#endif




	///////////////////////////////////////////
	//GENERATING ONE TABLE FROM ANOTHER ON THE GPU
	///////////////////////////////////////////
	#if SEARCHMODE==4


	printf("\nGENERATING ONE TABLE FROM ANOTHER ONE THE GPU\n");
	printf("\nThis is an example using the Single Batch Implementation");

	
	
	//a lookup table that points to the array of neighbors above
	//because we cant use vectors on the GPU
	struct gpulookuptable * gpuLookupArray;
	gpuLookupArray= new gpulookuptable[dataPoints.size()];
	


	//first neighbor table:
	//generate the second table from the first table
	//neighbor table:
	table * neighborTable;
	neighborTable=new table[dataPoints.size()];
	

	int * totalNeighborsGPU;
	totalNeighborsGPU=(int*)malloc(sizeof(int));
	*totalNeighborsGPU=0;		




	
	
	double gpuGridTstart=omp_get_wtime();
	makeDistanceTableGPUGridIndex(&dataPoints,epsilon, index, &gridMin_x, &gridMin_y, &gridNumXCells, &gridNumYCells, lookupArr, neighborTable, totalNeighborsGPU);	
	double gpuGridTend=omp_get_wtime();
	
	printf("\nTime to run the GPU Grid table implementation: %f", gpuGridTend - gpuGridTstart);
	

	double dbscantstartgpuindex=omp_get_wtime();
	dbscan(neighborTable, dataPoints.size(), *minpts); //5 is minpts
	double dbscantendgpuindex=omp_get_wtime();
	printf("\ntime to dbscan: %f",dbscantendgpuindex-dbscantstartgpuindex);

	double totalGPUTend=omp_get_wtime();

	

	


	//second neighbor table:
	//generate the second table from the first table
	//neighbor table:

	printf("\n*****************\nGenerating subsequent table:");
	int * directNeighborArray;
	directNeighborArray=new int[*totalNeighborsGPU];

	generateNeighborArrayForGPU(dataPoints.size(),neighborTable, directNeighborArray, gpuLookupArray);	

	table * neighborTable2;
	neighborTable2=new table[dataPoints.size()];


	double * epsilon2;
	epsilon2=(double*)malloc(sizeof(double));
	*epsilon2=*epsilon/2.0;

	printf("\nHalf of epsilon: %f",*epsilon2);

	double tstart2=omp_get_wtime();
	generateDistanceTableFromPreviousTable(&dataPoints, gpuLookupArray, directNeighborArray, totalNeighborsGPU, epsilon2, neighborTable2);
	double tend2=omp_get_wtime();
	printf("\nTime to create subsequent table for second epsilon value %f: %f",*epsilon2,tend2-tstart2);

	//////////////////////////////////



	//call DBSCAN (from first table- neighborTable):
	double dbscantstart=omp_get_wtime();
	dbscan(neighborTable2, dataPoints.size(), *minpts); //5 is minpts
	double dbscantend=omp_get_wtime();
	printf("\ntime to dbscan: %f",dbscantend-dbscantstart);

	
	



	#endif





	///////////////
	//GRID KERNEL GLOBAL MEMORY
	//BATCHING-- DBSCAN WITH A PREVIOUSLY GENERATED NEIGHBOR TABLE
	//ONE ONE INITIAL TABLE FOR THE LARGEST EPSILON
	//THE REST OF THE VARIANTS CLUSTER FROM THE FIRST ONE BY FILTERING THE CANDIDATES
	///////////////
	#if SEARCHMODE==5

	printf("\nCalling GPU grid kernel\n WITH NOSHARED MEMORY KERNEL\n");
	printf("\n***DBSCAN THAT FILTERS FROM A PREVIOUS NEIGHBORTABLE FOR SMALLER EPSILON***\n");
	double totalGPUTstart=omp_get_wtime();

	int * totalNeighborsGPU;
	totalNeighborsGPU=(int*)malloc(sizeof(int));
	*totalNeighborsGPU=0;		

	table * neighborTable;
	neighborTable=new table[dataPoints.size()];

	

	printf("\nGenerating the NeighborTable for the largest epsilon value: %f", experimentList[0].epsilon);	
	double gpuGridTstart=omp_get_wtime();
	makeDistanceTableGPUGridIndexBatches(&dataPoints, &experimentList[0].epsilon, index, &gridMin_x, &gridMin_y, &gridNumXCells, &gridNumYCells, lookupArr, neighborTable, totalNeighborsGPU);
	double gpuGridTend=omp_get_wtime();
	
	printf("\nTime to run the GPU Grid table implementation: %f", gpuGridTend - gpuGridTstart);
	
	printf("\nEnd calling GPU grid kernel");


	//dbscan all of the instances using the neighbortable
	//the first instance dbscans with the dbscan method because all of the neighbors are direct neighbors and don't
	//need to be filtered
	//The rest need to call DBScan with a filter for the additional points
	int nthreads=std::min((int)experimentList.size(),16);
	omp_set_num_threads(nthreads);
	#pragma omp parallel for 
	for (int i=0; i<experimentList.size(); i++)
	{
		double dbscantstartgpuindex=omp_get_wtime();	
		if (i==0){
			dbscan(neighborTable, dataPoints.size(), experimentList[i].minpts); 
		}
		else
		{
			dbscanWithFilter(&dataPoints, neighborTable, experimentList[i].epsilon, experimentList[i].minpts);		
		}

		double dbscantendgpuindex=omp_get_wtime();	
		printf("\nTime to dbscan: %f with Epsilon: %f, minpts: %d",dbscantendgpuindex-dbscantstartgpuindex, experimentList[i].epsilon, experimentList[i].minpts);
	}
	

	// double dbscantstartgpuindex=omp_get_wtime();
	// dbscan(neighborTable, dataPoints.size(), *minpts); //5 is minpts
	// double dbscantendgpuindex=omp_get_wtime();
	// printf("\nTime to dbscan: %f",dbscantendgpuindex-dbscantstartgpuindex);


	// double * epsilon2;
	// epsilon2=(double*)malloc(sizeof(double));
	// *epsilon2=*epsilon/2.0;

	// printf("\nNow running DBSCAN (with filtering) for half of the epsilon value: %f", *epsilon2);
	// double dbscantstartgpuindex2=omp_get_wtime();
	// dbscanWithFilter(dataPoints, neighborTable, dataPoints.size(), *epsilon2, *minpts);
	// double dbscantendgpuindex2=omp_get_wtime();
	// printf("\nTime to dbscan second: %f",dbscantendgpuindex2-dbscantstartgpuindex2);
	

	double totalGPUTend=omp_get_wtime();

	printf("\nTime excluding the grid construction and lookup array: %f", totalGPUTend-totalGPUTstart);
	printf("\nGrand total Time gpu dbscan with indexing: %f", (totalGPUTend - totalGPUTstart) + totalGPUGridTtotal);


	#endif


	///////////////
	//GRID KERNEL GLOBAL MEMORY
	//BATCHING-- DBSCAN WITH A PREVIOUSLY GENERATED NEIGHBOR TABLE
	//ONE TABLE PER DBSCAN VARIANT
	//EITHER GENERATE THE NEIGHBORTABLE USING THE INDEX, OR FROM A PREVIOUS TABLE
	//DEPRECATED, USING THE NEW NEIGHBORTABLES
	///////////////
	#if SEARCHMODE==6

	printf("\nCalling GPU grid kernel\n WITH NOSHARED MEMORY KERNEL\n");
	printf("\n***DBSCAN THAT GENERATES A NEIGHBORTABLE FOR EACH VARIANT INSTANCE***\n");

	printf("\n*******\nCreating Indexes for each experiment, although we may not use all of them.\n\n");

	double pipeline_GPUGridTstart=omp_get_wtime();

	double * pipeline_gridMin_x;
	pipeline_gridMin_x=new double[experimentList.size()];

	double * pipeline_gridMin_y;
	pipeline_gridMin_y=new double[experimentList.size()]; 

	int * pipeline_gridNumXCells;
	pipeline_gridNumXCells= new int [experimentList.size()]; 
	
	int * pipeline_gridNumYCells;
	pipeline_gridNumYCells=new int [experimentList.size()];

	

	//pointers to the grid indexes for each experiment
	//the memory is allocated in the loop below once then umber of cells have been calculated for each experiment
	struct grid ** pipeline_index=new grid*[experimentList.size()];
	
	
	
	//Allocate lookup array
	unsigned int ** pipeline_lookupArr=new unsigned int*[experimentList.size()];
	for (int i=0; i<experimentList.size(); i++)
	{
	pipeline_lookupArr[i]=new unsigned int[dataPoints.size()];
	}
	
	//number of full cells:
	int * pipeline_numFullCells;
	pipeline_numFullCells=new int[experimentList.size()];

	//initialize:
	for (int i=0; i<experimentList.size(); i++)
	{
		pipeline_numFullCells[i]=0;
	}	

	
	

	for (int i=0; i<experimentList.size(); i++)
	{
	printf("\n**************\nCreating index for the experiment epsilon: %f, minpts: %d ",experimentList[i].epsilon,experimentList[i].minpts);	
	//generate grid dimensions:
	generateGridDimensions(&dataPoints, experimentList[i].epsilon, &pipeline_gridMin_x[i], &pipeline_gridMin_y[i], &pipeline_gridNumXCells[i], &pipeline_gridNumYCells[i]);
	
	//allocate memory for the index now that the number of cells have been calculated.
	unsigned int numcells=pipeline_gridNumXCells[i]*pipeline_gridNumYCells[i];
	pipeline_index[i]=new grid[numcells];

	// printf("\n In main: Min x: %f, Min y: %f",gridMin_x,gridMin_y);
	printf("\n Populating indexes for the pipeline: Min x: %f, Min y: %f, Num X cells: %d, Num Y cells: %d",pipeline_gridMin_x[i],pipeline_gridMin_y[i],pipeline_gridNumXCells[i],pipeline_gridNumYCells[i]);
	//populate the index
	populateGridIndexAndLookupArray(&dataPoints, experimentList[i].epsilon, pipeline_index[i], pipeline_lookupArr[i], pipeline_gridMin_x[i], pipeline_gridMin_y[i], pipeline_gridNumXCells[i], pipeline_gridNumYCells[i], &pipeline_numFullCells[i]);
	printf("\nIn main number of full cells: %d", pipeline_numFullCells[i]);cout.flush();
	}

	double pipeline_GPUGridTend=omp_get_wtime();
	

	double pipeline_totalGPUGridTtotal=pipeline_GPUGridTend - pipeline_GPUGridTstart;

	printf("\ntime to populate the grid and lookup array for all experiments: %f",pipeline_totalGPUGridTtotal);	

	printf("\nEND Creating Indexes for each experiment\nMay not use all of them, depending on if neighbortables are reused\n*********************************");


	

	




	double maxSizeNeighborTable=1.5; //in GiB //the maximum size we allow to generate a neighbortable from a previous one (GiB).
									 //if we don't limit this, then the input size may be larger than the GPUs memory
									//and we don't want to batch BOTH the previous neighborTable AND the resultset	
	
	


	double totalGPUTstart=omp_get_wtime();

	unsigned int * totalNeighborsGPU;
	totalNeighborsGPU= new unsigned int[experimentList.size()];
	

	for (int i=0; i<experimentList.size(); i++)
	{
	totalNeighborsGPU[i]=0;		
	}


	//the neighbor tables for each variant
	table * neighborTable[experimentList.size()];
	for (int i=0; i<experimentList.size(); i++)
	{	
	neighborTable[i]=new table[dataPoints.size()];
	}

	//a lookup table that points to the array of neighbors
	//because we cant use vectors on the GPU
	struct gpulookuptable * gpuLookupArray[experimentList.size()];
	for (int i=0; i<experimentList.size(); i++)
	{
	gpuLookupArray[i]= new gpulookuptable[dataPoints.size()];
	}


	

	//first neighbor table:
	//generate the second table from the first table
	//neighbor table:
	// table * neighborTable;
	// neighborTable=new table[dataPoints.size()];
	

	// int * totalNeighborsGPU;
	// totalNeighborsGPU=(int*)malloc(sizeof(int));
	// *totalNeighborsGPU=0;		

	//for nested parallelism for performing dbscan and neighbortable at the same time
	int NUMTHREADS=2;
	omp_set_num_threads(NUMTHREADS);
	int experimentCnt=0;
	std::queue <int> workQueue;
	std::queue <int> freeQueue;
	int experimentID[NUMTHREADS];

	bool workFinished[experimentList.size()];
	for (int i=0; i<experimentList.size(); i++)
	{
		workFinished[i]=false;
	}

	bool usingTable[experimentList.size()];
	for (int i=0; i<experimentList.size(); i++)
	{
		usingTable[i]=true; //assume all experiments can use a previous table
	}


	// bool workReady[experimentList.size()];
	// for (int i=0; i<experimentList.size(); i++)
	// {
	// 	workReady[i]=false;
	// }
		

	double tstartgpudbscan=omp_get_wtime();


	printf("\nGenerating the NeighborTable for the largest epsilon value: %f", experimentList[0].epsilon);	
	double gpuGridTstart=omp_get_wtime();
	makeDistanceTableGPUGridIndexBatches(&dataPoints, &experimentList[0].epsilon, pipeline_index[0], &pipeline_gridMin_x[0], &pipeline_gridMin_y[0], &pipeline_gridNumXCells[0], &pipeline_gridNumYCells[0], pipeline_lookupArr[0], neighborTable[0], &totalNeighborsGPU[0]);
	workQueue.push(0);
	double gpuGridTend=omp_get_wtime();
	printf("\nTime to run the GPU Grid table implementation for the largest epsilon value: %f", gpuGridTend - gpuGridTstart);
	
	printf("\nEnd calling GPU grid kernel");

	



	printf("\n*****************\nGenerating subsequent tables:");
	
	// dbscan(neighborTable[0], dataPoints.size(), experimentList[0].minpts); 
	//printf("\nDBSCAN then Exiting early");
	//return 0;	

	
	printf("\nomp num threads before parallel: %d", omp_get_num_threads());

	#pragma omp parallel num_threads(4)
	{

		int tid=omp_get_thread_num();
		if (tid==0)
		{
			for (int i=1; i<experimentList.size(); i++)
			{
				double sizeOfPreviousTable=(double)sizeof(int)*(totalNeighborsGPU[i-1])/(1024*1024*1024);
				printf("\nSize of previous table: %f", sizeOfPreviousTable);

				if (sizeOfPreviousTable>maxSizeNeighborTable)
				{
					printf("\n\nSize of direct neighbor array: %f (GiB) is too large!\n Going to execute with the index to avoid batching the direct neighbors as well as results\n\n", sizeOfPreviousTable);	
					makeDistanceTableGPUGridIndexBatches(&dataPoints, &experimentList[i].epsilon, pipeline_index[i], &pipeline_gridMin_x[i], &pipeline_gridMin_y[i], &pipeline_gridNumXCells[i], &pipeline_gridNumYCells[i], pipeline_lookupArr[i], neighborTable[i], &totalNeighborsGPU[i]);

					//note that the table from the previous experiment won't be used so it can be freed	
					usingTable[i-1]=false;
				}
				//execute by generating subsequent neighbortable
				else
				{
					printf("\n\n*******************************\n\nGenerating subsequent table: \n");
					

					int * directNeighborArray;
					directNeighborArray=new int[totalNeighborsGPU[i-1]];
					generateNeighborArrayForGPU(dataPoints.size(),neighborTable[i-1], directNeighborArray, gpuLookupArray[i-1]);	
					generateDistanceTableFromPreviousTableBatches(&dataPoints, gpuLookupArray[i-1], directNeighborArray, &totalNeighborsGPU[i-1], &experimentList[i].epsilon, &experimentList[i-1].epsilon, neighborTable[i], &totalNeighborsGPU[i]);
					
					//Note that the table is no longer being used from the previous experiment
					
					usingTable[i-1]=false;	
					
					


				}

				//lets the threads know there's a table to process by DBSCAN
				workQueue.push(i);
			}


		}
		
		//dbscan here (nested parallelism, if/else around the for loop):

		//double tstartGPUdbscanOnly=omp_get_wtime();

		//nested parallelism to take the neighbortable and then dbscan after the table has been created:
		

		else
		{


				printf("\nOuter TID: %d",omp_get_thread_num());
				printf("\nomp num threads: %d", omp_get_num_threads());


				while (experimentCnt<experimentList.size())
				{
					int tid=omp_get_thread_num();
					experimentID[tid]=-1;
					#pragma omp critical
					{
						if(!workQueue.empty())
						{
							experimentID[tid]=workQueue.front();
							workQueue.pop();
							experimentCnt++;
						}
					} //end critical

					//DBSCAN using the table from the experiment 
					if (experimentID[tid]!=-1)
					{
					printf("\nDBSCAN: tid: %d , experiment: %d ",tid,experimentID[tid]);
					
					double dbscanOnlyStart=omp_get_wtime();
					dbscan(neighborTable[experimentID[tid]], dataPoints.size(), experimentList[experimentID[tid]].minpts);
					double dbscanOnlyEnd=omp_get_wtime();	

					
					

					printf("\nTime just for DBSCAN: %f", dbscanOnlyEnd - dbscanOnlyStart);


					//note that dbscan occured and finished
					
					workFinished[experimentID[tid]]=true;
					freeQueue.push(experimentID[tid]);
					
					}


					//while the threads are waiting for work to do.
					//cleaning up neighborTables that can't be used anymore
					
					if (experimentID[tid]==-1)
					{
						#pragma omp critical
						{	
							if (!freeQueue.empty())
							{
								int freeCandidate=freeQueue.front();
								//Check to see if there is a dependancy on the neighborTable from the candidate table to be freed
								//this means that the following experiment has finished DBSCAN, or doesn't need the table because
								//the table was too large, so it needs to produce the table using the index implementation
								//on the GPU
								//if ((usingTable[freeCandidate]==false) && (workFinished[freeCandidate+1]==true) && ((freeCandidate+1) <experimentList.size()))
								if ((usingTable[freeCandidate]==false) && ((freeCandidate+1) <experimentList.size()))
								{
									printf("\nFreeing neighbortable for experiment: %d",freeCandidate);cout.flush();
									delete[] neighborTable[freeCandidate];
									freeQueue.pop();

								}

							} //end if not empty

						} //end critical

					} //end of cleaning up neighbortables
					

				}
		}

	} //end omp parallel section

	//double tendGPUdbscanOnly=omp_get_wtime();

	//printf("\nTime just for DBSCAN: %f", tendGPUdbscanOnly - tstartGPUdbscanOnly);
	

	double tendgpudbscan=omp_get_wtime();

	printf("\nTotal GPU time: %f", tendgpudbscan - tstartgpudbscan);


	/*
	//OLD:
	int * directNeighborArray;
	directNeighborArray=new int[totalNeighborsGPU[0]];

	double tstartneighborarr=omp_get_wtime();
	generateNeighborArrayForGPU(dataPoints.size(),neighborTable[0], directNeighborArray, gpuLookupArray[0]);	
	double tendneighborarr=omp_get_wtime();
	printf("\nTime to create the neighbor array for the GPU: %f", tendneighborarr - tstartneighborarr);


	double tstart2=omp_get_wtime();

	//generateDistanceTableFromPreviousTable(&dataPoints, gpuLookupArray[0], directNeighborArray, &totalNeighborsGPU[0], &experimentList[1].epsilon, neighborTable[1]);
	 generateDistanceTableFromPreviousTableBatches(&dataPoints, gpuLookupArray[0], directNeighborArray, &totalNeighborsGPU[0], &experimentList[1].epsilon, neighborTable[1]);
	double tend2=omp_get_wtime();
	printf("\nTime to create subsequent table for second epsilon value %f: %f",experimentList[0].epsilon,tend2-tstart2);




	double tstartdbscan=omp_get_wtime();
	dbscan(neighborTable[0], dataPoints.size(), experimentList[0].minpts); 
	dbscan(neighborTable[1], dataPoints.size(), experimentList[1].minpts); 
	double tenddbscan=omp_get_wtime();
	printf("\nTime just for DBSCAN: %f", tenddbscan - tstartdbscan);
	
	

	double pipeline_totalGPUTend=omp_get_wtime();

	printf("\nTime excluding the grid construction and lookup array: %f", pipeline_totalGPUTend-pipeline_totalGPUTstart);
	printf("\nGrand total Time gpu dbscan WITH indexing: %f", (pipeline_totalGPUTend - pipeline_totalGPUTstart) + totalGPUGridTtotal);
	*/

	#endif
	

















	///////////////
	//GRID KERNEL GLOBAL MEMORY
	//BATCHING-- DBSCAN WITH A PREVIOUSLY GENERATED NEIGHBOR TABLE
	//ONE TABLE PER DBSCAN VARIANT
	//EITHER GENERATE THE NEIGHBORTABLE USING THE INDEX, OR FROM A PREVIOUS TABLE
	//WITH NEW NEIGHBORTABLES!!!!
	///////////////
	#if SEARCHMODE==7
	printf("\nTHE NEIGHBOR TABLE METHOD WITHOUT VECTORS!!");
	printf("\nCalling GPU grid kernel\n WITH NOSHARED MEMORY KERNEL\n");
	printf("\n***DBSCAN THAT GENERATES A NEIGHBORTABLE FOR EACH VARIANT INSTANCE***\n");

	printf("\n*******\nCreating Indexes for each experiment, although we may not use all of them.\n\n");


	//for nested parallelism for performing dbscan and neighbortable at the same time
	omp_set_num_threads(NTHREADS);


	

	double * pipeline_gridMin_x;
	pipeline_gridMin_x=new double[experimentList.size()];

	double * pipeline_gridMin_y;
	pipeline_gridMin_y=new double[experimentList.size()]; 

	int * pipeline_gridNumXCells;
	pipeline_gridNumXCells= new int [experimentList.size()]; 
	
	int * pipeline_gridNumYCells;
	pipeline_gridNumYCells=new int [experimentList.size()];

	

	//pointers to the grid indexes for each experiment
	//the memory is allocated in the loop below once then umber of cells have been calculated for each experiment
	struct grid ** pipeline_index=new grid*[experimentList.size()];
	
	
	
	//Allocate lookup array
	unsigned int ** pipeline_lookupArr=new unsigned int*[experimentList.size()];
	for (int i=0; i<experimentList.size(); i++)
	{
	pipeline_lookupArr[i]=new unsigned int[dataPoints.size()];
	}
	
	//number of full cells:
	int * pipeline_numFullCells;
	pipeline_numFullCells=new int[experimentList.size()];

	//initialize:
	for (int i=0; i<experimentList.size(); i++)
	{
		pipeline_numFullCells[i]=0;
	}	

	
	

	for (int i=0; i<experimentList.size(); i++)
	{
	printf("\n**************\nCreating index for the experiment epsilon: %f, minpts: %d ",experimentList[i].epsilon,experimentList[i].minpts);	
	//generate grid dimensions:
	generateGridDimensions(&dataPoints, experimentList[i].epsilon, &pipeline_gridMin_x[i], &pipeline_gridMin_y[i], &pipeline_gridNumXCells[i], &pipeline_gridNumYCells[i]);
	
	//allocate memory for the index now that the number of cells have been calculated.
	unsigned int numcells=pipeline_gridNumXCells[i]*pipeline_gridNumYCells[i];
	pipeline_index[i]=new grid[numcells];

	// printf("\n In main: Min x: %f, Min y: %f",gridMin_x,gridMin_y);
	printf("\n Populating indexes for the pipeline: Min x: %f, Min y: %f, Num X cells: %d, Num Y cells: %d",pipeline_gridMin_x[i],pipeline_gridMin_y[i],pipeline_gridNumXCells[i],pipeline_gridNumYCells[i]);
	//populate the index
	populateGridIndexAndLookupArray(&dataPoints, experimentList[i].epsilon, pipeline_index[i], pipeline_lookupArr[i], pipeline_gridMin_x[i], pipeline_gridMin_y[i], pipeline_gridNumXCells[i], pipeline_gridNumYCells[i], &pipeline_numFullCells[i]);
	printf("\nIn main number of full cells: %d", pipeline_numFullCells[i]);cout.flush();
	}

	double pipeline_GPUGridTend=omp_get_wtime();
	


	printf("\nEND Creating Indexes for each experiment\nMay not use all of them, depending on if neighbortables are reused\n*********************************");


	//START TIMER FOR PREAMBLE STUFF (DOESN'T INCLUDE THE INDEX)	
	//MEMORY ALLOCATION FOR ARRAYS ETC. WE ADD THIS TO THE TOTAL TIME AT THE END
	//NEED TO CLEAR THE MEMORY ALLOCATION BETWEEN TRIALS WHICH WE DO NOT INCLUDE


	double preambleTstart=omp_get_wtime();


	




	double maxSizeNeighborTable=1.5; //in GiB //the maximum size we allow to generate a neighbortable from a previous one (GiB).
									 //if we don't limit this, then the input size may be larger than the GPUs memory
									//and we don't want to batch BOTH the previous neighborTable AND the resultset	
	
	

	unsigned int * totalNeighborsGPU;
	totalNeighborsGPU= new unsigned int[experimentList.size()];
	

	for (int i=0; i<experimentList.size(); i++)
	{
	totalNeighborsGPU[i]=0;		
	}


	//the neighbor tables for each variant- each neighbortable points to an array floating somewhere
	//in memory, which are stored in pointersToNeighbors vector (below)
	//the number of arrays for each experiment will be equal to the number of batches to process it.
	neighborTableLookup * neighborTable[experimentList.size()];
	for (int i=0; i<experimentList.size(); i++)
	{	
	neighborTable[i]=new neighborTableLookup[dataPoints.size()];
	}
	//vector of pointers to arrays in memory containing the neighbors within epsilon of each
	//point in the dataset.
	//the struct has the pointer and the size of each array
	std::vector<struct neighborDataPtrs> pointersToNeighbors[experimentList.size()];






	//a lookup table that points to the array of neighbors
	//because we cant use vectors on the GPU
	//only used when constructing a new neighbortable from an old one
	//DEPRICATED: NOW DO IT IN THE MAIN LOOP, SINCE WE DON'T CONSTRUCT THE NEIGHBORTABLE ALL THE TIME
		
	// struct gpulookuptable * gpuLookupArray[experimentList.size()];
	// for (int i=0; i<experimentList.size(); i++)
	// {
	// gpuLookupArray[i]= new gpulookuptable[dataPoints.size()];
	// }


	
	int experimentCnt=0;
	int experimentID[NTHREADS];

	bool workFinished[experimentList.size()];
	for (int i=0; i<experimentList.size(); i++)
	{
		workFinished[i]=false;
	}

	bool usingTable[experimentList.size()];
	for (int i=0; i<experimentList.size(); i++)
	{
		usingTable[i]=true; //assume all experiments can use a previous table
	}


	// bool workReady[experimentList.size()];
	// for (int i=0; i<experimentList.size(); i++)
	// {
	// 	workReady[i]=false;
	// }
		
	double preambleTend=omp_get_wtime();
	printf("\nPREAMBLE time: %f", preambleTend - preambleTstart);


	double trialtimesTotal[NUM_TRIALS]; //The total time
	double trialtimesDBSCAN[NUM_TRIALS]; //Time just to DBSCAN
	
	double trialtimesGPUInitialTable[NUM_TRIALS]; //Time just for GPU Initial Table
	double trialtimesGPUSubsequentTables[NUM_TRIALS]; //Time just for GPU subsequent tables
	double trialtimesGPUTotalTables[NUM_TRIALS]; //Time for initial GPU AND subsequent tables (the addition of the above two)


	for (int x=0; x<NUM_TRIALS; x++)
	{
				printf("\n\n************\nEXECUTING TRIAL NUM: %d\n************\n",x);

				

				////////////////////////////////////////
				//reset variables for next trial:
				////////////////////////////////////////
				std::queue <int> workQueue;
				std::queue <int> freeQueue;	

				for (int j=0; j<experimentList.size(); j++)
				{
					neighborTable[j]=new neighborTableLookup[dataPoints.size()];
					pointersToNeighbors[j].clear();
					totalNeighborsGPU[j]=0;	
					workFinished[j]=false;	
					usingTable[j]=true;
				}
				
				experimentCnt=0;

					

					
				////////////////////////////////////////
				//end reset variables
				////////////////////////////////////////



				//Times not included in the trials: freeing memory for the neighbortables,
				//resetting whether the work has been completed etc.
	

				double tstartTotalTime=omp_get_wtime();


				printf("\nGenerating the NeighborTable for the largest epsilon value: %f", experimentList[0].epsilon);	
				double gpuGridTstart=omp_get_wtime();
				makeDistanceTableGPUGridIndexBatchesAlternateTest(&dataPoints, &experimentList[0].epsilon, pipeline_index[0], &pipeline_gridMin_x[0], &pipeline_gridMin_y[0], &pipeline_gridNumXCells[0], &pipeline_gridNumYCells[0], pipeline_lookupArr[0], neighborTable[0], &pointersToNeighbors[0], &totalNeighborsGPU[0]);
				workQueue.push(0);
				double gpuGridTend=omp_get_wtime();
				printf("\nTime to run the GPU Grid table implementation for the largest epsilon value: %f", gpuGridTend - gpuGridTstart);
				trialtimesGPUInitialTable[x]=gpuGridTend - gpuGridTstart;


				

				
				
				
				
				printf("\n*****************\nStart of generating subsequent tables:");

				
				printf("\nomp num threads before parallel: %d", omp_get_num_threads());

				#pragma omp parallel num_threads(NTHREADS)
				{

					int tid=omp_get_thread_num();
					if (tid==0)
					{

						double subsequentTablesTstart=omp_get_wtime();
						for (int i=1; i<experimentList.size(); i++)
						{
							double sizeOfPreviousTable=(double)sizeof(int)*(totalNeighborsGPU[i-1])/(1024*1024*1024);
							printf("\nSize of previous table: %f", sizeOfPreviousTable);

							if ((sizeOfPreviousTable>maxSizeNeighborTable) || (TABLEFROMPREVIOUS==0))
							{
								printf("\n\nSize of direct neighbor array: %f (GiB) is too large! OR REUSING A TABLE IS DISABLED\n Going to execute with the index to avoid batching the direct neighbors as well as results\n\n", sizeOfPreviousTable);	
								//printf("\nEpsilon of experiment: %f",experimentList[i].epsilon);
								//makeDistanceTableGPUGridIndexBatchesAlternateTest(&dataPoints, &experimentList[i].epsilon, pipeline_index[i], &pipeline_gridMin_x[i], &pipeline_gridMin_y[i], &pipeline_gridNumXCells[i], &pipeline_gridNumYCells[i], pipeline_lookupArr[i], neighborTable[i], &totalNeighborsGPU[i]);
								makeDistanceTableGPUGridIndexBatchesAlternateTest(&dataPoints, &experimentList[i].epsilon, pipeline_index[i], &pipeline_gridMin_x[i], &pipeline_gridMin_y[i], &pipeline_gridNumXCells[i], &pipeline_gridNumYCells[i], pipeline_lookupArr[i], neighborTable[i], &pointersToNeighbors[i], &totalNeighborsGPU[i]);
								//note that the table from the previous experiment won't be used so it can be freed	
								usingTable[i-1]=false;
							}
							//execute by generating subsequent neighbortable from previous one
							#if TABLEFROMPREVIOUS==1
							else
							{
								printf("\n\n*******************************\n\nGenerating subsequent table from previous one: \n");
								
								//The direct neighbors of all of the data points which are pounted to by
								//the struct array below it, gpuLookupArr
								int * directNeighborArray;
								directNeighborArray=new int[totalNeighborsGPU[i-1]];
								//printf("\nAllocating memory for direct neighbor array: %d",totalNeighborsGPU[i-1]);

								struct gpulookuptable * gpuLookupArr;
								gpuLookupArr= new gpulookuptable[dataPoints.size()];
								
								
								generateNeighborArrayForGPUAlternative(dataPoints.size(),neighborTable[i-1], directNeighborArray, gpuLookupArr);	
								generateDistanceTableFromPreviousTableBatchesAlternate(&dataPoints, gpuLookupArr, directNeighborArray, &totalNeighborsGPU[i-1], &experimentList[i].epsilon, &experimentList[i-1].epsilon, neighborTable[i], &pointersToNeighbors[i], &totalNeighborsGPU[i]);
								
								//Note that the table is no longer being used from the previous experiment
								usingTable[i-1]=false;	
								


							}
							//Note that the table is no longer being used from the previous experiment
							usingTable[i-1]=false;
							#endif	
							//lets the threads know there's a table to process by DBSCAN
							workQueue.push(i);
							printf("\nPushing experiment: %d",i);
						}

						double subsequentTablesTend=omp_get_wtime();
						trialtimesGPUSubsequentTables[x]=subsequentTablesTend - subsequentTablesTstart;


					}
					//nested parallelism to take the neighbortable and then dbscan after the table has been created:
					else
					{


							printf("\nOuter TID: %d",omp_get_thread_num());
							printf("\nomp num threads: %d", omp_get_num_threads());

							
							while (experimentCnt<experimentList.size())
							{
								int tid=omp_get_thread_num();
								experimentID[tid]=-1;
								#pragma omp critical
								{
									if(!workQueue.empty())
									{
										experimentID[tid]=workQueue.front();
										workQueue.pop();
										experimentCnt++;
									}
								} //end critical

								//DBSCAN using the table from the experiment 
								if (experimentID[tid]!=-1)
								{
								
								//dbscan(neighborTable[experimentID[tid]], dataPoints.size(), experimentList[experimentID[tid]].minpts);
								double dbscanOnlyStart=omp_get_wtime();
								
								dbscanAlternate(neighborTable[experimentID[tid]], dataPoints.size(), experimentList[experimentID[tid]].minpts);
								
								double dbscanOnlyEnd=omp_get_wtime();	

								printf("\nDBSCAN: tid: %d , experiment: %d, Time just for DBSCAN: %f ",tid,experimentID[tid],dbscanOnlyEnd - dbscanOnlyStart);
								
								//note that dbscan occured and finished
								
								workFinished[experimentID[tid]]=true;
								freeQueue.push(experimentID[tid]);
								
								}


								
								//while the threads are waiting for work to do.
								//cleaning up neighborTables that can't be used anymore
								
								if (experimentID[tid]==-1)
								{
									#pragma omp critical
									{	
										if (!freeQueue.empty())
										{
											int freeCandidate=freeQueue.front();
											//Check to see if there is a dependancy on the neighborTable from the candidate table to be freed
											//this means that the following experiment has finished DBSCAN, or doesn't need the table because
											//the table was too large, so it needs to produce the table using the index implementation
											//on the GPU
											//if ((usingTable[freeCandidate]==false) && (workFinished[freeCandidate+1]==true) && ((freeCandidate+1) <experimentList.size()))
											if ((usingTable[freeCandidate]==false) && ((freeCandidate+1) <experimentList.size()))
											{
												printf("\nFreeing neighbortable for experiment: %d",freeCandidate);cout.flush();
												//delete the struct containing the pointers to the direct neighbors
												delete[] neighborTable[freeCandidate];
												//delete the arrays themselves pointed to by this struct:
												for (int j=0; j<pointersToNeighbors[freeCandidate].size(); j++)
												{
													//free the individual arrays:
													int * ptr= pointersToNeighbors[freeCandidate][j].dataPtr;
													delete[] ptr;
													//delete [] pointersToNeighbors[freeCandidate][j].dataPtr;	
												}

												freeQueue.pop();

											}

										} //end if not empty

									} //end critical

								} //end of cleaning up neighbortables
								

							} //end of while 
							
					
					} //end of else
					

				} //end omp parallel section
				
				double tendTotalTime=omp_get_wtime();

				trialtimesTotal[x]=tendTotalTime - tstartTotalTime;
				
				
				//total time of the GPU:
				trialtimesGPUTotalTables[x]=trialtimesGPUInitialTable[x]+trialtimesGPUSubsequentTables[x];

				//DBSCAN time:
				trialtimesDBSCAN[x]=trialtimesTotal[x]-trialtimesGPUTotalTables[x];

				printf("\nTrial: %d, Total time: %f", x, trialtimesTotal[x]);
				printf("\nTrial: %d, GPU: Initial table: %f",x, trialtimesGPUInitialTable[x]);
				printf("\nTrial: %d, GPU: Subsequent tables:%f",x, trialtimesGPUSubsequentTables[x]);
				printf("\nTrial: %d, GPU: Total tables:%f",x, trialtimesGPUTotalTables[x]);
				printf("\nTrial: %d, CPU: DBSCAN (Total-GPU):%f",x, trialtimesDBSCAN[x]);

				printf("\nOUTPUT STATISTICS ON THE TRIALS, LIKE MIN/MAX FOR SANITY CHECKS");


				//free any remaining neighbortable data that might be around that wasn't freed above
				//so we dont get a memory leak between trials
				
				while(!freeQueue.empty())
				{
					int freeCandidate=freeQueue.front();
					delete[] neighborTable[freeCandidate];

					for (int j=0; j<pointersToNeighbors[freeCandidate].size(); j++)
					{
						//free the individual arrays:
						int * ptr= pointersToNeighbors[freeCandidate][j].dataPtr;
						delete[] ptr;
						//delete [] pointersToNeighbors[freeCandidate][j].dataPtr;	
					}

					freeQueue.pop();
					printf("\nFreed neighbortable for experiment: %d",freeCandidate);

				}	



	} //END TRIALS LOOP	

	
	
	//OUTPUT STATISTICS:
	//SKIP THE FIRST TRIAL THAT WARMS UP THE GPU

	char fname[]="pipeline_stats.txt";
	ofstream pipelineOut;
	pipelineOut.open(fname,ios::app);	


	//averages:
	double trialtimesTotalAvg=0;
	double trialtimesDBSCANAvg=0;

	double trialtimesGPUInitialTableAvg=0;
	double trialtimesGPUSubsequentTablesAvg=0;
	double trialtimesGPUTotalTablesAvg=0;


	//only output if the number of trials is >1 so division by 0 doesn't occur.
	if (NUM_TRIALS>1)
	{
		for (int i=1; i<NUM_TRIALS; i++)
		{
			trialtimesTotalAvg+=trialtimesTotal[i];	
			trialtimesDBSCANAvg+=trialtimesDBSCAN[i];
			trialtimesGPUInitialTableAvg+=trialtimesGPUInitialTable[i];
			trialtimesGPUSubsequentTablesAvg+=trialtimesGPUSubsequentTables[i];
			trialtimesGPUTotalTablesAvg+=trialtimesGPUTotalTables[i];
		}

			trialtimesTotalAvg=trialtimesTotalAvg/((double)NUM_TRIALS-1.0);
			trialtimesDBSCANAvg=trialtimesDBSCANAvg/((double)NUM_TRIALS-1.0);
			trialtimesGPUInitialTableAvg=trialtimesGPUInitialTableAvg/((double)NUM_TRIALS-1.0);
			trialtimesGPUSubsequentTablesAvg=trialtimesGPUSubsequentTablesAvg/((double)NUM_TRIALS-1.0);
			trialtimesGPUTotalTablesAvg=trialtimesGPUTotalTablesAvg/((double)NUM_TRIALS-1.0);

			
			pipelineOut<<"\nTotal time, DBSCAN time (Total-GPU, with multiple experiments DBSCAN overlaps with GPU time. So this isn't quite DBSCAN time.), GPU Tables time, Initial table time, Subsequent table time (if applicable), Allow Table From Previous, NUM THREADS";
			pipelineOut<<endl<<inputFname<<", "<<inputInstanceName;
			pipelineOut<<endl<<std::setprecision(5)<<trialtimesTotalAvg<<", "<<trialtimesDBSCANAvg<<", "<<trialtimesGPUTotalTablesAvg<<", "<<trialtimesGPUInitialTableAvg<<", "<<trialtimesGPUSubsequentTablesAvg<<", "<<TABLEFROMPREVIOUS<<", "<<NTHREADS;

	


	}
	else
	{
		printf("\nStatistics not output, need NUM_TRIALS>1");
	}


	return 0;


	#endif










///////////////
	//GRID KERNEL GLOBAL MEMORY
	//BATCHING-- DBSCAN WITH A PREVIOUSLY GENERATED NEIGHBOR TABLE
	//USES NEW NEIGHBORTABLES
	//SINGLE TABLE WITH ONE EPSILON, MULTIPLE MINPTS IN PARALLEL
	///////////////
	#if SEARCHMODE==8
	printf("\nNEIGHBOR TABLE METHOD WITHOUT VECTORS!!");
	printf("\nCalling GPU grid kernel\n WITH NOSHARED MEMORY KERNEL\n");
	printf("\n***DBSCAN THAT HAS A SINGLE TABLE, FOR ONE EPSILON AND LOTS OF DIFFERENT MINPTS VALUES\n");

	printf("\n*******\nCreating Index\n\n");





	
	
	//OMP
	omp_set_num_threads(NTHREADS);


	

	double * pipeline_gridMin_x;
	pipeline_gridMin_x=new double[1];

	double * pipeline_gridMin_y;
	pipeline_gridMin_y=new double[1]; 

	int * pipeline_gridNumXCells;
	pipeline_gridNumXCells= new int [1]; 
	
	int * pipeline_gridNumYCells;
	pipeline_gridNumYCells=new int [1];

	

	//pointers to the grid indexes for each experiment
	//the memory is allocated in the loop below once then umber of cells have been calculated for each experiment
	struct grid ** pipeline_index=new grid*[1];
	
	//printf("\nExiting early;"); return 0;	cout.flush();
	
	//Allocate lookup array
	// unsigned int ** pipeline_lookupArr=new unsigned int*[1];
	// for (int i=0; i<experimentList.size(); i++)
	// {
	// pipeline_lookupArr[i]=new unsigned int[dataPoints.size()];
	// }

	unsigned int * pipeline_lookupArr=new unsigned int[dataPoints.size()];
	

	//printf("\n**Size of experiment list: %d", (int)experimentList.size());	cout.flush();
	//printf("\nExperiment list epsilon index 0: %f", experimentList[0].epsilon); cout.flush();
	//printf("\nExiting early;"); return 0;	cout.flush();
	
	//number of full cells:
	int * pipeline_numFullCells;
	pipeline_numFullCells=new int[1];

	//initialize:
	for (int i=0; i<1; i++)
	{
		pipeline_numFullCells[i]=0;
	}	



	
	
	
	printf("\n**************\nCreating index for the experiment epsilon: %f",experimentList[0].epsilon);	
	

	//generate grid dimensions:
	generateGridDimensions(&dataPoints, experimentList[0].epsilon, &pipeline_gridMin_x[0], &pipeline_gridMin_y[0], &pipeline_gridNumXCells[0], &pipeline_gridNumYCells[0]);
	
	//allocate memory for the index now that the number of cells have been calculated.
	unsigned int numcells=pipeline_gridNumXCells[0]*pipeline_gridNumYCells[0];
	pipeline_index[0]=new grid[numcells];




	// printf("\n In main: Min x: %f, Min y: %f",gridMin_x,gridMin_y);
	printf("\n Populating indexes for the pipeline: Min x: %f, Min y: %f, Num X cells: %d, Num Y cells: %d",pipeline_gridMin_x[0],pipeline_gridMin_y[0],pipeline_gridNumXCells[0],pipeline_gridNumYCells[0]);
	//populate the index
	populateGridIndexAndLookupArray(&dataPoints, experimentList[0].epsilon, pipeline_index[0], pipeline_lookupArr, pipeline_gridMin_x[0], pipeline_gridMin_y[0], pipeline_gridNumXCells[0], pipeline_gridNumYCells[0], &pipeline_numFullCells[0]);
	printf("\nIn main number of full cells: %d", pipeline_numFullCells[0]);cout.flush();
	

	double pipeline_GPUGridTend=omp_get_wtime();
	


	printf("\nEND Creating Indexes for ONE epsilon value that gets reused for multiple minpts\n*********************************");


	//START TIMER FOR PREAMBLE STUFF (DOESN'T INCLUDE THE INDEX)	
	//MEMORY ALLOCATION FOR ARRAYS ETC. WE ADD THIS TO THE TOTAL TIME AT THE END
	//NEED TO CLEAR THE MEMORY ALLOCATION BETWEEN TRIALS WHICH WE DO NOT INCLUDE


	double preambleTstart=omp_get_wtime();


	





	unsigned int * totalNeighborsGPU;
	totalNeighborsGPU= new unsigned int[1];
	

	for (int i=0; i<1; i++)
	{
	totalNeighborsGPU[i]=0;		
	}


	//the neighbor tables for each variant- each neighbortable points to an array floating somewhere
	//in memory, which are stored in pointersToNeighbors vector (below)
	//the number of arrays for each experiment will be equal to the number of batches to process it.
	neighborTableLookup * neighborTable[1];
	for (int i=0; i<1; i++)
	{	
	neighborTable[i]=new neighborTableLookup[dataPoints.size()];
	}
	//vector of pointers to arrays in memory containing the neighbors within epsilon of each
	//point in the dataset.
	//the struct has the pointer and the size of each array
	std::vector<struct neighborDataPtrs> pointersToNeighbors[1];






	//a lookup table that points to the array of neighbors
	//because we cant use vectors on the GPU
	//only used when constructing a new neighbortable from an old one
	//DEPRICATED: NOW DO IT IN THE MAIN LOOP, SINCE WE DON'T CONSTRUCT THE NEIGHBORTABLE ALL THE TIME
		
	// struct gpulookuptable * gpuLookupArray[experimentList.size()];
	// for (int i=0; i<experimentList.size(); i++)
	// {
	// gpuLookupArray[i]= new gpulookuptable[dataPoints.size()];
	// }


	
	// int experimentCnt=0;
	// int experimentID[NUMTHREADS];

	// bool workFinished[experimentList.size()];
	// for (int i=0; i<experimentList.size(); i++)
	// {
	// 	workFinished[i]=false;
	// }

	// bool usingTable[experimentList.size()];
	// for (int i=0; i<experimentList.size(); i++)
	// {
	// 	usingTable[i]=true; //assume all experiments can use a previous table
	// }


	// bool workReady[experimentList.size()];
	// for (int i=0; i<experimentList.size(); i++)
	// {
	// 	workReady[i]=false;
	// }
		
	double preambleTend=omp_get_wtime();
	printf("\nPREAMBLE time: %f", preambleTend - preambleTstart);


	double trialtimesTotal[NUM_TRIALS]; //The total time
	double trialtimesDBSCAN[NUM_TRIALS]; //Time just to DBSCAN
	double trialtimesGPUInitialTable[NUM_TRIALS]; //Time just for GPU Initial Table

	for (int x=0; x<NUM_TRIALS; x++)
	{
				printf("\n\n************\nEXECUTING TRIAL NUM: %d\n************\n",x);




				//Times not included in the trials: freeing memory for the neighbortables, since
				//this is related to the trials
				neighborTable[0]=new neighborTableLookup[dataPoints.size()];
				pointersToNeighbors[0].clear();
				totalNeighborsGPU[0]=0;	
				

	

				double tstartTotalTime=omp_get_wtime();


				printf("\nGenerating single NeighborTable for the epsilon value: %f", experimentList[0].epsilon);	
				double gpuGridTstart=omp_get_wtime();
				makeDistanceTableGPUGridIndexBatchesAlternateTest(&dataPoints, &experimentList[0].epsilon, pipeline_index[0], &pipeline_gridMin_x[0], &pipeline_gridMin_y[0], &pipeline_gridNumXCells[0], &pipeline_gridNumYCells[0], pipeline_lookupArr, neighborTable[0], &pointersToNeighbors[0], &totalNeighborsGPU[0]);
				//workQueue.push(0);
				double gpuGridTend=omp_get_wtime();
				printf("\nTime to run the GPU Grid table implementation for the epsilon value common to all DBSCAN instances: %f", gpuGridTend - gpuGridTstart);
				trialtimesGPUInitialTable[x]=gpuGridTend - gpuGridTstart;

				//DBSCAN ALL VARIANTS:
				#pragma omp parallel for num_threads(NTHREADS) schedule(static,1)
				for (int i=0; i<experimentList.size(); i++)
				{
				int tid=omp_get_thread_num();	
				printf("\nDBSCAN experiment: %d, tid: %d",i,tid);	
				dbscanAlternate(neighborTable[0], dataPoints.size(), experimentList[i].minpts);
				}
				
				double tendTotalTime=omp_get_wtime();

				//end total time
				trialtimesTotal[x]=tendTotalTime - tstartTotalTime;

				//DBSCAN time:
				trialtimesDBSCAN[x]=trialtimesTotal[x]-trialtimesGPUInitialTable[x];

				printf("\nTrial: %d, Total time: %f", x, trialtimesTotal[x]);
				printf("\nTrial: %d, GPU: Initial table: %f",x, trialtimesGPUInitialTable[x]);
				printf("\nTrial: %d, CPU: DBSCAN (Total-GPU):%f",x, trialtimesDBSCAN[x]);

				//free memory between trials so we don't get a memory leak:
				//But don't time this part since its related to the trials
				for (int j=0; j<pointersToNeighbors[0].size(); j++)
				{
					//free individual arrays
					int * ptr=pointersToNeighbors[0][j].dataPtr;
					delete[] ptr;
					

				}
				pointersToNeighbors[0].clear();


				// for (int j=0; j<pointersToNeighbors[freeCandidate].size(); j++)
				// 	{
				// 		//free the individual arrays:
				// 		int * ptr= pointersToNeighbors[freeCandidate][j].dataPtr;
				// 		delete[] ptr;
				// 		//delete [] pointersToNeighbors[freeCandidate][j].dataPtr;	
				// 	}
				
				

	} //END TRIALS LOOP	




	//OUTPUT STATISTICS:
	//SKIP THE FIRST TRIAL THAT WARMS UP THE GPU

	char fname[]="reuse_single_table_stats.txt";
	ofstream reuseOut;
	reuseOut.open(fname,ios::app);	


	//averages:
	double trialtimesTotalAvg=0;
	double trialtimesDBSCANAvg=0;

	double trialtimesGPUInitialTableAvg=0;
	
	


	//only output if the number of trials is >1 so division by 0 doesn't occur.
	if (NUM_TRIALS>1)
	{
		for (int i=1; i<NUM_TRIALS; i++)
		{
			trialtimesTotalAvg+=trialtimesTotal[i];	
			trialtimesDBSCANAvg+=trialtimesDBSCAN[i];
			trialtimesGPUInitialTableAvg+=trialtimesGPUInitialTable[i];
			
		}

			trialtimesTotalAvg=trialtimesTotalAvg/((double)NUM_TRIALS-1.0);
			trialtimesDBSCANAvg=trialtimesDBSCANAvg/((double)NUM_TRIALS-1.0);
			trialtimesGPUInitialTableAvg=trialtimesGPUInitialTableAvg/((double)NUM_TRIALS-1.0);
			

			
			reuseOut<<"\nTotal time, DBSCAN time, Initial table time, NUM THREADS";
			reuseOut<<endl<<inputFname<<", "<<inputInstanceName;
			reuseOut<<endl<<std::setprecision(5)<<trialtimesTotalAvg<<", "<<trialtimesDBSCANAvg<<", "<<trialtimesGPUInitialTableAvg<<", "<<NTHREADS;

	

	}
	else
	{
		printf("\nStatistics not output, need NUM_TRIALS>1");
	}



	
	

	return 0;


	#endif










	/*


	///////////////
	//GPU implementation
	///////////////

	#if SEARCHMODE==5
	///////////////////////////////////
	//GPU kernel variables	
	///////////////////////////////////

	


	
	

	
	


	//to create a neighbor table on the GPU using a higher epsilon for a lower epsilon:
	
	//an array that contains all of the neighbors for each point.
	//There is an array of structures that point to this array that denote the indicies in the array
	//that contains the neighbors of each point  
	int * directNeighborArray;
	directNeighborArray=new int[*totalNeighbors];
	
	//a lookup table that points to the array of neighbors above
	//because we cant use vectors on the GPU
	struct gpulookuptable * gpuLookupArray;
	gpuLookupArray= new gpulookuptable[dataPoints.size()];
	generateNeighborArrayForGPU(dataPoints.size(),neighborTable, directNeighborArray, gpuLookupArray);
	
	//makeDistanceTableFromPreviousTable();
	

	//print GPU lookup[ array]
	// printf("\nGPU lookup array *******");
	// for (int i=0; i<dataPoints.size(); i++)
	// {
	// 	printf("\nPoint id: %d In distance: ",i);
	// 	for (int j=gpuLookupArray[i].indexmin; j<=gpuLookupArray[i].indexmax; j++)
	// 	{
	// 		printf("%d, ", directNeighborArray[j]);
	// 	}
	// }

	///////////////////////////////////////////
	//GENERATING ONE TABLE FROM ANOTHER ON THE GPU
	///////////////////////////////////////////
	//second neighbor table:
	//generate the second table from the first table
	//neighbor table:
	table * neighborTable2;
	neighborTable2=new table[dataPoints.size()];


	double * epsilon2;
	epsilon2=(double*)malloc(sizeof(double));
	*epsilon2=*epsilon/2.0;

	double tstart2=omp_get_wtime();
	generateDistanceTableFromPreviousTable(&dataPoints, gpuLookupArray, directNeighborArray, totalNeighbors, epsilon2, neighborTable2);
	double tend2=omp_get_wtime();
	printf("\nTime to create subsequent table for epsilon value %f: %f",*epsilon2,tend2-tstart2);

	//////////////////////////////////



	//call DBSCAN (from first table- neighborTable):
	double dbscantstart=omp_get_wtime();
	dbscan(neighborTable, dataPoints.size(), 5); //5 is minpts
	double dbscantend=omp_get_wtime();
	printf("\ntime to dbscan: %f",dbscantend-dbscantstart);

	printf("\ntotal time for GPU implementation: %f",(tend-tstart)+(dbscantend-dbscantstart));
	



	//***************************
	//Grid implementation
	
	//void generateGrid(std::vector<struct dataElem>* dataPoints, double epsilon, struct grid * index, int * dataLookup, double * gridMin_x, double * gridMin_y, int * gridNumXCells, int * gridNumYCells)
	//struct grid * index; //allocate the grid cells and index min and max in generate grid
	//int * dataLookupArray; //allocate lookup array in generate grid
	
	

	


	// //given a point, get the ids of the adjacent grid cells that contain points
	// int * gridIDs; //these grid ids contain points
	// gridIDs=new int[9];
	// int numAdjCells=0; //the number of adjacent cells from a grid cell that contain points


	// //calcLinearID(&dataPoints[0], index, *epsilon, gridMin_x, gridMin_y, gridNumXCells, gridNumYCells, gridIDs, &numAdjCells);
	// //printf("\nlinear id: %d",id);

	// //test looking up all of the adjacent cell IDs for each point on the CPU:

	// for (int i=0; i<dataPoints.size(); i++)
	// {
	// 	calcLinearID(&dataPoints[i], index, *epsilon, gridMin_x, gridMin_y, gridNumXCells, gridNumYCells, gridIDs, &numAdjCells);
	// 	printf("\ndata point %d, num grid ids: %d, adj cells: ",i, numAdjCells);
	// 	for (int j=0; j<numAdjCells; j++)
	// 	{
	// 		printf("%d, ", gridIDs[j]);
	// 	}
	// }	

	

	
	printf("\n***************************************");

	////////////////////////////////////////////////
	//testing using key value pairs for the results instead of a struct
	//temp limit scope
	{

	int * totalNeighborsGPU;
	totalNeighborsGPU=(int*)malloc(sizeof(int));
	*totalNeighborsGPU=0;		

	table * neighborTable3;
	neighborTable3=new table[dataPoints.size()];



	


	printf("\nCalling GPU grid kernel\n WITH NOSHARED MEMORY KERNEL \nTESTING KEY VALUE\n");
	printf("\nKEY VALUE IMPLEMENTED! NOW TESTING BATCHED EXECUTION\n\n");
	double totalGPUTstart=omp_get_wtime();
	
	
	double gpuGridTstart=omp_get_wtime();
	//original:
	// makeDistanceTableGPUGridIndex(&dataPoints,epsilon, index, &gridMin_x, &gridMin_y, &gridNumXCells, &gridNumYCells, lookupArr, neighborTable3, totalNeighborsGPU);
	
	//with batches:
	makeDistanceTableGPUGridIndexTestBatches(&dataPoints,epsilon, index, &gridMin_x, &gridMin_y, &gridNumXCells, &gridNumYCells, lookupArr, neighborTable3, totalNeighborsGPU);
	
	double gpuGridTend=omp_get_wtime();
	
	printf("\ntime to run the GPU Grid table implementation: %f", gpuGridTend - gpuGridTstart);
	
	printf("\nEnd calling GPU grid kernel");

	// printf("\nEXITING early...");
	// return 0;

	double dbscantstartgpuindex=omp_get_wtime();
	dbscan(neighborTable3, dataPoints.size(), *minpts); //5 is minpts
	double dbscantendgpuindex=omp_get_wtime();
	printf("\ntime to dbscan: %f",dbscantendgpuindex-dbscantstartgpuindex);

	double totalGPUTend=omp_get_wtime();

	printf("\ntime excluding the grid construction and lookup array: %f", totalGPUTend-totalGPUTstart);
	printf("\ngrand total time gpu dbscan with indexing: %f", (totalGPUTend - totalGPUTstart) + totalGPUGridTtotal);


	// printf("\nstarting sleep");
	// #pragma omp barrier

	// system("sleep 3");
	// printf("\nending sleep");

	}

	//////////////////////////////////







	


	#endif
	*/


	#if SEARCHMODE==9


	//GPU test using the grid index to generate the table
	//each grid cell is processed by a block
	//this one is DATA AWARE: we pass in the amount of shared memory required
	
	
		
	printf("\nCalling GPU grid kernel With Shared Memory\n DATA AWARE\nEach non-empty grid cell processed by a block!\n");
	printf("\nNOT VALIDATED, UNFINISHED DUE TO NEEDING TO ACCOMODATE CELLS WITH MORE SHARED MEMORY THAN POSSIBLE TO ALLOCATE!\n");



	double totalGPUTstart=omp_get_wtime();
	
	int * totalNeighborsGPU;
	totalNeighborsGPU=(int*)malloc(sizeof(int));
	*totalNeighborsGPU=0;

	table * neighborTable;
	neighborTable=new table[dataPoints.size()];


	unsigned int maxDataPoints=calcMaxSharedMemDataAware(index, gridNumXCells, gridNumYCells);
	printf("\nThe maximum number of data points overlapping adjacent grid cells is: %d", maxDataPoints);


	
	double gpuGridTstart=omp_get_wtime();

	makeDistanceTableGPUGridIndexWithSMBlockDataAware(&dataPoints,epsilon, index, &gridMin_x, &gridMin_y, &gridNumXCells, &gridNumYCells, lookupArr, neighborTable, totalNeighborsGPU, maxDataPoints);
	double gpuGridTend=omp_get_wtime();
	
	printf("\nTime to run the GPU Grid table implementation: %f", gpuGridTend - gpuGridTstart);
	
	printf("\nEnd calling GPU grid kernel");

	//printf("\nRETURNING EARLY IN MAIN!!!!");
	//return 0;


	double dbscantstartgpuindex=omp_get_wtime();
	dbscan(neighborTable, dataPoints.size(), *minpts); //5 is minpts
	double dbscantendgpuindex=omp_get_wtime();
	printf("\nTime to dbscan: %f",dbscantendgpuindex-dbscantstartgpuindex);

	double totalGPUTend=omp_get_wtime();

	printf("\nTime excluding the grid construction and lookup array: %f", totalGPUTend-totalGPUTstart);
	printf("\nGrand total Time gpu dbscan with indexing: %f", (totalGPUTend - totalGPUTstart) + totalGPUGridTtotal);
	
	
	
	



	#endif



	//TEST THE TOTAL COUNTS OF NEIGHBORS IN CPU PROTOTYPE IMPLEMENTATIONS OF
	//GRIDS VS. BRUTE FORCE

	#if SEARCHMODE==10

	//TESTING CPU VERSION OF THE GRID VS CPU BRUTE FORCE:
	printf("\nPROTOTYPE CPU GRID VS CPU BRUTE FORCE TOTAL NEIGHBORS -- MAY NOT WORK");
	printf("\n*******GRID CPU");

	int * gridIDs; //these grid ids contain points
	gridIDs=new int[9];
	int numAdjCells=0; //the number of adjacent cells from a grid cell that contain points

	unsigned int cpucnt=0;

	std::vector<int> tmpInDist;

	double tstartcpu=omp_get_wtime();	
	for (int i=0; i<dataPoints.size(); i++)
	{
		double x1=dataPoints[i].x;
		double y1=dataPoints[i].y;

		printf("\nPoint id: %d In distance: ",i);
		
		calcLinearID(&dataPoints[i], index, *epsilon, gridMin_x, gridMin_y, gridNumXCells, gridNumYCells, gridIDs, &numAdjCells);

		tmpInDist.clear();

		for (int j=0; j<numAdjCells; j++)
		{
			int gridID=gridIDs[j];

			
			for (int k=index[gridID].indexmin; k<=index[gridID].indexmax; k++)
			{
				int elemid=lookupArr[k];
				double x2=dataPoints[elemid].x;
				double y2=dataPoints[elemid].y;	

				//XXXXXXXXXXXXXX
				//STORE AND SORT THE ONES IN THE DISTANCE
				//XXXXXXXXXXXXXX

					if (sqrt(((x1-x2)*(x1-x2))+((y1-y2)*(y1-y2)))<=(*epsilon))
					{
						//printf("%d, ",elemid);
						tmpInDist.push_back(elemid);
						cpucnt++;
					}
			}

		}

		//print the ones in distance after sorted
		std::sort(tmpInDist.begin(),tmpInDist.end());
		for (int l=0; l<tmpInDist.size(); l++)
		{
			printf("%d, ", tmpInDist[l]);
		}

	}
	double tendcpu=omp_get_wtime();

	printf("\nGrid CPU: count within epsilon: %d", cpucnt);
	printf("\nGrid CPU time: %f",tendcpu-tstartcpu);
	



	//test comparison vs CPU brute force:
	


	
	printf("\n*******CPU BRUTE FORCE");

	unsigned int cpucnt2=0;

	double tstartcpu2=omp_get_wtime();	
	for (int i=0; i<dataPoints.size(); i++)
	{
		double x1=dataPoints[i].x;
		double y1=dataPoints[i].y;

		printf("\nPoint id: %d In distance: ",i);
		
		for (int j=0; j<dataPoints.size(); j++)
		{
			
			double x2=dataPoints[j].x;
			double y2=dataPoints[j].y;
			if (sqrt(((x1-x2)*(x1-x2))+((y1-y2)*(y1-y2)))<=(*epsilon))
			{
				printf("%d, ",j);
				cpucnt2++;
			}
		}
	}
	double tendcpu2=omp_get_wtime();

	printf("\nCPU: count within epsilon: %d", cpucnt2);
	printf("\nCPU time: %f",tendcpu2-tstartcpu2);
	
		
	

	////////////////////////////



	#endif
	



	//////////////////////////
	//END GPU IMPLEMENTATION
	/////////////////////////



		



	printf("\n\n\n");
	return 0;
}





//prototype CPU implementation.

//some of it to be ported to GPU:

//calculates adjacent cell ids from a point
void calcLinearID(struct dataElem * point, struct grid * index, double epsilon, double gridMin_x, double gridMin_y, int gridNumXCells, int gridNumYCells, int * outGridCellIDs, int * outNumGridCells)
{
	int xCellID=(point->x-gridMin_x)/epsilon;
	int yCellID=(point->y-gridMin_y)/epsilon;

	
	int minXCellID=0;
	int maxXCellID=0;
	int minYCellID=0;
	int maxYCellID=0;
	

	//calculate the min and max x and y cell ids by adding and subtracting one from each value
	//deal with exception cases below.

	minXCellID=max(0,xCellID-1);
	maxXCellID=min(xCellID+1,gridNumXCells-1);
	minYCellID=max(0,yCellID-1);
	maxYCellID=min(yCellID+1,gridNumYCells-1);


	//enumerate the cells in 2D, then convert into 1D
	//only store the cells that have data in them

	//IN CUDA DO #pragma unroll on both loops
	int cnt=0;
	for (int i=minYCellID; i<=maxYCellID; i++)
		{
		for (int j=minXCellID; j<=maxXCellID; j++){
			int linearID=(i*gridNumXCells)+j;			
			
			if(index[linearID].indexmin!=-1) 
			{
				outGridCellIDs[cnt]=linearID;
				cnt++;
			} 
		}
	}

	*outNumGridCells=cnt;
}


void populateGridIndexAndLookupArray(std::vector<struct dataElem>* dataPoints, double epsilon, struct grid * index, unsigned int * lookupArr, double gridMin_x, double gridMin_y, int gridNumXCells, int gridNumYCells, int * numFullCells)
{

	/////////////////////////////////
	//Populate grid lookup array
	//and corresponding indicies in the lookup array
	/////////////////////////////////

	printf("\nSize of dataset: %lu", dataPoints->size());

	//contains the indices of the data points in the database that are in each grid cell.
	//int * cellMemberArray=new int[dataPoints->size()];



	//Temp vector that stores the data points inside each cell that are based on the linearized cell IDs (1 dimensional)
	//std::vector<int> gridElemIDs[gridNumXCells*gridNumYCells];

	unsigned int totalCells=(gridNumXCells*gridNumYCells);

	std::vector<unsigned int> * gridElemIDs;
	gridElemIDs = new std::vector<unsigned int>[totalCells];

	

	for (int i=0; i<dataPoints->size(); i++)
	{
		//calculate the linearized cell id in row-major order from the x and y values of the point
		int xCellID=((*dataPoints)[i].x-gridMin_x)/epsilon;
		int yCellID=((*dataPoints)[i].y-gridMin_y)/epsilon;

		unsigned int linearID=(yCellID*gridNumXCells)+xCellID;

		if (linearID > totalCells)
		{
			printf("\n\nERROR Linear ID is: %d\n\n", linearID);
		}

		//printf("\nX cell id: %d, Y cell id: %d, linear ID: %d",xCellID,yCellID,linearID);
		gridElemIDs[linearID].push_back(i);
	}




	

	int cnt=0;
	int cntEmptyCells=0;
	int cntFullCells=0;

	//populate index and lookup array
	for (int i=0; i<totalCells; i++)
	{

		if(gridElemIDs[i].size()!=0)
		{	
			index[i].indexmin=cnt;
			cntFullCells++;
			for (int j=0; j<gridElemIDs[i].size(); j++)
			{
				if (j>((dataPoints->size()-1)))
				{
					printf("\n\n***ERROR Value of a data point is larger than the dataset! %d\n\n", j);
					return;
				}
				lookupArr[cnt]=gridElemIDs[i][j]; //problem is with this line, is it i or j?
				cnt++;
			}
			index[i].indexmax=cnt-1;
		}

		else
		{
			index[i].indexmin=-1;
			index[i].indexmax=-1;			
			cntEmptyCells++;
		}
	}

	// printf("\nExiting grid populate method early!");
	// return;

	printf("\nFull cells: %d (%f, fraction full)",cntFullCells, cntFullCells/double(totalCells));
	printf("\nEmpty cells: %d (%f, fraction empty)",cntEmptyCells, cntEmptyCells/double(totalCells));

	*numFullCells=cntFullCells;


	printf("\nSize of index to be sent to GPU (GiB): %f", (double)sizeof(struct grid)*(totalCells)/(1024.0*1024.0*1024.0));


	//print for testing
	// int count=0;
	// for (int i=0; i<gridNumXCells*gridNumYCells; i++)
	// {
		
	// 	printf("\nLinear id: %d, index min: %d, index max: %d \nids: ", i, index[i].indexmin, index[i].indexmax);
	// 	if (index[i].indexmin!=-1 && index[i].indexmax!=-1)
	// 	{
	// 		for (int j=index[i].indexmin; j<=index[i].indexmax; j++)
	// 		{
	// 			count++;
	// 			printf("%d, ",lookupArr[j]);
	// 		}
	// 	}

	// }

	// printf("\ntest number of data elems: %d", count);





	/////////////////////////////////
	//END Populate grid lookup array
	//and corresponding indicies in the lookup array
	/////////////////////////////////


}


void generateGridDimensions(std::vector<struct dataElem>* dataPoints, double epsilon, double * gridMin_x, double * gridMin_y, int * gridNumXCells, int * gridNumYCells)
{

	printf("\nGenerating grid dimensions.");
	
		

	/////////////////////////////////
	//calculate the min and max points in the x and y dimension
	/////////////////////////////////
	double minPnt_x=(*dataPoints)[0].x;
	double maxPnt_x=(*dataPoints)[0].x;
	double minPnt_y=(*dataPoints)[0].y;
	double maxPnt_y=(*dataPoints)[0].y;





	for (int i=1; i<dataPoints->size(); i++)
	{
		if ((*dataPoints)[i].x<minPnt_x)
		{
			minPnt_x=(*dataPoints)[i].x;
		}

		if ((*dataPoints)[i].x>maxPnt_x)
		{
			maxPnt_x=(*dataPoints)[i].x;
		}

		if ((*dataPoints)[i].y<minPnt_y)
		{
			minPnt_y=(*dataPoints)[i].y;
		}

		if ((*dataPoints)[i].y>maxPnt_y)
		{
			maxPnt_y=(*dataPoints)[i].y;
		}

	}

	printf("\nGrid: Points in dataset: \nMin x,y: %f, %f", minPnt_x, minPnt_y);
	printf("\nGrid: Points in dataset: \nMax x,y: %f, %f", maxPnt_x, maxPnt_y);


	/////////////////////////////////
	//end calculate the min and max points in the x and y dimension
	/////////////////////////////////


	/////////////////////////////////
	//Calculate the start of the grid along the x and y dimensions
	/////////////////////////////////

	//The cell width is epsilon, such that we only check the neighbors
	//The starting point of the cell in the grid is going to be the min points in the
	//x and y dimensions subtract 1.1 epsilon to avoid possible boundary conditions

	
	minPnt_x=minPnt_x-(0.5*epsilon);
	minPnt_y=minPnt_y-(0.5*epsilon);



	//pass out of the function
	*gridMin_x=minPnt_x;
	*gridMin_y=minPnt_y;

	

	//total spatial extent in the x and y dimensions
	double xWidth=maxPnt_x-minPnt_x;
	double yWidth=maxPnt_y-minPnt_y;

	//total cells in x and y dimensions:
	int xCells=ceil(xWidth/epsilon);
	int yCells=ceil(yWidth/epsilon);

	//pass out of the function
	*gridNumXCells=xCells;
	*gridNumYCells=yCells;

	printf("\nGrid: Total x cells, y cells: %d, %d",xCells,yCells);




	/////////////////////////////////
	//End calculate start of the grid along x and y dimensions
	/////////////////////////////////




}


//FOR THE DATA-AWARE IMPLEMENTATION
//CALCULATE THE MAXIMUM AMOUNT OF SHARED MEMORY REQUIRED FOR THE OVERLAPPING
//DATA ELEMENTS IN THE CELLS
unsigned int calcMaxSharedMemDataAware(struct grid * index, int gridNumXCells, int gridNumYCells)
{

	int totalGridCells=gridNumXCells*gridNumYCells;
	

	unsigned int maxOverlappingPoints=0;


	for (int i=0; i<totalGridCells; i++)
	{

	int xCellID=i%gridNumXCells;
	int yCellID=i/gridNumXCells;

	

	int minXCellID=0;
	int maxXCellID=0;
	int minYCellID=0;
	int maxYCellID=0;
	

	//calculate the min and max x and y cell ids by adding and subtracting one from each value
	//deal with exception cases below.

	minXCellID=max(0,xCellID-1);
	maxXCellID=min(xCellID+1,gridNumXCells-1);
	minYCellID=max(0,yCellID-1);
	maxYCellID=min(yCellID+1,gridNumYCells-1);

	
	//enumerate the cells in 2D, then convert into 1D
	//only store the cells that have data in them

	
	int tmpCntPoints=0;
	
	for (int i=minYCellID; i<=maxYCellID; i++){
		for (int j=minXCellID; j<=maxXCellID; j++){
			int linearID=(i*gridNumXCells)+j;			
			
			if(index[linearID].indexmin!=-1) 
			{
				tmpCntPoints+=index[linearID].indexmax-index[linearID].indexmin+1;
			} 
		}
	}
	
	if (tmpCntPoints>maxOverlappingPoints)
	{
		maxOverlappingPoints=tmpCntPoints;
	}


	} //end outer for loop


	return maxOverlappingPoints;

} //end of function









/*
//multiple points per MBB:
//called this a multiple point box (MPB)
void createEntryMBBMultiplePoints(std::vector<dataElem> *dataPoints, std::vector<std::vector<int> > *MPB_ids, MPBRect * dataRectsMPB)
{
	int MPB_cnt=0;
	int dataPointCnt=0;
	for (int i=0; i<(*dataPoints).size(); i+=MBBSIZE){
		
		//create new space in the MSB vector	
		(*MPB_ids).push_back(vector<int>());
		for (int j=0; j<MBBSIZE; j++)
		{



			//don't want to go over the size of the number of datapoints.	
			//the last MBB might have less than MBBSIZE number of points in it	
			//insert the dataElem ID into the vector of vectors
			if (((MPB_cnt*MBBSIZE)+j)<(*dataPoints).size()){	
			(*MPB_ids)[MPB_cnt].push_back((MPB_cnt*MBBSIZE)+j);
			
			//printf("\nMBB count: %d, data point cnt: %d",MPB_cnt, dataPointCnt);
			dataPointCnt++;
			}
		
		}
		
		
		//create a new MBB for the point(s)
		//&(*MPB_ids)[MPB_cnt] is confusing, but it's a pointer to a vector inside the vector that stores the IDs
		//of the data points that are within each MPB
		dataRectsMPB[MPB_cnt].CreateMBB(dataPoints, &(*MPB_ids)[MPB_cnt]);
		

		//printf("\nMBB min: %f,%f,%f,%f, MBB max: %f,%f,%f,%f",dataRectsMPB[MPB_cnt].MBB_min[0],dataRectsMPB[MPB_cnt].MBB_min[1],dataRectsMPB[MPB_cnt].MBB_min[2],dataRectsMPB[MPB_cnt].MBB_min[3],dataRectsMPB[MPB_cnt].MBB_max[0],dataRectsMPB[MPB_cnt].MBB_max[1],dataRectsMPB[MPB_cnt].MBB_max[2],dataRectsMPB[MPB_cnt].MBB_max[3]);
		MPB_cnt++;
	} //end main for loop


	//testing MSBs:
	// for (int i=0; i<(*MSB_ids).size();i++)
	// {
	// 	printf("\nMBB num: %d", i);
	// 	printf("\nMBB dims: min: %f,%f,%f,%f max: %f,%f,%f,%f", dataRectsMSB[i].MBB_min[0],dataRectsMSB[i].MBB_min[1],dataRectsMSB[i].MBB_min[2],dataRectsMSB[i].MBB_min[3],dataRectsMSB[i].MBB_max[0],dataRectsMSB[i].MBB_max[1],dataRectsMSB[i].MBB_max[2],dataRectsMSB[i].MBB_max[3]);
	// 	for (int j=0; j<(*MSB_ids)[i].size();j++)
	// 	{
	// 		int pntid=(*MSB_ids)[i][j];
	// 		printf("\npoint id: %d, point: %f, %f, %f, %f", pntid, (*dataPoints)[pntid].x,(*dataPoints)[pntid].y,(*dataPoints)[pntid].val,(*dataPoints)[pntid].time);
	// 	}
	// }

	printf("\ninserted this many points into the MPBs: %d",dataPointCnt);
	printf("\n Total MPBs created: %d", MPB_cnt);

}	





//create MBBs for R-tree
void createEntryMBBs(std::vector<dataElem> *dataPoints, Rect * dataRects)
{
	for (int i=0; i<(*dataPoints).size(); i++){
		dataRects[i].P1[0]=(*dataPoints)[i].x;
		dataRects[i].P1[1]=(*dataPoints)[i].y;
		// dataRects[i].P1[2]=(*dataPoints)[i].val;
		// dataRects[i].P1[3]=(*dataPoints)[i].time;
		dataRects[i].pid=i;
		dataRects[i].CreateMBB();
	}

}	

void BruteForceFindPoints(std::vector<dataElem> *dataPoints, std::vector<int> *candidateSet)
{
	for (int i=0; i<(*dataPoints).size();i++){

		if((*dataPoints)[i].val>80 && (*dataPoints)[i].x>160 && (*dataPoints)[i].x<180 && (*dataPoints)[i].y>10 && (*dataPoints)[i].y<20)
		{
			(*candidateSet).push_back(i);
		}

	}
}



//create Query MBBs for R-tree
*/



