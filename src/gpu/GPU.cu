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


//precompute direct neighbors with the GPU:

#include <cuda_runtime.h>
#include <cuda.h>
#include "structs.h"
#include <stdio.h>
#include "kernel.h"
#include <math.h>
#include "GPU.h"
#include <algorithm>
#include "omp.h"
#include <queue>

//thrust
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/system/cuda/execution_policy.h> //for streams for thrust (added with Thrust v1.8)

//elements for the result set
//FOR A SINGLE KERNEL INVOCATION
//NOT FOR THE BATCHED ONE
#define BUFFERELEM 300000000 //400000000-original (when removing the data from the device before putting it back for the sort)

//FOR THE BATCHED EXECUTION:
//#define BATCHTOTALELEM 1200000000 //THE TOTAL SIZE ALLOCATED ON THE HOST
//THE NUMBER OF BATCHES AND THE SIZE OF THE BUFFER FOR EACH KERNEL EXECUTION ARE NOT RELATED TO THE TOTAL NUMBER
//OF ELEMENTS (ABOVE).
#define NUMBATCHES 20
#define BATCHBUFFERELEM 100000000 //THE SMALLER SIZE ALLOCATED ON THE DEVICE FOR EACH KERNEL EXECUTION 

#define GPUSTREAMS 3 //number of concurrent gpu streams



using namespace std;





//Uses the grid index to compute the direct neighbor table
//uses shared memory
//each grid cell is processed by a block of threads
//IN THIS ONE, WE PASS INTO THE GPU THE MAXIMUM AMOUNT OF SHARED MEMORY REQUIRED TO STORE THE OVERLAPPING
//DATA ELEMENTS
void makeDistanceTableGPUGridIndexWithSMBlockDataAware(std::vector<struct dataElem> * dataPoints, double * epsilon, struct grid * index, double * gridMin_x, double * gridMin_y, int * gridNumXCells, int * gridNumYCells, int * lookupArr, struct table * neighborTable, int * totalNeighbors, unsigned int maxNumSMDataItems)
{
	//CUDA error code:
	cudaError_t errCode;


	///////////////////////////////////
	//COPY THE DATABASE TO THE GPU
	///////////////////////////////////


	unsigned int * N;
	N=(unsigned int*)malloc(sizeof(unsigned int));
	*N=dataPoints->size();
	
	printf("\n in main GPU method: N is: %u",*N);cout.flush();

	
	//pinned memory for the database:
	struct point * database;
	database=(struct point*)malloc(sizeof(struct point)*(*N));
	//dont use pinned memory for the database, its slower than using cudaMalloc
	//cudaMallocHost((void **) &database, sizeof(struct point)*(*N));


	struct point * dev_database;
	dev_database=(struct point*)malloc(sizeof(struct point)*(*N));

	//allocate memory on device:
	
	errCode=cudaMalloc( (void**)&dev_database, sizeof(struct point)*(*N));

	printf("\n !!in main GPU method: N is: %u",*N);cout.flush();	
		
	if(errCode != cudaSuccess) {
	cout << "\nError: database Got error with code " << errCode << endl; cout.flush(); 
	}


	



	//first, we copy the x and y values from dataPoints to the database
	for (int i=0; i<(*N); i++)
	{
		database[i].x=(*dataPoints)[i].x;
		database[i].y=(*dataPoints)[i].y;
	}



	//printf("\n size of database: %d",N);

	//copy database to the device:
	errCode=cudaMemcpy(dev_database, database, sizeof(struct point)*(*N), cudaMemcpyHostToDevice);	
	if(errCode != cudaSuccess) {
	cout << "\nError: database Got error with code " << errCode << endl; 
	}

	///////////////////////////////////
	//END COPY THE DATABASE TO THE GPU
	///////////////////////////////////




	///////////////////////////////////
	//COPY THE INDEX TO THE GPU
	///////////////////////////////////

	// //test print the index
	// for (int i=0; i<(*gridNumXCells)*(*gridNumYCells); i++)
	// {
	// 	printf("\nCell %d: min: %d, max: %d", i, index[i].indexmin, index[i].indexmax);
	// }


	int totalGridCells=(*gridNumXCells)*(*gridNumYCells);

	struct grid * dev_grid;
	dev_grid=(struct grid*)malloc(sizeof(struct grid)*totalGridCells);

	//allocate memory on device:
	errCode=cudaMalloc( (void**)&dev_grid, sizeof(struct grid)*totalGridCells);
	
		
	if(errCode != cudaSuccess) {
	cout << "\nError: grid index Got error with code " << errCode << endl; cout.flush(); 
	}

	//copy grid index to the device:
	errCode=cudaMemcpy(dev_grid, index, sizeof(struct grid)*totalGridCells, cudaMemcpyHostToDevice);	
	if(errCode != cudaSuccess) {
	cout << "\nError: grid index allocation Got error with code " << errCode << endl; 
	}	


	///////////////////////////////////
	//END COPY THE INDEX TO THE GPU
	///////////////////////////////////


	///////////////////////////////////
	//COPY THE LOOKUP ARRAY TO THE GPU
	///////////////////////////////////

	//test print the lookup array:
	// for (int i=0; i<*N; i++)
	// {
	// 	printf("\nlookup %d: %d",i, lookupArr[i]);
	// }


	int * dev_lookupArr;
	dev_lookupArr=(int*)malloc(sizeof(int)*(*N));

	//allocate memory on device:
	errCode=cudaMalloc( (void**)&dev_lookupArr, sizeof(int)*(*N));

	if(errCode != cudaSuccess) {
	cout << "\nError: lookup array Got error with code " << errCode << endl; cout.flush(); 
	}

	//copy lookup array to the device:
	errCode=cudaMemcpy(dev_lookupArr, lookupArr, sizeof(int)*(*N), cudaMemcpyHostToDevice);	
	if(errCode != cudaSuccess) {
	cout << "\nError: lookup array allocation Got error with code " << errCode << endl; 
	}	


	///////////////////////////////////
	//END COPY THE LOOKUP ARRAY TO THE GPU
	///////////////////////////////////


	///////////////////////////////////
	//COPY GRID DIMENSIONS TO THE GPU
	//THIS INCLUDES THE NUMBER OF CELLS IN EACH X AND Y DIMENSION, 
	//AND THE STARTING POINT IN THE X AND Y DIMENSIONS THAT THE GRID STARTS AT
	///////////////////////////////////

	//The minimum x boundary of the grid:
	//gridMin_x
	double * dev_gridMin_x;
	dev_gridMin_x=(double*)malloc(sizeof( double ));

	//Allocate on the device
	errCode=cudaMalloc((void**)&dev_gridMin_x, sizeof(double));
	if(errCode != cudaSuccess) {
	cout << "\nError: gridMin_x Got error with code " << errCode << endl; 
	}

	errCode=cudaMemcpy( dev_gridMin_x, gridMin_x, sizeof(double), cudaMemcpyHostToDevice );
	if(errCode != cudaSuccess) {
	cout << "\nError: gridMin_x Got error with code " << errCode << endl; 
	}	


	//The minimum y boundary of the grid:
	//gridMin_y
	double * dev_gridMin_y;
	dev_gridMin_y=(double*)malloc(sizeof( double ));

	//Allocate on the device
	errCode=cudaMalloc((void**)&dev_gridMin_y, sizeof(double));
	if(errCode != cudaSuccess) {
	cout << "\nError: gridMin_y Got error with code " << errCode << endl; 
	}

	errCode=cudaMemcpy( dev_gridMin_y, gridMin_y, sizeof(double), cudaMemcpyHostToDevice );
	if(errCode != cudaSuccess) {
	cout << "\nError: gridMin_y Got error with code " << errCode << endl; 
	}	


	//The number of cells in the x dimension:
	//gridNumXCells
	int * dev_gridNumXCells; 
	dev_gridNumXCells=(int*)malloc(sizeof(int));
	*dev_gridNumXCells=0;

	//allocate on the device
	errCode=cudaMalloc((int**)&dev_gridNumXCells, sizeof(int));	
	if(errCode != cudaSuccess) {
	cout << "\nError: dev_gridNumXCells Got error with code " << errCode << endl; 
	}


	errCode=cudaMemcpy( dev_gridNumXCells, gridNumXCells, sizeof(int), cudaMemcpyHostToDevice );
	if(errCode != cudaSuccess) {
	cout << "\nError: dev_gridNumXCells memcpy Got error with code " << errCode << endl; 
	}


	//The number of cells in the y dimension:
	//gridNumYCells
	int * dev_gridNumYCells; 
	dev_gridNumYCells=(int*)malloc(sizeof(int));
	*dev_gridNumYCells=0;

	//allocate on the device
	errCode=cudaMalloc((int**)&dev_gridNumYCells, sizeof(int));	
	if(errCode != cudaSuccess) {
	cout << "\nError: dev_gridNumYCells Got error with code " << errCode << endl; 
	}


	errCode=cudaMemcpy( dev_gridNumYCells, gridNumYCells, sizeof(int), cudaMemcpyHostToDevice );
	if(errCode != cudaSuccess) {
	cout << "\nError: dev_gridNumYCells memcpy Got error with code " << errCode << endl; 
	}



	///////////////////////////////////
	//END COPY GRID DIMENSIONS TO THE GPU
	///////////////////////////////////





	///////////////////////////////////
	//ALLOCATE MEMORY FOR THE RESULT SET
	///////////////////////////////////
	
	//NON-PINNED MEMORY FOR SINGLE KERNEL INVOCATION (NO BATCHING)


	//CHANGING THE RESULTS TO KEY VALUE PAIR SORT, WHICH IS TWO ARRAYS
	//KEY IS THE POINT ID
	//THE VALUE IS THE POINT ID WITHIN THE DISTANCE OF KEY

	int * dev_pointIDKey; //key
	int * dev_pointInDistValue; //value

	int * pointIDKey; //key
	int * pointInDistValue; //value


	errCode=cudaMalloc((void **)&dev_pointIDKey, sizeof(int)*BUFFERELEM);
	if(errCode != cudaSuccess) {
	cout << "CUDA: Got error with code " << errCode << endl; //2 means not enough memory
	}

	errCode=cudaMalloc((void **)&dev_pointInDistValue, sizeof(int)*BUFFERELEM);
	if(errCode != cudaSuccess) {
	cout << "CUDA: Got error with code " << errCode << endl; //2 means not enough memory
	}


	printf("\nmemory requested for results (GiB): %f",(double)(sizeof(int)*2*BUFFERELEM)/(1024*1024*1024));

	double tstartalloc=omp_get_wtime();

	//host result allocation:
	//pinned result set memory for the host
	// cudaMallocHost((void **) &pointIDKey, sizeof(int)*BUFFERELEM);
	// cudaMallocHost((void **) &pointInDistValue, sizeof(int)*BUFFERELEM);



	//PAGED MEMORY ALLOCATION FOR SMALL RESULT SET WITH SINGLE KERNEL EXECUTION?
	pointIDKey=(int*)malloc(sizeof(int)*BUFFERELEM);
	pointInDistValue=(int*)malloc(sizeof(int)*BUFFERELEM);

	double tendalloc=omp_get_wtime();


	//printf("\nTime to allocate pinned memory on the host: %f", tendalloc - tstartalloc);
	printf("\nTime to allocate (non-pinned) memory on the host: %f", tendalloc - tstartalloc);

	///////////////////////////////////
	//END ALLOCATE MEMORY FOR THE RESULT SET
	///////////////////////////////////















	///////////////////////////////////
	//SET OTHER KERNEL PARAMETERS
	///////////////////////////////////

	

	//count values
	unsigned int * cnt;
	cnt=(unsigned int*)malloc(sizeof(unsigned int));
	*cnt=0;

	unsigned int * dev_cnt; 
	dev_cnt=(unsigned int*)malloc(sizeof(unsigned int));
	*dev_cnt=0;

	//allocate on the device
	errCode=cudaMalloc((unsigned int**)&dev_cnt, sizeof(unsigned int));	
	if(errCode != cudaSuccess) {
	cout << "\nError: cnt Got error with code " << errCode << endl; 
	}


	errCode=cudaMemcpy( dev_cnt, cnt, sizeof(unsigned int), cudaMemcpyHostToDevice );
	if(errCode != cudaSuccess) {
	cout << "\nError: dev_cnt memcpy Got error with code " << errCode << endl; 
	}

	
	//Epsilon
	double * dev_epsilon;
	dev_epsilon=(double*)malloc(sizeof( double ));
	//*dev_epsilon=*epsilon;

	//Allocate on the device
	errCode=cudaMalloc((void**)&dev_epsilon, sizeof(double));
	if(errCode != cudaSuccess) {
	cout << "\nError: epsilon Got error with code " << errCode << endl; 
	}

	//epsilon
	errCode=cudaMemcpy( dev_epsilon, epsilon, sizeof(double), cudaMemcpyHostToDevice );
	if(errCode != cudaSuccess) {
	cout << "\nError: epsilon Got error with code " << errCode << endl; 
	}

	
	//size of the database:
	unsigned int * dev_N; 
	dev_N=(unsigned int*)malloc(sizeof( unsigned int ));

	//allocate on the device
	errCode=cudaMalloc((void**)&dev_N, sizeof(unsigned int));
	if(errCode != cudaSuccess) {
	cout << "\nError: N Got error with code " << errCode << endl; 
	}	

	//N (DATASET SIZE)
	errCode=cudaMemcpy( dev_N, N, sizeof(unsigned int), cudaMemcpyHostToDevice );
	if(errCode != cudaSuccess) {
	cout << "\nError: N Got error with code " << errCode << endl; 
	}		


	//THE NUMBER OF THREADS
	//The number of threads is the blocksize * the number of grid cells
	//Therefore, each data item is not assigned to a single thread
	unsigned int * numgputhreads;
	numgputhreads=(unsigned int*)malloc(sizeof(unsigned int));
	*numgputhreads=totalGridCells*BLOCKSIZE;


	unsigned int * dev_numThreads; 
	dev_numThreads=(unsigned int*)malloc(sizeof( unsigned int ));

	//allocate on the device
	errCode=cudaMalloc((void**)&dev_numThreads, sizeof(unsigned int));
	if(errCode != cudaSuccess) {
	cout << "\nError: dev_numThreads Got error with code " << errCode << endl; 
	}	

	//Number of threads
	errCode=cudaMemcpy( dev_numThreads, numgputhreads, sizeof(unsigned int), cudaMemcpyHostToDevice );
	if(errCode != cudaSuccess) {
	cout << "\nError: dev_numThreads Got error with code " << errCode << endl; 
	}		



	//THE AMOUNT OF SHARED MEMORY REQUIRED TO STORE THE OVERLAPPING POINTS OF A GIVEN ORIGIN CELL
	unsigned int * elemsSM;
	elemsSM=(unsigned int*)malloc(sizeof(unsigned int));
	*elemsSM=maxNumSMDataItems;


	unsigned int * dev_elemsSM; 
	dev_elemsSM=(unsigned int*)malloc(sizeof( unsigned int ));

	//allocate on the device
	errCode=cudaMalloc((void**)&dev_elemsSM, sizeof(unsigned int));
	if(errCode != cudaSuccess) {
	cout << "\nError: dev_elemsSM Got error with code " << errCode << endl; 
	}	

	//SHARED MEMORY ELEMENTS
	errCode=cudaMemcpy( dev_elemsSM, elemsSM, sizeof(unsigned int), cudaMemcpyHostToDevice );
	if(errCode != cudaSuccess) {
	cout << "\nError: dev_elemsSM Got error with code " << errCode << endl; 
	}	






	//debug values
	unsigned int * dev_debug1; 
	dev_debug1=(unsigned int *)malloc(sizeof(unsigned int ));
	*dev_debug1=0;

	unsigned int * dev_debug2; 
	dev_debug2=(unsigned int *)malloc(sizeof(unsigned int ));
	*dev_debug2=0;

	unsigned int * debug1; 
	debug1=(unsigned int *)malloc(sizeof(unsigned int ));
	*debug1=0;

	unsigned int * debug2; 
	debug2=(unsigned int *)malloc(sizeof(unsigned int ));
	*debug2=0;



	//allocate on the device
	errCode=cudaMalloc( (unsigned int **)&dev_debug1, sizeof(unsigned int ) );
	if(errCode != cudaSuccess) {
	cout << "\nError: debug1 Got error with code " << errCode << endl; 
	}		
	errCode=cudaMalloc( (unsigned int **)&dev_debug2, sizeof(unsigned int ) );
	if(errCode != cudaSuccess) {
	cout << "\nError: debug2 Got error with code " << errCode << endl; 
	}		

	//set to 0
	//copy debug to device
	errCode=cudaMemcpy( dev_debug1, debug1, sizeof(unsigned int), cudaMemcpyHostToDevice );
	if(errCode != cudaSuccess) {
	cout << "\nError: dev_debug1 Got error with code " << errCode << endl; 
	}

	errCode=cudaMemcpy( dev_debug2, debug2, sizeof(unsigned int), cudaMemcpyHostToDevice );
	if(errCode != cudaSuccess) {
	cout << "\nError: dev_debug2 Got error with code " << errCode << endl; 
	}

	

	

	





	///////////////////////////////////
	//END SET OTHER KERNEL PARAMETERS
	///////////////////////////////////


	


	///////////////////////////////////
	//LAUNCH KERNEL
	///////////////////////////////////

	//the total blocks is the number of grid cells
	const int TOTALBLOCKS=totalGridCells;	
	printf("\ntotal blocks: %d",TOTALBLOCKS);

	//execute kernel	
	//The third parameter in the kernel allocation is for dynamic shared memory.
	//We need shared memory for 3 arrays (2x doubles and 1x int)
	const int SIZE_SM=(2*(*elemsSM)*sizeof(double))+((*elemsSM)*sizeof(int));
	
	printf("\nMemory requested for DYNAMIC shared memory (kb): %f",SIZE_SM/1024.0);

	kernelGridIndexSMBlockDataAware<<< TOTALBLOCKS, BLOCKSIZE, SIZE_SM >>>(dev_numThreads, dev_N, dev_debug1, dev_debug2, dev_epsilon, dev_grid, dev_gridMin_x, dev_gridMin_y, dev_gridNumXCells, dev_gridNumYCells, dev_lookupArr, dev_cnt, dev_database, dev_elemsSM, dev_pointIDKey, dev_pointInDistValue);
	if ( cudaSuccess != cudaGetLastError() ){
    	printf( "\n\nERROR IN KERNEL LAUNCH!\nMIGHT BE TOO MUCH DYNAMIC SHARED MEMORY REQUESTED\n\n" );
    }

    ///////////////////////////////////
	//END LAUNCH KERNEL
	///////////////////////////////////

    

    ///////////////////////////////////
	//GET RESULT SET
	///////////////////////////////////

	//first find the size of the number of results
	errCode=cudaMemcpy( cnt, dev_cnt, sizeof(unsigned int), cudaMemcpyDeviceToHost );
	if(errCode != cudaSuccess) {
	cout << "\nError: getting cnt from GPU Got error with code " << errCode << endl; 
	}
	else
	{
		printf("\nGPU: result set size within epsilon (GPU grid): %d",*cnt);
	}

	/*

	//copy the results, but only transfer the number of results, not the entire buffer
	errCode=cudaMemcpy(results, dev_results, sizeof(struct structresults)*(*cnt), cudaMemcpyDeviceToHost );
	
	if(errCode != cudaSuccess) {
	cout << "\nError: getting results from GPU Got error with code " << errCode << endl; 
	}
	*/
	printf("\nIn block GPU method, Count is: %d",*cnt);


	*totalNeighbors=(*cnt);

	//SORTING FOR TESTING ONLY
	//XXXXXXXXX
	//XXXXXXXXX
	// std::sort(results, results+(*cnt),compResults);
	// printf("\n**** GPU\n");
	// for (int i=0; i<(*cnt); i++)
	// {
	// 	printf("\n%d,%d",results[i].pointID, results[i].pointInDist);
	// }

	//XXXXXXXXX
	//XXXXXXXXX
	//XXXXXXXXX


	//get debug information (optional)
	errCode=cudaMemcpy(debug1, dev_debug1, sizeof(unsigned int), cudaMemcpyDeviceToHost );
	
	if(errCode != cudaSuccess) {
	cout << "\nError: getting debug1 from GPU Got error with code " << errCode << endl; 
	}
	else
	{
		printf("\nDebug1 value: %u",*debug1);
	}

	errCode=cudaMemcpy(debug2, dev_debug2, sizeof(unsigned int), cudaMemcpyDeviceToHost );
	
	if(errCode != cudaSuccess) {
	cout << "\nError: getting debug1 from GPU Got error with code " << errCode << endl; 
	}
	else
	{
		printf("\nDebug2 value: %u",*debug2);
	}	


	///////////////////////////////////
	//END GET RESULT SET
	///////////////////////////////////


	///////////////////////////////////
	//FREE MEMORY FROM THE GPU
	///////////////////////////////////
    //free:
    cudaFree(dev_N);
    cudaFree(dev_numThreads);
	cudaFree(dev_database);
	cudaFree(dev_debug1);
	cudaFree(dev_debug2);
	cudaFree(dev_cnt);
	cudaFree(dev_epsilon);
	//cudaFree(dev_results);
	cudaFree(dev_grid);
	cudaFree(dev_lookupArr);
	cudaFree(dev_gridNumXCells);
	cudaFree(dev_gridNumYCells);
	cudaFree(dev_gridMin_x);
	cudaFree(dev_gridMin_y);

	////////////////////////////////////

	

	////////////////////////////////////
	//SORT THE TABLE DATA ON THE GPU
	//THERE IS NO ORDERING BETWEEN EACH POINT AND THE ONES THAT IT'S WITHIN THE DISTANCE OF
	////////////////////////////////////



	/////////////////////////////
	//ONE PROBLEM WITH NOT TRANSFERING THE RESULT OFF OF THE DEVICE IS THAT
	//YOU CAN'T RESIZE THE RESULTS TO BE THE SIZE OF *CNT
	//SO THEN YOU HAVE POTENTIALLY LOTS OF WASTED SPACE
	/////////////////////////////

	//sort by key with the data already on the device:
	//wrap raw pointer with a device_ptr to use with Thrust functions
	thrust::device_ptr<int> dev_keys_ptr(dev_pointIDKey);
	thrust::device_ptr<int> dev_data_ptr(dev_pointInDistValue);



	try{
	thrust::sort_by_key(dev_keys_ptr, dev_keys_ptr + (*cnt), dev_data_ptr);
	}
	catch(std::bad_alloc &e)
	  {
	    std::cerr << "Ran out of memory while sorting" << std::endl;
	    exit(-1);
	  }
	
	
	//copy the sorted arays back to the host
	thrust::copy(dev_keys_ptr,dev_keys_ptr+(*cnt),pointIDKey);
	thrust::copy(dev_data_ptr,dev_data_ptr+(*cnt),pointInDistValue);
	  


	//free the data on the device
	cudaFree(dev_pointIDKey);
	cudaFree(dev_pointInDistValue);

	
	////////////////////////////////////
	//END SORT THE DATA ON THE GPU
	////////////////////////////////////


	////////////////////////////////////
	//CONSTRUCT TABLE
	////////////////////////////////////	

	double tStartTableConstruct=omp_get_wtime();
	constructNeighborTableKeyValue(pointIDKey, pointInDistValue, neighborTable, cnt);
	double tEndTableConstruct=omp_get_wtime();	
	printf("\nTime constructing table: %f",tEndTableConstruct - tStartTableConstruct);	

	
	////////////////////////////////////
	//END CONSTRUCT TABLE
	////////////////////////////////////	

	






}























//Uses the grid index to compute the direct neighbor table
//uses shared memory
//each grid cell is processed by a block of threads (set at compile time)
void makeDistanceTableGPUGridIndexWithSMBlockDataOblivious(std::vector<struct dataElem> * dataPoints, double * epsilon, struct grid * index, int * numNonEmptyCells, double * gridMin_x, double * gridMin_y, int * gridNumXCells, int * gridNumYCells, unsigned int * lookupArr, struct table * neighborTable, int * totalNeighbors)
{

	double tKernelResultsStart=omp_get_wtime();

	//CUDA error code:
	cudaError_t errCode;


	///////////////////////////////////
	//COPY THE DATABASE TO THE GPU
	///////////////////////////////////


	unsigned int * N;
	N=(unsigned int*)malloc(sizeof(unsigned int));
	*N=dataPoints->size();
	
	printf("\n in main GPU method: N is: %u",*N);cout.flush();

	//pinned memory for the database:
	struct point * database;
	//dont use pinned memory for the database, its slower than using cudaMalloc
	database=(struct point*)malloc(sizeof(struct point)*(*N));
	//cudaMallocHost((void **) &database, sizeof(struct point)*(*N));
	


	struct point * dev_database;
	dev_database=(struct point*)malloc(sizeof(struct point)*(*N));

	//allocate memory on device:
	
	errCode=cudaMalloc( (void**)&dev_database, sizeof(struct point)*(*N));

	printf("\n !!in main GPU method: N is: %u",*N);cout.flush();	
		
	if(errCode != cudaSuccess) {
	cout << "\nError: database Got error with code " << errCode << endl; cout.flush(); 
	}


	



	//first, we copy the x and y values from dataPoints to the database
	for (int i=0; i<(*N); i++)
	{
		database[i].x=(*dataPoints)[i].x;
		database[i].y=(*dataPoints)[i].y;
	}



	//printf("\n size of database: %d",N);

	//copy database to the device:
	errCode=cudaMemcpy(dev_database, database, sizeof(struct point)*(*N), cudaMemcpyHostToDevice);	
	if(errCode != cudaSuccess) {
	cout << "\nError: database Got error with code " << errCode << endl; 
	}

	///////////////////////////////////
	//END COPY THE DATABASE TO THE GPU
	///////////////////////////////////




	///////////////////////////////////
	//COPY THE INDEX TO THE GPU
	///////////////////////////////////

	// //test print the index
	// for (int i=0; i<(*gridNumXCells)*(*gridNumYCells); i++)
	// {
	// 	printf("\nCell %d: min: %d, max: %d", i, index[i].indexmin, index[i].indexmax);
	// }


	int totalGridCells=(*gridNumXCells)*(*gridNumYCells);

	struct grid * dev_grid;
	dev_grid=(struct grid*)malloc(sizeof(struct grid)*totalGridCells);

	//allocate memory on device:
	errCode=cudaMalloc( (void**)&dev_grid, sizeof(struct grid)*totalGridCells);
	
		
	if(errCode != cudaSuccess) {
	cout << "\nError: grid index Got error with code " << errCode << endl; cout.flush(); 
	}

	//copy grid index to the device:
	errCode=cudaMemcpy(dev_grid, index, sizeof(struct grid)*totalGridCells, cudaMemcpyHostToDevice);	
	if(errCode != cudaSuccess) {
	cout << "\nError: grid index allocation Got error with code " << errCode << endl; 
	}	


	///////////////////////////////////
	//END COPY THE INDEX TO THE GPU
	///////////////////////////////////


	///////////////////////////////////
	//COPY THE LOOKUP ARRAY TO THE GPU
	///////////////////////////////////

	//test print the lookup array:
	// for (int i=0; i<*N; i++)
	// {
	// 	printf("\nlookup %d: %d",i, lookupArr[i]);
	// }


	int * dev_lookupArr;
	dev_lookupArr=(int*)malloc(sizeof(int)*(*N));

	//allocate memory on device:
	errCode=cudaMalloc( (void**)&dev_lookupArr, sizeof(int)*(*N));

	if(errCode != cudaSuccess) {
	cout << "\nError: lookup array Got error with code " << errCode << endl; cout.flush(); 
	}

	//copy lookup array to the device:
	errCode=cudaMemcpy(dev_lookupArr, lookupArr, sizeof(int)*(*N), cudaMemcpyHostToDevice);	
	if(errCode != cudaSuccess) {
	cout << "\nError: lookup array allocation Got error with code " << errCode << endl; 
	}	


	///////////////////////////////////
	//END COPY THE LOOKUP ARRAY TO THE GPU
	///////////////////////////////////


	///////////////////////////////////
	//COPY GRID DIMENSIONS TO THE GPU
	//THIS INCLUDES THE NUMBER OF CELLS IN EACH X AND Y DIMENSION, 
	//AND THE STARTING POINT IN THE X AND Y DIMENSIONS THAT THE GRID STARTS AT
	///////////////////////////////////

	//The minimum x boundary of the grid:
	//gridMin_x
	double * dev_gridMin_x;
	dev_gridMin_x=(double*)malloc(sizeof( double ));

	//Allocate on the device
	errCode=cudaMalloc((void**)&dev_gridMin_x, sizeof(double));
	if(errCode != cudaSuccess) {
	cout << "\nError: gridMin_x Got error with code " << errCode << endl; 
	}

	errCode=cudaMemcpy( dev_gridMin_x, gridMin_x, sizeof(double), cudaMemcpyHostToDevice );
	if(errCode != cudaSuccess) {
	cout << "\nError: gridMin_x Got error with code " << errCode << endl; 
	}	


	//The minimum y boundary of the grid:
	//gridMin_y
	double * dev_gridMin_y;
	dev_gridMin_y=(double*)malloc(sizeof( double ));

	//Allocate on the device
	errCode=cudaMalloc((void**)&dev_gridMin_y, sizeof(double));
	if(errCode != cudaSuccess) {
	cout << "\nError: gridMin_y Got error with code " << errCode << endl; 
	}

	errCode=cudaMemcpy( dev_gridMin_y, gridMin_y, sizeof(double), cudaMemcpyHostToDevice );
	if(errCode != cudaSuccess) {
	cout << "\nError: gridMin_y Got error with code " << errCode << endl; 
	}	


	//The number of cells in the x dimension:
	//gridNumXCells
	int * dev_gridNumXCells; 
	dev_gridNumXCells=(int*)malloc(sizeof(int));
	*dev_gridNumXCells=0;

	//allocate on the device
	errCode=cudaMalloc((int**)&dev_gridNumXCells, sizeof(int));	
	if(errCode != cudaSuccess) {
	cout << "\nError: dev_gridNumXCells Got error with code " << errCode << endl; 
	}


	errCode=cudaMemcpy( dev_gridNumXCells, gridNumXCells, sizeof(int), cudaMemcpyHostToDevice );
	if(errCode != cudaSuccess) {
	cout << "\nError: dev_gridNumXCells memcpy Got error with code " << errCode << endl; 
	}


	//The number of cells in the y dimension:
	//gridNumYCells
	int * dev_gridNumYCells; 
	dev_gridNumYCells=(int*)malloc(sizeof(int));
	*dev_gridNumYCells=0;

	//allocate on the device
	errCode=cudaMalloc((int**)&dev_gridNumYCells, sizeof(int));	
	if(errCode != cudaSuccess) {
	cout << "\nError: dev_gridNumYCells Got error with code " << errCode << endl; 
	}


	errCode=cudaMemcpy( dev_gridNumYCells, gridNumYCells, sizeof(int), cudaMemcpyHostToDevice );
	if(errCode != cudaSuccess) {
	cout << "\nError: dev_gridNumYCells memcpy Got error with code " << errCode << endl; 
	}



	///////////////////////////////////
	//END COPY GRID DIMENSIONS TO THE GPU
	///////////////////////////////////





	///////////////////////////////////
	//ALLOCATE MEMORY FOR THE RESULT SET
	///////////////////////////////////

	//NON-PINNED MEMORY FOR SINGLE KERNEL INVOCATION (NO BATCHING)


	//CHANGING THE RESULTS TO KEY VALUE PAIR SORT, WHICH IS TWO ARRAYS
	//KEY IS THE POINT ID
	//THE VALUE IS THE POINT ID WITHIN THE DISTANCE OF KEY

	int * dev_pointIDKey; //key
	int * dev_pointInDistValue; //value

	int * pointIDKey; //key
	int * pointInDistValue; //value


	errCode=cudaMalloc((void **)&dev_pointIDKey, sizeof(int)*BUFFERELEM);
	if(errCode != cudaSuccess) {
	cout << "CUDA: Got error with code " << errCode << endl; //2 means not enough memory
	}

	errCode=cudaMalloc((void **)&dev_pointInDistValue, sizeof(int)*BUFFERELEM);
	if(errCode != cudaSuccess) {
	cout << "CUDA: Got error with code " << errCode << endl; //2 means not enough memory
	}


	printf("\nmemory requested for results (GiB): %f",(double)(sizeof(int)*2*BUFFERELEM)/(1024*1024*1024));

	double tstartalloc=omp_get_wtime();

	//host result allocation:
	//pinned result set memory for the host
	// cudaMallocHost((void **) &pointIDKey, sizeof(int)*BUFFERELEM);
	// cudaMallocHost((void **) &pointInDistValue, sizeof(int)*BUFFERELEM);



	//PAGED MEMORY ALLOCATION FOR SMALL RESULT SET WITH SINGLE KERNEL EXECUTION?
	pointIDKey=(int*)malloc(sizeof(int)*BUFFERELEM);
	pointInDistValue=(int*)malloc(sizeof(int)*BUFFERELEM);

	double tendalloc=omp_get_wtime();


	//printf("\nTime to allocate pinned memory on the host: %f", tendalloc - tstartalloc);
	printf("\nTime to allocate (non-pinned) memory on the host: %f", tendalloc - tstartalloc);

	///////////////////////////////////
	//END ALLOCATE MEMORY FOR THE RESULT SET
	///////////////////////////////////















	///////////////////////////////////
	//SET OTHER KERNEL PARAMETERS
	///////////////////////////////////

	

	//count values
	unsigned int * cnt;
	cnt=(unsigned int*)malloc(sizeof(unsigned int));
	*cnt=0;

	unsigned int * dev_cnt; 
	dev_cnt=(unsigned int*)malloc(sizeof(unsigned int));
	*dev_cnt=0;

	//allocate on the device
	errCode=cudaMalloc((unsigned int**)&dev_cnt, sizeof(unsigned int));	
	if(errCode != cudaSuccess) {
	cout << "\nError: cnt Got error with code " << errCode << endl; 
	}


	errCode=cudaMemcpy( dev_cnt, cnt, sizeof(unsigned int), cudaMemcpyHostToDevice );
	if(errCode != cudaSuccess) {
	cout << "\nError: dev_cnt memcpy Got error with code " << errCode << endl; 
	}

	
	//Epsilon
	double * dev_epsilon;
	dev_epsilon=(double*)malloc(sizeof( double ));
	//*dev_epsilon=*epsilon;

	//Allocate on the device
	errCode=cudaMalloc((void**)&dev_epsilon, sizeof(double));
	if(errCode != cudaSuccess) {
	cout << "\nError: epsilon Got error with code " << errCode << endl; 
	}

	//epsilon
	errCode=cudaMemcpy( dev_epsilon, epsilon, sizeof(double), cudaMemcpyHostToDevice );
	if(errCode != cudaSuccess) {
	cout << "\nError: epsilon Got error with code " << errCode << endl; 
	}

	
	//size of the database:
	unsigned int * dev_N; 
	dev_N=(unsigned int*)malloc(sizeof( unsigned int ));

	//allocate on the device
	errCode=cudaMalloc((void**)&dev_N, sizeof(unsigned int));
	if(errCode != cudaSuccess) {
	cout << "\nError: N Got error with code " << errCode << endl; 
	}	

	//N (DATASET SIZE)
	errCode=cudaMemcpy( dev_N, N, sizeof(unsigned int), cudaMemcpyHostToDevice );
	if(errCode != cudaSuccess) {
	cout << "\nError: N Got error with code " << errCode << endl; 
	}		


	//THE NUMBER OF THREADS
	//The number of threads is the blocksize * the number of non-empty grid cells
	//Therefore, each data item is not assigned to a single thread
	unsigned int * numgputhreads;
	numgputhreads=(unsigned int*)malloc(sizeof(unsigned int));
	*numgputhreads=(*numNonEmptyCells)*BLOCKSIZE;


	unsigned int * dev_numThreads; 
	dev_numThreads=(unsigned int*)malloc(sizeof( unsigned int ));

	//allocate on the device
	errCode=cudaMalloc((void**)&dev_numThreads, sizeof(unsigned int));
	if(errCode != cudaSuccess) {
	cout << "\nError: dev_numThreads Got error with code " << errCode << endl; 
	}	

	//Number of threads
	errCode=cudaMemcpy( dev_numThreads, numgputhreads, sizeof(unsigned int), cudaMemcpyHostToDevice );
	if(errCode != cudaSuccess) {
	cout << "\nError: dev_numThreads Got error with code " << errCode << endl; 
	}		



	//debug values
	unsigned int * dev_debug1; 
	dev_debug1=(unsigned int *)malloc(sizeof(unsigned int ));
	*dev_debug1=0;

	unsigned int * dev_debug2; 
	dev_debug2=(unsigned int *)malloc(sizeof(unsigned int ));
	*dev_debug2=0;

	unsigned int * debug1; 
	debug1=(unsigned int *)malloc(sizeof(unsigned int ));
	*debug1=0;

	unsigned int * debug2; 
	debug2=(unsigned int *)malloc(sizeof(unsigned int ));
	*debug2=0;



	//allocate on the device
	errCode=cudaMalloc( (unsigned int **)&dev_debug1, sizeof(unsigned int ) );
	if(errCode != cudaSuccess) {
	cout << "\nError: debug1 Got error with code " << errCode << endl; 
	}		
	errCode=cudaMalloc( (unsigned int **)&dev_debug2, sizeof(unsigned int ) );
	if(errCode != cudaSuccess) {
	cout << "\nError: debug2 Got error with code " << errCode << endl; 
	}		

	//set to 0
	//copy debug to device
	errCode=cudaMemcpy( dev_debug1, debug1, sizeof(unsigned int), cudaMemcpyHostToDevice );
	if(errCode != cudaSuccess) {
	cout << "\nError: dev_debug1 Got error with code " << errCode << endl; 
	}

	errCode=cudaMemcpy( dev_debug2, debug2, sizeof(unsigned int), cudaMemcpyHostToDevice );
	if(errCode != cudaSuccess) {
	cout << "\nError: dev_debug2 Got error with code " << errCode << endl; 
	}






	////////////////////////////////////////////
	//the schedule
	//an array that tells each block what grid id to process
	//that way we only request the number of blocks that correspond to the number of non-empty cells

	unsigned int * schedule;
	schedule=(unsigned int*)malloc(sizeof(unsigned int)*(*numNonEmptyCells));
	
	int nonemptycnt=0;
	for (int i=0; i<totalGridCells; i++)
	{
		if (index[i].indexmin!=-1)
		{
			schedule[nonemptycnt]=i;
			nonemptycnt++;
		}
	}

	unsigned int * dev_schedule; 
	dev_schedule=(unsigned int*)malloc(sizeof(unsigned int)*(*numNonEmptyCells));

	//allocate on the device
	errCode=cudaMalloc((void**)&dev_schedule, sizeof(unsigned int)*(*numNonEmptyCells));
	if(errCode != cudaSuccess) {
	cout << "\nError: dev_schedule Got error with code " << errCode << endl; 
	}	


	//copy the schedule
	errCode=cudaMemcpy( dev_schedule, schedule, sizeof(unsigned int)*(*numNonEmptyCells), cudaMemcpyHostToDevice );
	if(errCode != cudaSuccess) {
	cout << "\nError: dev_schedule Got error with code " << errCode << endl; 
	}		


	////////////////////////////
	//END THE SCHEDULE
	////////////////////////////



	



	




	///////////////////////////////////
	//END SET OTHER KERNEL PARAMETERS
	///////////////////////////////////


	


	///////////////////////////////////
	//LAUNCH KERNEL
	///////////////////////////////////

	//the total blocks is the number of grid cells
	const int TOTALBLOCKS=(*numNonEmptyCells);	
	printf("\ntotal blocks: %d",TOTALBLOCKS);

	//execute kernel	
	
	kernelGridIndexSMBlock<<< TOTALBLOCKS, BLOCKSIZE >>>(dev_numThreads, dev_N, dev_debug1, dev_debug2, dev_epsilon, dev_grid, dev_gridMin_x, dev_gridMin_y, dev_gridNumXCells, dev_gridNumYCells, dev_lookupArr, dev_cnt, dev_database, dev_schedule, dev_pointIDKey, dev_pointInDistValue);
	if ( cudaSuccess != cudaGetLastError() ){
    	printf( "Error in kernel launch!\n" );
    }

    ///////////////////////////////////
	//END LAUNCH KERNEL
	///////////////////////////////////

    

    ///////////////////////////////////
	//GET RESULT SET
	///////////////////////////////////

	//first find the size of the number of results
	errCode=cudaMemcpy( cnt, dev_cnt, sizeof(unsigned int), cudaMemcpyDeviceToHost );
	if(errCode != cudaSuccess) {
	cout << "\nError: getting cnt from GPU Got error with code " << errCode << endl; 
	}
	else
	{
		printf("\nGPU: result set size within epsilon (GPU grid): %d",*cnt);
	}

	//copy the results, but only transfer the number of results, not the entire buffer
	/*
	errCode=cudaMemcpy(results, dev_results, sizeof(struct structresults)*(*cnt), cudaMemcpyDeviceToHost );
	
	if(errCode != cudaSuccess) {
	cout << "\nError: getting results from GPU Got error with code " << errCode << endl; 
	}
	*/

	printf("\nIn block GPU method, Count is: %d",*cnt);
	

	*totalNeighbors=(*cnt);


	double tKernelResultsEnd=omp_get_wtime();

	printf("\nTime to launch kernel and execute all of the previous part of the method and get the results back: %f",tKernelResultsEnd-tKernelResultsStart);


	//SORTING FOR TESTING ONLY
	//XXXXXXXXX
	//XXXXXXXXX
	// std::sort(results, results+(*cnt),compResults);
	// printf("\n**** GPU\n");
	// for (int i=0; i<(*cnt); i++)
	// {
	// 	printf("\n%d,%d",results[i].pointID, results[i].pointInDist);
	// }

	//XXXXXXXXX
	//XXXXXXXXX
	//XXXXXXXXX




	errCode=cudaMemcpy(debug1, dev_debug1, sizeof(unsigned int), cudaMemcpyDeviceToHost );
	
	if(errCode != cudaSuccess) {
	cout << "\nError: getting debug1 from GPU Got error with code " << errCode << endl; 
	}
	else
	{
		printf("\nDebug1 value: %u",*debug1);
	}

	errCode=cudaMemcpy(debug2, dev_debug2, sizeof(unsigned int), cudaMemcpyDeviceToHost );
	
	if(errCode != cudaSuccess) {
	cout << "\nError: getting debug1 from GPU Got error with code " << errCode << endl; 
	}
	else
	{
		printf("\nDebug2 value: %u",*debug2);
	}	


	///////////////////////////////////
	//END GET RESULT SET
	///////////////////////////////////


	///////////////////////////////////
	//FREE MEMORY FROM THE GPU
	///////////////////////////////////
    //free:
	


	cudaFree(dev_N);
	cudaFree(dev_database);
	cudaFree(dev_debug1);
	cudaFree(dev_debug2);
	cudaFree(dev_cnt);
	cudaFree(dev_epsilon);
	// cudaFree(dev_results);
	cudaFree(dev_grid);
	cudaFree(dev_lookupArr);
	cudaFree(dev_gridNumXCells);
	cudaFree(dev_gridNumYCells);
	cudaFree(dev_gridMin_x);
	cudaFree(dev_gridMin_y);
	cudaFree(dev_numThreads);
	cudaFree(dev_schedule);



	////////////////////////////////////

	

	////////////////////////////////////
	//SORT THE TABLE DATA ON THE GPU
	//THERE IS NO ORDERING BETWEEN EACH POINT AND THE ONES THAT IT'S WITHIN THE DISTANCE OF
	////////////////////////////////////

	/////////////////////////////
	//ONE PROBLEM WITH NOT TRANSFERING THE RESULT OFF OF THE DEVICE IS THAT
	//YOU CAN'T RESIZE THE RESULTS TO BE THE SIZE OF *CNT
	//SO THEN YOU HAVE POTENTIALLY LOTS OF WASTED SPACE
	/////////////////////////////

	//sort by key with the data already on the device:
	//wrap raw pointer with a device_ptr to use with Thrust functions
	thrust::device_ptr<int> dev_keys_ptr(dev_pointIDKey);
	thrust::device_ptr<int> dev_data_ptr(dev_pointInDistValue);

	


	try{
	thrust::sort_by_key(dev_keys_ptr, dev_keys_ptr + (*cnt), dev_data_ptr);
	}
	catch(std::bad_alloc &e)
	  {
	    std::cerr << "Ran out of memory while sorting" << std::endl;
	    exit(-1);
	  }
	
	


	//copy the sorted arays back to the host
	thrust::copy(dev_keys_ptr,dev_keys_ptr+(*cnt),pointIDKey);
	thrust::copy(dev_data_ptr,dev_data_ptr+(*cnt),pointInDistValue);
	  
	

	//free the data on the device
	cudaFree(dev_pointIDKey);
	cudaFree(dev_pointInDistValue);

	
	////////////////////////////////////
	//END SORT THE DATA ON THE GPU
	////////////////////////////////////


	////////////////////////////////////
	//CONSTRUCT TABLE
	////////////////////////////////////	

	double tStartTableConstruct=omp_get_wtime();
	constructNeighborTableKeyValue(pointIDKey, pointInDistValue, neighborTable, cnt);
	double tEndTableConstruct=omp_get_wtime();	
	printf("\nTime constructing table: %f",tEndTableConstruct - tStartTableConstruct);	

	
	////////////////////////////////////
	//END CONSTRUCT TABLE
	////////////////////////////////////	

	






}


































//Uses the grid index to compute the direct neighbor table
//NO SHARED MEMORY PAGING
void makeDistanceTableGPUGridIndex(std::vector<struct dataElem> * dataPoints, double * epsilon, struct grid * index, double * gridMin_x, double * gridMin_y, int * gridNumXCells, int * gridNumYCells, unsigned int * lookupArr, struct table * neighborTable, int * totalNeighbors)
{

	double tKernelResultsStart=omp_get_wtime();

	//CUDA error code:
	cudaError_t errCode;


	///////////////////////////////////
	//COPY THE DATABASE TO THE GPU
	///////////////////////////////////


	unsigned int * N;
	N=(unsigned int*)malloc(sizeof(unsigned int));
	*N=dataPoints->size();
	
	printf("\n in main GPU method: N is: %u",*N);cout.flush();

	struct point * database;
	
	//pinned memory for the database:
	database=(struct point*)malloc(sizeof(struct point)*(*N));
	//dont use pinned memory for the database, its slower than using cudaMalloc
	//cudaMallocHost((void **) &database, sizeof(struct point)*(*N));


	struct point * dev_database;
	dev_database=(struct point*)malloc(sizeof(struct point)*(*N));

	//allocate memory on device:
	
	errCode=cudaMalloc( (void**)&dev_database, sizeof(struct point)*(*N));

	printf("\n !!in main GPU method: N is: %u",*N);cout.flush();	
		
	if(errCode != cudaSuccess) {
	cout << "\nError: database Got error with code " << errCode << endl; cout.flush(); 
	}


	



	//first, we copy the x and y values from dataPoints to the database
	for (int i=0; i<(*N); i++)
	{
		database[i].x=(*dataPoints)[i].x;
		database[i].y=(*dataPoints)[i].y;
	}



	//printf("\n size of database: %d",N);

	//copy database to the device:
	errCode=cudaMemcpy(dev_database, database, sizeof(struct point)*(*N), cudaMemcpyHostToDevice);	
	if(errCode != cudaSuccess) {
	cout << "\nError: database Got error with code " << errCode << endl; 
	}

	///////////////////////////////////
	//END COPY THE DATABASE TO THE GPU
	///////////////////////////////////




	///////////////////////////////////
	//COPY THE INDEX TO THE GPU
	///////////////////////////////////

	// //test print the index
	// for (int i=0; i<(*gridNumXCells)*(*gridNumYCells); i++)
	// {
	// 	printf("\nCell %d: min: %d, max: %d", i, index[i].indexmin, index[i].indexmax);
	// }


	int totalGridCells=(*gridNumXCells)*(*gridNumYCells);

	struct grid * dev_grid;
	dev_grid=(struct grid*)malloc(sizeof(struct grid)*totalGridCells);

	//allocate memory on device:
	errCode=cudaMalloc( (void**)&dev_grid, sizeof(struct grid)*totalGridCells);
	
		
	if(errCode != cudaSuccess) {
	cout << "\nError: grid index Got error with code " << errCode << endl; cout.flush(); 
	}

	//copy grid index to the device:
	errCode=cudaMemcpy(dev_grid, index, sizeof(struct grid)*totalGridCells, cudaMemcpyHostToDevice);	
	if(errCode != cudaSuccess) {
	cout << "\nError: grid index allocation Got error with code " << errCode << endl; 
	}	


	///////////////////////////////////
	//END COPY THE INDEX TO THE GPU
	///////////////////////////////////


	///////////////////////////////////
	//COPY THE LOOKUP ARRAY TO THE GPU
	///////////////////////////////////

	//test print the lookup array:
	// for (int i=0; i<*N; i++)
	// {
	// 	printf("\nlookup %d: %d",i, lookupArr[i]);
	// }


	int * dev_lookupArr;
	dev_lookupArr=(int*)malloc(sizeof(int)*(*N));

	//allocate memory on device:
	errCode=cudaMalloc( (void**)&dev_lookupArr, sizeof(int)*(*N));

	if(errCode != cudaSuccess) {
	cout << "\nError: lookup array Got error with code " << errCode << endl; cout.flush(); 
	}

	//copy lookup array to the device:
	errCode=cudaMemcpy(dev_lookupArr, lookupArr, sizeof(int)*(*N), cudaMemcpyHostToDevice);	
	if(errCode != cudaSuccess) {
	cout << "\nError: lookup array allocation Got error with code " << errCode << endl; 
	}	


	///////////////////////////////////
	//END COPY THE LOOKUP ARRAY TO THE GPU
	///////////////////////////////////


	///////////////////////////////////
	//COPY GRID DIMENSIONS TO THE GPU
	//THIS INCLUDES THE NUMBER OF CELLS IN EACH X AND Y DIMENSION, 
	//AND THE STARTING POINT IN THE X AND Y DIMENSIONS THAT THE GRID STARTS AT
	///////////////////////////////////

	//The minimum x boundary of the grid:
	//gridMin_x
	double * dev_gridMin_x;
	dev_gridMin_x=(double*)malloc(sizeof( double ));

	//Allocate on the device
	errCode=cudaMalloc((void**)&dev_gridMin_x, sizeof(double));
	if(errCode != cudaSuccess) {
	cout << "\nError: gridMin_x Got error with code " << errCode << endl; 
	}

	errCode=cudaMemcpy( dev_gridMin_x, gridMin_x, sizeof(double), cudaMemcpyHostToDevice );
	if(errCode != cudaSuccess) {
	cout << "\nError: gridMin_x Got error with code " << errCode << endl; 
	}	


	//The minimum y boundary of the grid:
	//gridMin_y
	double * dev_gridMin_y;
	dev_gridMin_y=(double*)malloc(sizeof( double ));

	//Allocate on the device
	errCode=cudaMalloc((void**)&dev_gridMin_y, sizeof(double));
	if(errCode != cudaSuccess) {
	cout << "\nError: gridMin_y Got error with code " << errCode << endl; 
	}

	errCode=cudaMemcpy( dev_gridMin_y, gridMin_y, sizeof(double), cudaMemcpyHostToDevice );
	if(errCode != cudaSuccess) {
	cout << "\nError: gridMin_y Got error with code " << errCode << endl; 
	}	


	//The number of cells in the x dimension:
	//gridNumXCells
	int * dev_gridNumXCells; 
	dev_gridNumXCells=(int*)malloc(sizeof(int));
	*dev_gridNumXCells=0;

	//allocate on the device
	errCode=cudaMalloc((int**)&dev_gridNumXCells, sizeof(int));	
	if(errCode != cudaSuccess) {
	cout << "\nError: dev_gridNumXCells Got error with code " << errCode << endl; 
	}


	errCode=cudaMemcpy( dev_gridNumXCells, gridNumXCells, sizeof(int), cudaMemcpyHostToDevice );
	if(errCode != cudaSuccess) {
	cout << "\nError: dev_gridNumXCells memcpy Got error with code " << errCode << endl; 
	}


	//The number of cells in the y dimension:
	//gridNumYCells
	int * dev_gridNumYCells; 
	dev_gridNumYCells=(int*)malloc(sizeof(int));
	*dev_gridNumYCells=0;

	//allocate on the device
	errCode=cudaMalloc((int**)&dev_gridNumYCells, sizeof(int));	
	if(errCode != cudaSuccess) {
	cout << "\nError: dev_gridNumYCells Got error with code " << errCode << endl; 
	}


	errCode=cudaMemcpy( dev_gridNumYCells, gridNumYCells, sizeof(int), cudaMemcpyHostToDevice );
	if(errCode != cudaSuccess) {
	cout << "\nError: dev_gridNumYCells memcpy Got error with code " << errCode << endl; 
	}



	///////////////////////////////////
	//END COPY GRID DIMENSIONS TO THE GPU
	///////////////////////////////////





	///////////////////////////////////
	//ALLOCATE MEMORY FOR THE RESULT SET
	///////////////////////////////////
	
	//ORIGINAL, TESTING PINNED MEMORY
	/*
	struct structresults * dev_results;
	struct structresults * results;

	errCode=cudaMalloc((void **)&dev_results, sizeof(struct structresults)*BUFFERELEM);
	if(errCode != cudaSuccess) {
	cout << "CUDA: Got error with code " << errCode << endl; //2 means not enough memory
	}

	printf("\nmemory requested for results (GiB): %f",(double)(sizeof(struct structresults)*BUFFERELEM)/(1024*1024*1024));

	//host result allocation:
	results=(struct structresults*)malloc(sizeof(struct structresults)*BUFFERELEM);
	*/


	//PINNED MEMORY FOR THE RESULT SET
	/*
	struct structresults * dev_results;
	struct structresults * results;

	errCode=cudaMalloc((void **)&dev_results, sizeof(struct structresults)*BUFFERELEM);
	if(errCode != cudaSuccess) {
	cout << "CUDA: Got error with code " << errCode << endl; //2 means not enough memory
	}

	printf("\nmemory requested for results (GiB): %f",(double)(sizeof(struct structresults)*BUFFERELEM)/(1024*1024*1024));

	//host result allocation:
	//results=(struct structresults*)malloc(sizeof(struct structresults)*BUFFERELEM);

	//pinned result set memory for the host
	cudaMallocHost((void **) &results, sizeof(struct structresults)*BUFFERELEM);
	*/



	//NON-PINNED MEMORY FOR SINGLE KERNEL INVOCATION (NO BATCHING)


	//CHANGING THE RESULTS TO KEY VALUE PAIR SORT, WHICH IS TWO ARRAYS
	//KEY IS THE POINT ID
	//THE VALUE IS THE POINT ID WITHIN THE DISTANCE OF KEY

	int * dev_pointIDKey; //key
	int * dev_pointInDistValue; //value

	int * pointIDKey; //key
	int * pointInDistValue; //value


	errCode=cudaMalloc((void **)&dev_pointIDKey, sizeof(int)*BUFFERELEM);
	if(errCode != cudaSuccess) {
	cout << "CUDA: Got error with code " << errCode << endl; //2 means not enough memory
	}

	errCode=cudaMalloc((void **)&dev_pointInDistValue, sizeof(int)*BUFFERELEM);
	if(errCode != cudaSuccess) {
	cout << "CUDA: Got error with code " << errCode << endl; //2 means not enough memory
	}


	printf("\nmemory requested for results (GiB): %f",(double)(sizeof(int)*2*BUFFERELEM)/(1024*1024*1024));

	double tstartalloc=omp_get_wtime();

	//host result allocation:
	//pinned result set memory for the host
	// cudaMallocHost((void **) &pointIDKey, sizeof(int)*BUFFERELEM);
	// cudaMallocHost((void **) &pointInDistValue, sizeof(int)*BUFFERELEM);



	//PAGED MEMORY ALLOCATION FOR SMALL RESULT SET WITH SINGLE KERNEL EXECUTION?
	pointIDKey=(int*)malloc(sizeof(int)*BUFFERELEM);
	pointInDistValue=(int*)malloc(sizeof(int)*BUFFERELEM);

	double tendalloc=omp_get_wtime();


	//printf("\nTime to allocate pinned memory on the host: %f", tendalloc - tstartalloc);
	printf("\nTime to allocate (non-pinned) memory on the host: %f", tendalloc - tstartalloc);




	
	///////////////////////////////////
	//END ALLOCATE MEMORY FOR THE RESULT SET
	///////////////////////////////////















	///////////////////////////////////
	//SET OTHER KERNEL PARAMETERS
	///////////////////////////////////

	

	//count values
	unsigned int * cnt;
	cnt=(unsigned int*)malloc(sizeof(unsigned int));
	*cnt=0;

	unsigned int * dev_cnt; 
	dev_cnt=(unsigned int*)malloc(sizeof(unsigned int));
	*dev_cnt=0;

	//allocate on the device
	errCode=cudaMalloc((unsigned int**)&dev_cnt, sizeof(unsigned int));	
	if(errCode != cudaSuccess) {
	cout << "\nError: cnt Got error with code " << errCode << endl; 
	}


	errCode=cudaMemcpy( dev_cnt, cnt, sizeof(unsigned int), cudaMemcpyHostToDevice );
	if(errCode != cudaSuccess) {
	cout << "\nError: dev_cnt memcpy Got error with code " << errCode << endl; 
	}

	
	//Epsilon
	double * dev_epsilon;
	dev_epsilon=(double*)malloc(sizeof( double ));
	//*dev_epsilon=*epsilon;

	//Allocate on the device
	errCode=cudaMalloc((void**)&dev_epsilon, sizeof(double));
	if(errCode != cudaSuccess) {
	cout << "\nError: epsilon Got error with code " << errCode << endl; 
	}

	//copy to device
	errCode=cudaMemcpy( dev_epsilon, epsilon, sizeof(double), cudaMemcpyHostToDevice );
	if(errCode != cudaSuccess) {
	cout << "\nError: epsilon Got error with code " << errCode << endl; 
	}		




	
	//size of the database:
	unsigned int * dev_N; 
	dev_N=(unsigned int*)malloc(sizeof( unsigned int ));

	//allocate on the device
	errCode=cudaMalloc((void**)&dev_N, sizeof(unsigned int));
	if(errCode != cudaSuccess) {
	cout << "\nError: N Got error with code " << errCode << endl; 
	}	


	//copy N to device
	errCode=cudaMemcpy( dev_N, N, sizeof(unsigned int), cudaMemcpyHostToDevice );
	if(errCode != cudaSuccess) {
	cout << "\nError: N Got error with code " << errCode << endl; 
	}		




	printf("\n\nMODIFIED THIS FUNCTION TO ADD THE OFFSET FOR BATCHING AND THE BATCH NUMBER\n\n");
	printf("\nWITH A SINGLE BATCH-- THE BATCH OFFSET IS SET TO 1 AND THE BATCH NUMBER IS SET TO 0.");

	//offset into the database when batching the results
	unsigned int * batchOffset; 
	batchOffset=(unsigned int*)malloc(sizeof( unsigned int ));
	*batchOffset=0;


	unsigned int * dev_offset; 
	dev_offset=(unsigned int*)malloc(sizeof( unsigned int ));

	//allocate on the device
	errCode=cudaMalloc((void**)&dev_offset, sizeof(unsigned int));
	if(errCode != cudaSuccess) {
	cout << "\nError: offset Got error with code " << errCode << endl; 
	}
	

	//the offset for batching, which keeps track of where to start processing at each batch
	*batchOffset=1;
	errCode=cudaMemcpy( dev_offset, batchOffset, sizeof(unsigned int), cudaMemcpyHostToDevice );
	if(errCode != cudaSuccess) {
	cout << "\nError: dev_offset memcpy Got error with code " << errCode << endl; 
	}


	//Batch number to calculate the point to process (in conjunction with the offset)

	//offset into the database when batching the results
	unsigned int * batchNumber; 
	batchNumber=(unsigned int*)malloc(sizeof(unsigned int));
	*batchNumber=0;


	unsigned int * dev_batchNumber; 
	dev_batchNumber=(unsigned int*)malloc(sizeof(unsigned int));

	//allocate on the device
	errCode=cudaMalloc((void**)&dev_batchNumber, sizeof(unsigned int));
	if(errCode != cudaSuccess) {
	cout << "\nError: batchNumber Got error with code " << errCode << endl; 
	}


	errCode=cudaMemcpy( dev_batchNumber, batchNumber, sizeof(unsigned int), cudaMemcpyHostToDevice);
	if(errCode != cudaSuccess) {
	cout << "\nError: dev_batchNumber memcpy Got error with code " << errCode << endl; 
	}






	//debug values
	unsigned int * dev_debug1; 
	dev_debug1=(unsigned int *)malloc(sizeof(unsigned int ));
	*dev_debug1=0;

	unsigned int * dev_debug2; 
	dev_debug2=(unsigned int *)malloc(sizeof(unsigned int ));
	*dev_debug2=0;

	unsigned int * debug1; 
	debug1=(unsigned int *)malloc(sizeof(unsigned int ));
	*debug1=0;

	unsigned int * debug2; 
	debug2=(unsigned int *)malloc(sizeof(unsigned int ));
	*debug2=0;



	//allocate on the device
	errCode=cudaMalloc( (unsigned int **)&dev_debug1, sizeof(unsigned int ) );
	if(errCode != cudaSuccess) {
	cout << "\nError: debug1 Got error with code " << errCode << endl; 
	}		
	errCode=cudaMalloc( (unsigned int **)&dev_debug2, sizeof(unsigned int ) );
	if(errCode != cudaSuccess) {
	cout << "\nError: debug2 Got error with code " << errCode << endl; 
	}		

	//set to 0
	//copy debug to device
	errCode=cudaMemcpy( dev_debug1, debug1, sizeof(unsigned int), cudaMemcpyHostToDevice );
	if(errCode != cudaSuccess) {
	cout << "\nError: dev_debug1 Got error with code " << errCode << endl; 
	}

	errCode=cudaMemcpy( dev_debug2, debug2, sizeof(unsigned int), cudaMemcpyHostToDevice );
	if(errCode != cudaSuccess) {
	cout << "\nError: dev_debug2 Got error with code " << errCode << endl; 
	}




	

	




	///////////////////////////////////
	//END SET OTHER KERNEL PARAMETERS
	///////////////////////////////////


	


	///////////////////////////////////
	//LAUNCH KERNEL
	///////////////////////////////////
	


	const int TOTALBLOCKS=ceil((1.0*(*N))/(1.0*BLOCKSIZE));	
	printf("\ntotal blocks: %d",TOTALBLOCKS);

	//execute kernel	
	
	kernelGridIndex<<< TOTALBLOCKS, BLOCKSIZE >>>(dev_N, dev_offset, dev_batchNumber, dev_debug1, dev_debug2, dev_epsilon, dev_grid, dev_gridMin_x, dev_gridMin_y, dev_gridNumXCells, dev_gridNumYCells, dev_lookupArr, dev_cnt, dev_database, dev_pointIDKey, dev_pointInDistValue);
	
	// errCode=cudaDeviceSynchronize();
	// cout <<"\n\nError from device synchronize: "<<errCode;

	cout <<"\n\nKERNEL LAUNCH RETURN: "<<cudaGetLastError()<<endl<<endl;
	if ( cudaSuccess != cudaGetLastError() ){
    	cout <<"\n\nERROR IN KERNEL LAUNCH. ERROR: "<<cudaSuccess<<endl<<endl;
    }

    ///////////////////////////////////
	//END LAUNCH KERNEL
	///////////////////////////////////

    

    ///////////////////////////////////
	//GET RESULT SET
	///////////////////////////////////

    //dont get the result set because we leave it on the device for sorting 
    //without transfering back to the host

    
	//first find the size of the number of results
	errCode=cudaMemcpy( cnt, dev_cnt, sizeof(unsigned int), cudaMemcpyDeviceToHost );
	if(errCode != cudaSuccess) {
	cout << "\nError: getting cnt from GPU Got error with code " << errCode << endl; 
	}
	else
	{
		printf("\nGPU: result set size within epsilon (GPU grid): %d",*cnt);
	}

	//copy the results, but only transfer the number of results, not the entire buffer
	/*
	errCode=cudaMemcpy(results, dev_results, sizeof(struct structresults)*(*cnt), cudaMemcpyDeviceToHost );
	
	if(errCode != cudaSuccess) {
	cout << "\nError: getting results from GPU Got error with code " << errCode << endl; 
	}

	*/
	*totalNeighbors=(*cnt);

	double tKernelResultsEnd=omp_get_wtime();
	
	printf("\nTime to launch kernel and execute all of the previous part of the method and get the results back: %f",tKernelResultsEnd-tKernelResultsStart);



	//SORTING FOR TESTING ONLY
	//XXXXXXXXX
	//XXXXXXXXX
	// std::sort(results, results+(*cnt),compResults);
	// printf("\n**** GPU\n");
	// for (int i=0; i<(*cnt); i++)
	// {
	// 	printf("\n%d,%d",results[i].pointID, results[i].pointInDist);
	// }

	//XXXXXXXXX
	//XXXXXXXXX
	//XXXXXXXXX


	//get debug information (optional)
	// unsigned int * debug1;
	// debug1=(unsigned int*)malloc(sizeof(unsigned int));
	// *debug1=0;
	// unsigned int * debug2;
	// debug2=(unsigned int*)malloc(sizeof(unsigned int));
	// *debug2=0;

	double tStartdebug=omp_get_wtime();

	errCode=cudaMemcpy(debug1, dev_debug1, sizeof(unsigned int), cudaMemcpyDeviceToHost );
	
	if(errCode != cudaSuccess) {
	cout << "\nError: getting debug1 from GPU Got error with code " << errCode << endl; 
	}
	else
	{
		printf("\nDebug1 value: %u",*debug1);
	}

	errCode=cudaMemcpy(debug2, dev_debug2, sizeof(unsigned int), cudaMemcpyDeviceToHost );
	
	if(errCode != cudaSuccess) {
	cout << "\nError: getting debug1 from GPU Got error with code " << errCode << endl; 
	}
	else
	{
		printf("\nDebug2 value: %u",*debug2);
	}	

	double tEnddebug=omp_get_wtime();
	printf("\nTime to retrieve debug values: %f", tEnddebug - tStartdebug);


	///////////////////////////////////
	//END GET RESULT SET
	///////////////////////////////////


	///////////////////////////////////
	//FREE MEMORY FROM THE GPU
	///////////////////////////////////
    //free:
	double tFreeStart=omp_get_wtime();

    cudaFree(dev_N);
	cudaFree(dev_database);
	cudaFree(dev_debug1);
	cudaFree(dev_debug2);
	cudaFree(dev_cnt);
	cudaFree(dev_epsilon);
	//cudaFree(dev_results);
	cudaFree(dev_grid);
	cudaFree(dev_lookupArr);
	cudaFree(dev_gridNumXCells);
	cudaFree(dev_gridNumYCells);
	cudaFree(dev_gridMin_x);
	cudaFree(dev_gridMin_y);
	double tFreeEnd=omp_get_wtime();

	printf("\nTime freeing memory: %f", tFreeEnd - tFreeStart);

	////////////////////////////////////
	//cudaDeviceSynchronize();
	

	////////////////////////////////////
	//SORT THE TABLE DATA ON THE GPU
	//THERE IS NO ORDERING BETWEEN EACH POINT AND THE ONES THAT IT'S WITHIN THE DISTANCE OF
	////////////////////////////////////

	/////////////////////////////
	//ONE PROBLEM WITH NOT TRANSFERING THE RESULT OFF OF THE DEVICE IS THAT
	//YOU CAN'T RESIZE THE RESULTS TO BE THE SIZE OF *CNT
	//SO THEN YOU HAVE POTENTIALLY LOTS OF WASTED SPACE
	/////////////////////////////

	//sort by key with the data already on the device:
	//wrap raw pointer with a device_ptr to use with Thrust functions
	thrust::device_ptr<int> dev_keys_ptr(dev_pointIDKey);
	thrust::device_ptr<int> dev_data_ptr(dev_pointInDistValue);


	// allocate space for the output
	//thrust::device_vector<int> sortedKeys(*cnt);
	//thrust::device_vector<int> sortedVals(*cnt);
	


	try{
	thrust::sort_by_key(dev_keys_ptr, dev_keys_ptr + (*cnt), dev_data_ptr);
	}
	catch(std::bad_alloc &e)
	  {
	    std::cerr << "Ran out of memory while sorting" << std::endl;
	    exit(-1);
	  }
	
	
	//copy the sorted arays back to the host
	thrust::copy(dev_keys_ptr,dev_keys_ptr+(*cnt),pointIDKey);
	thrust::copy(dev_data_ptr,dev_data_ptr+(*cnt),pointInDistValue);
	  


	//free the data on the device
	cudaFree(dev_pointIDKey);
	cudaFree(dev_pointInDistValue);


	/*
	double tstartsort=omp_get_wtime();
	//make a host vector initialized with the results that have been transfered from the GPU
	
	double sort_test1_start=omp_get_wtime(); //TESTING

	thrust::host_vector<structresults> hVectResults(results,results+(*cnt));
	

	double sort_test1_end=omp_get_wtime(); //TESTING
	printf("\n Time to create the host vector: %f", sort_test1_end - sort_test1_start); //TESTING


	// for (int i=0;i<numResults;i++)
	// {
	// 	printf("\n host vector: %d, %d",hVectResults[i].pointID,hVectResults[i].pointInDist);
	// }

	// for (int i=0; i<numResults; i++)
	// {
	// 	structresults tmp;
	// 	tmp.pointID=0;
	// 	tmp.pointInDist=0;
	// 	hVectResults.push_back(tmp);
	// }

	//Now transfer the hostvector to the device:

	double sort_test2_start=omp_get_wtime(); //TESTING	

	
	thrust::device_vector<structresults> dVectResults=hVectResults;


	double sort_test2_end=omp_get_wtime(); //TESTING	
	printf("\n Time to create the device vector: %f", sort_test2_end - sort_test2_start); //TESTING





	//sort the device vector on the GPU

	try{
	thrust::sort(dVectResults.begin(), dVectResults.end(),compareThrust());
	}
	catch(std::bad_alloc &e)
	  {
	    std::cerr << "Ran out of memory while sorting" << std::endl;
	    exit(-1);
	  }

	// transfer the sorted results back to host
	thrust::copy(dVectResults.begin(), dVectResults.end(), hVectResults.begin());

	double tendsort=omp_get_wtime();

	printf("\nTime to sort on the GPU (grid index): %f",tendsort-tstartsort);

	*/

	//print GPU:
	// for (int i=0; i<(*cnt);i++)
	// {	
	// 	printf("\nGPU elem: %d, data: %d",hVectResults[i].pointID,hVectResults[i].pointInDist);
	// }


	////////////////////////////////////
	//END SORT THE DATA ON THE GPU
	////////////////////////////////////


	////////////////////////////////////
	//CONSTRUCT TABLE
	////////////////////////////////////	


	double tStartTableConstruct=omp_get_wtime();
	constructNeighborTableKeyValue(pointIDKey, pointInDistValue, neighborTable, cnt);
	double tEndTableConstruct=omp_get_wtime();	
	printf("\nTime constructing table: %f",tEndTableConstruct - tStartTableConstruct);	
	

	
	////////////////////////////////////
	//END CONSTRUCT TABLE
	////////////////////////////////////	

	






}
















//In this function we batch the results off of the GPU to accomodate larger epsilon values
//The results that come from the GPU are in the form of key/value pairs (in two arrays)
//Key-a point, Value-a point within epsilon of the key 
//The batches are mapped to differing streams
//Each batch requires its own memory space for the result set
//So the number of buffers on the GPU for the results is the number of streams (GPUSTREAMS)
//On the host, we use the same size buffers, and number of them, and then build part of the neighbor table with the batch

//This is an alternative to making one large array from all of the batches, which would require a large
//pinned cuda malloc which is very expensive.  It also allows for multiple threads to concurrently build the 
//neighbor table and interleave GPU work with work on the CPU

//Also, the number of batches is estimated by calling a kernel that samples the number of neighbours (1%) and then
//estimates the total neighbors, which is used to calculate the total number of batches
//To make sure each batch doesn't vary much, we use a strided scheme for each batch

//Uses the grid index to compute the direct neighbor table
//NO SHARED MEMORY PAGING
void makeDistanceTableGPUGridIndexBatches(std::vector<struct dataElem> * dataPoints, double * epsilon, struct grid * index, double * gridMin_x, double * gridMin_y, int * gridNumXCells, int * gridNumYCells, unsigned int * lookupArr, struct table * neighborTable, unsigned int * totalNeighbors)
{

	//testing new neighbortable:
	struct neighborTableLookup * newNeighborTable;
	newNeighborTable=new neighborTableLookup[dataPoints->size()];


	double tKernelResultsStart=omp_get_wtime();

	//CUDA error code:
	cudaError_t errCode;


	cout<<"\n** last error start of fn: "<<cudaGetLastError();


	




	///////////////////////////////////
	//COPY THE DATABASE TO THE GPU
	///////////////////////////////////


	unsigned int * DBSIZE;
	DBSIZE=(unsigned int*)malloc(sizeof(unsigned int));
	*DBSIZE=dataPoints->size();
	
	printf("\n in main GPU method: DBSIZE is: %u",*DBSIZE);cout.flush();

	struct point * database;
	
	//pinned memory for the database:
	database=(struct point*)malloc(sizeof(struct point)*(*DBSIZE));
	//dont use pinned memory for the database, its slower than using cudaMalloc
	//cudaMallocHost((void **) &database, sizeof(struct point)*(*DBSIZE));


	struct point * dev_database;
	dev_database=(struct point*)malloc(sizeof(struct point)*(*DBSIZE));

	//allocate memory on device:
	
	errCode=cudaMalloc( (void**)&dev_database, sizeof(struct point)*(*DBSIZE));

	printf("\n !!in main GPU method: DBSIZE is: %u",*DBSIZE);cout.flush();	
		
	if(errCode != cudaSuccess) {
	cout << "\nError: database Got error with code " << errCode << endl; cout.flush(); 
	}


	



	
	//first, we copy the x and y values from dataPoints to the database
	//we do this because the data points struct may contain other values than x and y
	for (int i=0; i<(*DBSIZE); i++)
	{
		database[i].x=(*dataPoints)[i].x;
		database[i].y=(*dataPoints)[i].y;
	}







	//printf("\n size of database: %d",N);

	//copy database to the device:
	errCode=cudaMemcpy(dev_database, database, sizeof(struct point)*(*DBSIZE), cudaMemcpyHostToDevice);	
	if(errCode != cudaSuccess) {
	cout << "\nError: database Got error with code " << errCode << endl; 
	}

	///////////////////////////////////
	//END COPY THE DATABASE TO THE GPU
	///////////////////////////////////




	///////////////////////////////////
	//COPY THE INDEX TO THE GPU
	///////////////////////////////////

	// //test print the index
	// for (int i=0; i<(*gridNumXCells)*(*gridNumYCells); i++)
	// {
	// 	printf("\nCell %d: min: %d, max: %d", i, index[i].indexmin, index[i].indexmax);
	// }


	int totalGridCells=(*gridNumXCells)*(*gridNumYCells);

	struct grid * dev_grid;
	dev_grid=(struct grid*)malloc(sizeof(struct grid)*totalGridCells);

	//allocate memory on device:
	errCode=cudaMalloc( (void**)&dev_grid, sizeof(struct grid)*totalGridCells);
	
		
	if(errCode != cudaSuccess) {
	cout << "\nError: grid index Got error with code " << errCode << endl; cout.flush(); 
	}

	//copy grid index to the device:
	errCode=cudaMemcpy(dev_grid, index, sizeof(struct grid)*totalGridCells, cudaMemcpyHostToDevice);	
	if(errCode != cudaSuccess) {
	cout << "\nError: grid index allocation Got error with code " << errCode << endl; 
	}	

	printf("\nSize of index sent to GPU (GiB): %f", (double)sizeof(struct grid)*totalGridCells/(1024.0*1024.0*1024.0));

	///////////////////////////////////
	//END COPY THE INDEX TO THE GPU
	///////////////////////////////////


	///////////////////////////////////
	//COPY THE LOOKUP ARRAY TO THE GPU
	///////////////////////////////////

	


	int * dev_lookupArr;
	dev_lookupArr=(int*)malloc(sizeof(int)*(*DBSIZE));

	//allocate memory on device:
	errCode=cudaMalloc( (void**)&dev_lookupArr, sizeof(int)*(*DBSIZE));

	if(errCode != cudaSuccess) {
	cout << "\nError: lookup array Got error with code " << errCode << endl; cout.flush(); 
	}

	//copy lookup array to the device:
	errCode=cudaMemcpy(dev_lookupArr, lookupArr, sizeof(int)*(*DBSIZE), cudaMemcpyHostToDevice);	
	if(errCode != cudaSuccess) {
	cout << "\nError: lookup array allocation Got error with code " << errCode << endl; 
	}	


	///////////////////////////////////
	//END COPY THE LOOKUP ARRAY TO THE GPU
	///////////////////////////////////


	///////////////////////////////////
	//COPY GRID DIMENSIONS TO THE GPU
	//THIS INCLUDES THE NUMBER OF CELLS IN EACH X AND Y DIMENSION, 
	//AND THE STARTING POINT IN THE X AND Y DIMENSIONS THAT THE GRID STARTS AT
	///////////////////////////////////

	//The minimum x boundary of the grid:
	//gridMin_x
	double * dev_gridMin_x;
	dev_gridMin_x=(double*)malloc(sizeof( double ));

	//Allocate on the device
	errCode=cudaMalloc((void**)&dev_gridMin_x, sizeof(double));
	if(errCode != cudaSuccess) {
	cout << "\nError: gridMin_x Got error with code " << errCode << endl; 
	}

	errCode=cudaMemcpy( dev_gridMin_x, gridMin_x, sizeof(double), cudaMemcpyHostToDevice );
	if(errCode != cudaSuccess) {
	cout << "\nError: gridMin_x Got error with code " << errCode << endl; 
	}	


	//The minimum y boundary of the grid:
	//gridMin_y
	double * dev_gridMin_y;
	dev_gridMin_y=(double*)malloc(sizeof( double ));

	//Allocate on the device
	errCode=cudaMalloc((void**)&dev_gridMin_y, sizeof(double));
	if(errCode != cudaSuccess) {
	cout << "\nError: gridMin_y Got error with code " << errCode << endl; 
	}

	errCode=cudaMemcpy( dev_gridMin_y, gridMin_y, sizeof(double), cudaMemcpyHostToDevice );
	if(errCode != cudaSuccess) {
	cout << "\nError: gridMin_y Got error with code " << errCode << endl; 
	}	


	//The number of cells in the x dimension:
	//gridNumXCells
	int * dev_gridNumXCells; 
	dev_gridNumXCells=(int*)malloc(sizeof(int));
	*dev_gridNumXCells=0;

	//allocate on the device
	errCode=cudaMalloc((int**)&dev_gridNumXCells, sizeof(int));	
	if(errCode != cudaSuccess) {
	cout << "\nError: dev_gridNumXCells Got error with code " << errCode << endl; 
	}


	errCode=cudaMemcpy( dev_gridNumXCells, gridNumXCells, sizeof(int), cudaMemcpyHostToDevice );
	if(errCode != cudaSuccess) {
	cout << "\nError: dev_gridNumXCells memcpy Got error with code " << errCode << endl; 
	}


	//The number of cells in the y dimension:
	//gridNumYCells
	int * dev_gridNumYCells; 
	dev_gridNumYCells=(int*)malloc(sizeof(int));
	*dev_gridNumYCells=0;

	//allocate on the device
	errCode=cudaMalloc((int**)&dev_gridNumYCells, sizeof(int));	
	if(errCode != cudaSuccess) {
	cout << "\nError: dev_gridNumYCells Got error with code " << errCode << endl; 
	}


	errCode=cudaMemcpy( dev_gridNumYCells, gridNumYCells, sizeof(int), cudaMemcpyHostToDevice );
	if(errCode != cudaSuccess) {
	cout << "\nError: dev_gridNumYCells memcpy Got error with code " << errCode << endl; 
	}



	///////////////////////////////////
	//END COPY GRID DIMENSIONS TO THE GPU
	///////////////////////////////////






	









	///////////////////////////////////
	//SET OTHER KERNEL PARAMETERS
	///////////////////////////////////

	//total size of the result set as it's batched
	//this isnt sent to the GPU
	unsigned int * totalResultSetCnt;
	totalResultSetCnt=(unsigned int*)malloc(sizeof(unsigned int));
	*totalResultSetCnt=0;

	//count values - for an individual kernel launch
	//need different count values for each stream
	unsigned int * cnt;
	cnt=(unsigned int*)malloc(sizeof(unsigned int)*GPUSTREAMS);
	*cnt=0;

	unsigned int * dev_cnt; 
	dev_cnt=(unsigned int*)malloc(sizeof(unsigned int)*GPUSTREAMS);
	*dev_cnt=0;

	//allocate on the device
	errCode=cudaMalloc((unsigned int**)&dev_cnt, sizeof(unsigned int)*GPUSTREAMS);	
	if(errCode != cudaSuccess) {
	cout << "\nError: cnt Got error with code " << errCode << endl; 
	}


	// errCode=cudaMemcpy( dev_cnt, cnt, sizeof(unsigned int), cudaMemcpyHostToDevice );
	// if(errCode != cudaSuccess) {
	// cout << "\nError: dev_cnt memcpy Got error with code " << errCode << endl; 
	// }

	
	//Epsilon
	double * dev_epsilon;
	dev_epsilon=(double*)malloc(sizeof( double ));
	//*dev_epsilon=*epsilon;

	//Allocate on the device
	errCode=cudaMalloc((void**)&dev_epsilon, sizeof(double));
	if(errCode != cudaSuccess) {
	cout << "\nError: epsilon Got error with code " << errCode << endl; 
	}

	//copy to device
	errCode=cudaMemcpy( dev_epsilon, epsilon, sizeof(double), cudaMemcpyHostToDevice );
	if(errCode != cudaSuccess) {
	cout << "\nError: epsilon Got error with code " << errCode << endl; 
	}		




	
	//number of threads per gpu stream

	//THE NUMBER OF THREADS THAT ARE LAUNCHED IN A SINGLE KERNEL INVOCATION
	//CAN BE FEWER THAN THE NUMBER OF ELEMENTS IN THE DATABASE IF MORE THAN 1 BATCH
	unsigned int * N;
	N=(unsigned int*)malloc(sizeof(unsigned int)*GPUSTREAMS);
	


	unsigned int * dev_N; 
	dev_N=(unsigned int*)malloc(sizeof(unsigned int)*GPUSTREAMS);

	//allocate on the device
	errCode=cudaMalloc((void**)&dev_N, sizeof(unsigned int)*GPUSTREAMS);
	if(errCode != cudaSuccess) {
	cout << "\nError: dev_N Got error with code " << errCode << endl; 
	}	


	//offset into the database when batching the results
	unsigned int * batchOffset; 
	batchOffset=(unsigned int*)malloc(sizeof(unsigned int)*GPUSTREAMS);
	//*batchOffset=0;


	unsigned int * dev_offset; 
	dev_offset=(unsigned int*)malloc(sizeof(unsigned int)*GPUSTREAMS);

	//allocate on the device
	errCode=cudaMalloc((void**)&dev_offset, sizeof(unsigned int)*GPUSTREAMS);
	if(errCode != cudaSuccess) {
	cout << "\nError: offset Got error with code " << errCode << endl; 
	}

	//Batch number to calculate the point to process (in conjunction with the offset)

	//offset into the database when batching the results
	unsigned int * batchNumber; 
	batchNumber=(unsigned int*)malloc(sizeof(unsigned int)*GPUSTREAMS);
	//*batchOffset=0;


	unsigned int * dev_batchNumber; 
	dev_batchNumber=(unsigned int*)malloc(sizeof(unsigned int)*GPUSTREAMS);

	//allocate on the device
	errCode=cudaMalloc((void**)&dev_batchNumber, sizeof(unsigned int)*GPUSTREAMS);
	if(errCode != cudaSuccess) {
	cout << "\nError: batchNumber Got error with code " << errCode << endl; 
	}


			

	//debug values
	unsigned int * dev_debug1; 
	dev_debug1=(unsigned int *)malloc(sizeof(unsigned int ));
	*dev_debug1=0;

	unsigned int * dev_debug2; 
	dev_debug2=(unsigned int *)malloc(sizeof(unsigned int ));
	*dev_debug2=0;

	unsigned int * debug1; 
	debug1=(unsigned int *)malloc(sizeof(unsigned int ));
	*debug1=0;

	unsigned int * debug2; 
	debug2=(unsigned int *)malloc(sizeof(unsigned int ));
	*debug2=0;



	//allocate on the device
	errCode=cudaMalloc( (unsigned int **)&dev_debug1, sizeof(unsigned int ) );
	if(errCode != cudaSuccess) {
	cout << "\nError: debug1 Got error with code " << errCode << endl; 
	}		
	errCode=cudaMalloc( (unsigned int **)&dev_debug2, sizeof(unsigned int ) );
	if(errCode != cudaSuccess) {
	cout << "\nError: debug2 Got error with code " << errCode << endl; 
	}		

	//set to 0
	//copy debug to device
	errCode=cudaMemcpy( dev_debug1, debug1, sizeof(unsigned int), cudaMemcpyHostToDevice );
	if(errCode != cudaSuccess) {
	cout << "\nError: dev_debug1 Got error with code " << errCode << endl; 
	}

	errCode=cudaMemcpy( dev_debug2, debug2, sizeof(unsigned int), cudaMemcpyHostToDevice );
	if(errCode != cudaSuccess) {
	cout << "\nError: dev_debug2 Got error with code " << errCode << endl; 
	}



	//////////////////////////////////////////////////////////
	//ESTIMATE THE BUFFER SIZE AND NUMBER OF BATCHES ETC BY COUNTING THE NUMBER OF RESULTS
	//TAKE A SAMPLE OF THE DATA POINTS, NOT ALL OF THEM
	//Use sampleRate for this
	/////////////////////////////////////////////////////////

	printf("\n\n***********************************\nEstimating Batches:");
	//Parameters for the batch size estimation.
	double sampleRate=0.01; //sample 1% of the points in the dataset sampleRate=0.01. 
						//Sample the entire dataset(no sampling) sampleRate=1
	int offsetRate=1.0/sampleRate;
	printf("\nOffset: %d", offsetRate);


	/////////////////
	//N-threads
	////////////////

	
	double tstartbatchest=omp_get_wtime();

	unsigned int * dev_N_batchEst; 
	dev_N_batchEst=(unsigned int*)malloc(sizeof(unsigned int));

	unsigned int * N_batchEst; 
	N_batchEst=(unsigned int*)malloc(sizeof(unsigned int));
	*N_batchEst=*DBSIZE*sampleRate;


	//allocate on the device
	errCode=cudaMalloc((void**)&dev_N_batchEst, sizeof(unsigned int));
	if(errCode != cudaSuccess) {
	cout << "\nError: dev_N_batchEst Got error with code " << errCode << endl; 
	}	

	//copy N to device 
	//N IS THE NUMBER OF THREADS
	errCode=cudaMemcpy( dev_N_batchEst, N_batchEst, sizeof(unsigned int), cudaMemcpyHostToDevice);
	if(errCode != cudaSuccess) {
	cout << "\nError: N batchEST Got error with code " << errCode << endl; 
	}


	/////////////
	//count the result set size 
	////////////

	unsigned int * dev_cnt_batchEst; 
	dev_cnt_batchEst=(unsigned int*)malloc(sizeof(unsigned int));

	unsigned int * cnt_batchEst; 
	cnt_batchEst=(unsigned int*)malloc(sizeof(unsigned int));
	*cnt_batchEst=0;


	//allocate on the device
	errCode=cudaMalloc((void**)&dev_cnt_batchEst, sizeof(unsigned int));
	if(errCode != cudaSuccess) {
	cout << "\nError: dev_cnt_batchEst Got error with code " << errCode << endl; 
	}	

	//copy cnt to device 
	errCode=cudaMemcpy( dev_cnt_batchEst, cnt_batchEst, sizeof(unsigned int), cudaMemcpyHostToDevice);
	if(errCode != cudaSuccess) {
	cout << "\nError: dev_cnt_batchEst Got error with code " << errCode << endl; 
	}

	
	//////////////////
	//SAMPLE OFFSET - TO SAMPLE THE DATA TO ESTIMATE THE TOTAL NUMBER OF KEY VALUE PAIRS
	/////////////////

	//offset into the database when batching the results
	unsigned int * sampleOffset; 
	sampleOffset=(unsigned int*)malloc(sizeof(unsigned int));
	*sampleOffset=offsetRate;


	unsigned int * dev_sampleOffset; 
	dev_sampleOffset=(unsigned int*)malloc(sizeof(unsigned int));

	//allocate on the device
	errCode=cudaMalloc((void**)&dev_sampleOffset, sizeof(unsigned int));
	if(errCode != cudaSuccess) {
	cout << "\nError: sample offset Got error with code " << errCode << endl; 
	}

	//copy offset to device 
	errCode=cudaMemcpy( dev_sampleOffset, sampleOffset, sizeof(unsigned int), cudaMemcpyHostToDevice);
	if(errCode != cudaSuccess) {
	cout << "\nError: dev_cnt_batchEst Got error with code " << errCode << endl; 
	}



	const int TOTALBLOCKSBATCHEST=ceil((1.0*(*DBSIZE)*sampleRate)/(1.0*BLOCKSIZE));	
	printf("\ntotal blocks: %d",TOTALBLOCKSBATCHEST);

	kernelGridIndexBatchEstimator<<< TOTALBLOCKSBATCHEST, BLOCKSIZE>>>(dev_N_batchEst, dev_sampleOffset, dev_debug1, dev_debug2, dev_epsilon, dev_grid, dev_gridMin_x, dev_gridMin_y, dev_gridNumXCells, dev_gridNumYCells, dev_lookupArr, dev_cnt_batchEst, dev_database);
	cout<<"\n** ERROR FROM KERNEL LAUNCH OF BATCH ESTIMATOR: "<<cudaGetLastError();
	// find the size of the number of results
		errCode=cudaMemcpy( cnt_batchEst, dev_cnt_batchEst, sizeof(unsigned int), cudaMemcpyDeviceToHost);
		if(errCode != cudaSuccess) {
		cout << "\nError: getting cnt for batch estimate from GPU Got error with code " << errCode << endl; 
		}
		else
		{
			printf("\nGPU: result set size for estimating the number of batches (sampled): %u",*cnt_batchEst);
		}




	cudaFree(dev_cnt_batchEst);	
	cudaFree(dev_N_batchEst);
	cudaFree(dev_sampleOffset);



	double tendbatchest=omp_get_wtime();

	printf("\nTime to get the total result set size from batch estimator: %f",tendbatchest-tstartbatchest);

	




	//WE CALCULATE THE BUFFER SIZES AND NUMBER OF BATCHES

	unsigned int GPUBufferSize=100000000;
	double alpha=0.05; //overestimation factor

	unsigned long long estimatedTotalSize=(unsigned long long)(*cnt_batchEst)*(unsigned long long)offsetRate;
	unsigned long long estimatedTotalSizeWithAlpha=(unsigned long long)(*cnt_batchEst)*(unsigned long long)offsetRate*(1.0+(alpha));
	printf("\nEstimated total result set size: %llu", estimatedTotalSize);
	printf("\nEstimated total result set size (with Alpha %f): %llu", alpha,estimatedTotalSizeWithAlpha);
	

	//to accomodate small datasets, we need smaller buffers because the pinned memory malloc is expensive
	if (estimatedTotalSize<(GPUBufferSize*GPUSTREAMS))
	{
		GPUBufferSize=estimatedTotalSize*(1.0+(alpha*2.0))/(GPUSTREAMS);		//we do 2*alpha for small datasets because the
																		//sampling will be worse for small datasets
																		//but we fix the 3 streams still (thats why divide by 3).			
	}

	unsigned int numBatches=ceil(((1.0+alpha)*estimatedTotalSize*1.0)/(GPUBufferSize*1.0));
	printf("\nNumber of batches: %d, buffer size: %d", numBatches, GPUBufferSize);


		

	printf("\nEnd Batch Estimator\n***********************************\n");

	/////////////////////////////////////////////////////////	
	//END BATCH ESTIMATOR	
	/////////////////////////////////////////////////////////



	///////////////////////////////////
	//ALLOCATE MEMORY FOR THE RESULT SET USING THE BATCH ESTIMATOR
	///////////////////////////////////
	

	//NEED BUFFERS ON THE GPU AND THE HOST FOR THE NUMBER OF CONCURRENT STREAMS	
	//GPU BUFFER ON THE DEVICE
	//BUFFER ON THE HOST WITH PINNED MEMORY FOR FAST MEMCPY
	//BUFFER ON THE HOST TO DUMP THE RESULTS OF BATCHES SO THAT GPU THREADS CAN CONTINUE
	//EXECUTING STREAMS ON THE HOST



	//GPU MEMORY ALLOCATION:

	//CHANGING THE RESULTS TO KEY VALUE PAIR SORT, WHICH IS TWO ARRAYS
	//KEY IS THE POINT ID
	//THE VALUE IS THE POINT ID WITHIN THE DISTANCE OF KEY

	int * dev_pointIDKey[GPUSTREAMS]; //key
	int * dev_pointInDistValue[GPUSTREAMS]; //value

	

	for (int i=0; i<GPUSTREAMS; i++)
	{
		errCode=cudaMalloc((void **)&dev_pointIDKey[i], sizeof(int)*GPUBufferSize);
		if(errCode != cudaSuccess) {
		cout << "CUDA: Got error with code " << errCode << endl; //2 means not enough memory
		}

		errCode=cudaMalloc((void **)&dev_pointInDistValue[i], sizeof(int)*GPUBufferSize);
		if(errCode != cudaSuccess) {
		cout << "CUDA: Got error with code " << errCode << endl; //2 means not enough memory
		}


	}


	

	//HOST RESULT ALLOCATION FOR THE GPU TO COPY THE DATA INTO A PINNED MEMORY ALLOCATION
	//ON THE HOST
	//pinned result set memory for the host
	//the number of elements are recorded for that batch in resultElemCountPerBatch
	//NEED PINNED MEMORY ALSO BECAUSE YOU NEED IT TO USE STREAMS IN THRUST FOR THE MEMCOPY OF THE SORTED RESULTS	

	//PINNED MEMORY TO COPY FROM THE GPU	
	int * pointIDKey[GPUSTREAMS]; //key
	int * pointInDistValue[GPUSTREAMS]; //value
	

	double tstartpinnedresults=omp_get_wtime();
	
	for (int i=0; i<GPUSTREAMS; i++)
	{
	cudaMallocHost((void **) &pointIDKey[i], sizeof(int)*GPUBufferSize);
	cudaMallocHost((void **) &pointInDistValue[i], sizeof(int)*GPUBufferSize);
	}

	double tendpinnedresults=omp_get_wtime();
	printf("\nTime to allocate pinned memory for results: %f", tendpinnedresults - tstartpinnedresults);
	

	// cudaMalloc((void **) &pointIDKey, sizeof(int)*GPUBufferSize*NUMBATCHES);
	// cudaMalloc((void **) &pointInDistValue, sizeof(int)*GPUBufferSize*NUMBATCHES);




	printf("\nmemory requested for results ON GPU (GiB): %f",(double)(sizeof(int)*2*GPUBufferSize*GPUSTREAMS)/(1024*1024*1024));
	printf("\nmemory requested for results in MAIN MEMORY (GiB): %f",(double)(sizeof(int)*2*GPUBufferSize*GPUSTREAMS)/(1024*1024*1024));

	
	///////////////////////////////////
	//END ALLOCATE MEMORY FOR THE RESULT SET
	///////////////////////////////////











	



	/////////////////////////////////
	//SET OPENMP ENVIRONMENT VARIABLES
	////////////////////////////////

	omp_set_num_threads(GPUSTREAMS);
	

	/////////////////////////////////
	//END SET OPENMP ENVIRONMENT VARIABLES
	////////////////////////////////
	
	

	/////////////////////////////////
	//CREATE STREAMS
	////////////////////////////////

	cudaStream_t stream[GPUSTREAMS];
	
	for (int i=0; i<GPUSTREAMS; i++){

    //cudaStreamCreate(&stream[i]);
	cudaStreamCreateWithFlags(&stream[i], cudaStreamNonBlocking);

	}	

	/////////////////////////////////
	//END CREATE STREAMS
	////////////////////////////////
	
	

	///////////////////////////////////
	//LAUNCH KERNEL IN BATCHES
	///////////////////////////////////
		
	//since we use the strided scheme, some of the batch sizes
	//are off by 1 of each other, a first group of batches will
	//have 1 extra data point to process, and we calculate which batch numbers will 
	//have that.  The batchSize is the lower value (+1 is added to the first ones)


	unsigned int batchSize=(*DBSIZE)/numBatches;
	unsigned int batchesThatHaveOneMore=(*DBSIZE)-(batchSize*numBatches); //batch number 0- < this value have one more
	printf("\n\n***Batches that have one more: %u batchSize(N): %u, \n\n***",batchSize, batchesThatHaveOneMore);

	unsigned int totalResultsLoop=0;


		
		//FOR LOOP OVER THE NUMBER OF BATCHES STARTS HERE
		#pragma omp parallel for schedule(static,1) reduction(+:totalResultsLoop) num_threads(GPUSTREAMS)
		for (int i=0; i<numBatches; i++)
		{	

			int tid=omp_get_thread_num();
			
			printf("\ntid: %d, starting iteration: %d",tid,i);

			//N NOW BECOMES THE NUMBER OF POINTS TO PROCESS PER BATCH
			//AS ONE THREAD PROCESSES A SINGLE POINT
			



			
			if (i<batchesThatHaveOneMore)
			{
				N[tid]=batchSize+1;	
				printf("\nN: %d, tid: %d",N[tid], tid);
			}
			else
			{
				N[tid]=batchSize;	
				printf("\nN (1 less): %d tid: %d",N[tid], tid);
			}

			//set relevant parameters for the batched execution that get reset
			
			//copy N to device 
			//N IS THE NUMBER OF THREADS
			errCode=cudaMemcpyAsync( &dev_N[tid], &N[tid], sizeof(unsigned int), cudaMemcpyHostToDevice, stream[tid] );
			if(errCode != cudaSuccess) {
			cout << "\nError: N Got error with code " << errCode << endl; 
			}

			//the batched result set size (reset to 0):
			cnt[tid]=0;
			errCode=cudaMemcpyAsync( &dev_cnt[tid], &cnt[tid], sizeof(unsigned int), cudaMemcpyHostToDevice, stream[tid] );
			if(errCode != cudaSuccess) {
			cout << "\nError: dev_cnt memcpy Got error with code " << errCode << endl; 
			}

			//the offset for batching, which keeps track of where to start processing at each batch
			//batchOffset[tid]=i*batchSize; //original
			batchOffset[tid]=numBatches; //for the strided
			errCode=cudaMemcpyAsync( &dev_offset[tid], &batchOffset[tid], sizeof(unsigned int), cudaMemcpyHostToDevice, stream[tid] );
			if(errCode != cudaSuccess) {
			cout << "\nError: dev_offset memcpy Got error with code " << errCode << endl; 
			}

			//the batch number for batching with strided
			batchNumber[tid]=i;
			errCode=cudaMemcpyAsync( &dev_batchNumber[tid], &batchNumber[tid], sizeof(unsigned int), cudaMemcpyHostToDevice, stream[tid] );
			if(errCode != cudaSuccess) {
			cout << "\nError: dev_batchNumber memcpy Got error with code " << errCode << endl; 
			}

			const int TOTALBLOCKS=ceil((1.0*(N[tid]))/(1.0*BLOCKSIZE));	
			printf("\ntotal blocks: %d",TOTALBLOCKS);

			//execute kernel	
			//0 is shared memory pool
			kernelGridIndex<<< TOTALBLOCKS, BLOCKSIZE, 0, stream[tid] >>>(&dev_N[tid], &dev_offset[tid], &dev_batchNumber[tid], dev_debug1, dev_debug2, dev_epsilon, dev_grid, dev_gridMin_x, dev_gridMin_y, dev_gridNumXCells, dev_gridNumYCells, dev_lookupArr, &dev_cnt[tid], dev_database, dev_pointIDKey[tid], dev_pointInDistValue[tid]);
			
			// errCode=cudaDeviceSynchronize();
			// cout <<"\n\nError from device synchronize: "<<errCode;

			cout <<"\n\nKERNEL LAUNCH RETURN: "<<cudaGetLastError()<<endl<<endl;
			if ( cudaSuccess != cudaGetLastError() ){
		    	cout <<"\n\nERROR IN KERNEL LAUNCH. ERROR: "<<cudaSuccess<<endl<<endl;
		    }

		    

		   
			// find the size of the number of results
			errCode=cudaMemcpyAsync( &cnt[tid], &dev_cnt[tid], sizeof(unsigned int), cudaMemcpyDeviceToHost, stream[tid] );
			if(errCode != cudaSuccess) {
			cout << "\nError: getting cnt from GPU Got error with code " << errCode << endl; 
			}
			else
			{
				printf("\nGPU: result set size within epsilon (GPU grid): %d",cnt[tid]);
			}


			
			


			////////////////////////////////////
			//SORT THE TABLE DATA ON THE GPU
			//THERE IS NO ORDERING BETWEEN EACH POINT AND THE ONES THAT IT'S WITHIN THE DISTANCE OF
			////////////////////////////////////

			/////////////////////////////
			//ONE PROBLEM WITH NOT TRANSFERING THE RESULT OFF OF THE DEVICE IS THAT
			//YOU CAN'T RESIZE THE RESULTS TO BE THE SIZE OF *CNT
			//SO THEN YOU HAVE POTENTIALLY LOTS OF WASTED SPACE
			/////////////////////////////

			//sort by key with the data already on the device:
			//wrap raw pointer with a device_ptr to use with Thrust functions
			thrust::device_ptr<int> dev_keys_ptr(dev_pointIDKey[tid]);
			thrust::device_ptr<int> dev_data_ptr(dev_pointInDistValue[tid]);

			//XXXXXXXXXXXXXXXX
			//THRUST USING STREAMS REQUIRES THRUST V1.8 
			//SEEMS TO BE WORKING :)
			//XXXXXXXXXXXXXXXX

			try{
			thrust::sort_by_key(thrust::cuda::par.on(stream[tid]), dev_keys_ptr, dev_keys_ptr + cnt[tid], dev_data_ptr);
			}
			catch(std::bad_alloc &e)
			  {
			    std::cerr << "Ran out of memory while sorting" << std::endl;
			    exit(-1);
			  }
			
			
			//#pragma omp critical  
			//{
			//copy the sorted arays back to the host
			//copy to the appropriate place in the larger host arrays
			
			//original thrust copy (doesnt have streams)
			//thrust::copy(dev_keys_ptr,dev_keys_ptr+cnt[tid],pointIDKey+(*totalResultSetCnt));
			//thrust::copy(dev_data_ptr,dev_data_ptr+cnt[tid],pointInDistValue+(*totalResultSetCnt));

			//thrust with streams (but into one big buffer)
			//copy the data back using the streams
			//cudaMemcpyAsync(thrust::raw_pointer_cast(pointIDKey+(*totalResultSetCnt)), thrust::raw_pointer_cast(dev_keys_ptr), cnt[tid]*sizeof(int), cudaMemcpyDeviceToHost, stream[tid]);
			//cudaMemcpyAsync(thrust::raw_pointer_cast(pointInDistValue+(*totalResultSetCnt)), thrust::raw_pointer_cast(dev_data_ptr), cnt[tid]*sizeof(int), cudaMemcpyDeviceToHost, stream[tid]);
	  		
	  		//thrust with streams into individual buffers for each batch
			//cudaMemcpyAsync(thrust::raw_pointer_cast(batchedResultSet[i].pointIDKey), thrust::raw_pointer_cast(dev_keys_ptr), cnt[tid]*sizeof(int), cudaMemcpyDeviceToHost, stream[tid]);
			//cudaMemcpyAsync(thrust::raw_pointer_cast(batchedResultSet[i].pointInDistValue), thrust::raw_pointer_cast(dev_data_ptr), cnt[tid]*sizeof(int), cudaMemcpyDeviceToHost, stream[tid]);	

			

			//thrust with streams (but into one big buffer) where each batch can write to a different spot in the big buffer
			//as the big array is chunked into the gpu batch size  
			//copy the data back using the streams
			//FOR PINNED MEMORY 
			//cudaMemcpyAsync(thrust::raw_pointer_cast(pointIDKey+(i*GPUBufferSize)), thrust::raw_pointer_cast(dev_keys_ptr), cnt[tid]*sizeof(int), cudaMemcpyDeviceToHost, stream[tid]);
			//cudaMemcpyAsync(thrust::raw_pointer_cast(pointInDistValue+(i*GPUBufferSize)), thrust::raw_pointer_cast(dev_data_ptr), cnt[tid]*sizeof(int), cudaMemcpyDeviceToHost, stream[tid]);
	  		 
			
			//thrust with streams (but into one big buffer) where each batch can write to a different spot in the big buffer
			//as the big array is chunked into the gpu batch size  
			
			//FOR PAGED MEMORY -- cant use streams
			// cudaMemcpy(thrust::raw_pointer_cast(pointIDKey+(i*GPUBufferSize)), thrust::raw_pointer_cast(dev_keys_ptr), cnt[tid]*sizeof(int), cudaMemcpyDeviceToHost);
			// cudaMemcpy(thrust::raw_pointer_cast(pointInDistValue+(i*GPUBufferSize)), thrust::raw_pointer_cast(dev_data_ptr), cnt[tid]*sizeof(int), cudaMemcpyDeviceToHost);
	  		

	  		//thrust with streams into individual buffers for each batch
			cudaMemcpyAsync(thrust::raw_pointer_cast(pointIDKey[tid]), thrust::raw_pointer_cast(dev_keys_ptr), cnt[tid]*sizeof(int), cudaMemcpyDeviceToHost, stream[tid]);
			cudaMemcpyAsync(thrust::raw_pointer_cast(pointInDistValue[tid]), thrust::raw_pointer_cast(dev_data_ptr), cnt[tid]*sizeof(int), cudaMemcpyDeviceToHost, stream[tid]);	

			//need to make sure the data is copied before constructing portion of the neighbor table
			cudaStreamSynchronize(stream[tid]);


			
			double tableconstuctstart=omp_get_wtime();
			constructNeighborTableKeyValue(pointIDKey[tid], pointInDistValue[tid], neighborTable, &cnt[tid]);
			double tableconstuctend=omp_get_wtime();	
			
			printf("\nTable construct time: %f", tableconstuctend - tableconstuctstart);

			
			//add the batched result set size to the total count
			totalResultsLoop+=cnt[tid];

			printf("\nRunning total of total size of result array, tid: %d: %u", tid, totalResultsLoop);
			//}



		

		} //END LOOP OVER THE GPU BATCHES


	
	
	printf("\nTOTAL RESULT SET SIZE ON HOST:  %u", totalResultsLoop);
	*totalNeighbors=totalResultsLoop;


	double tKernelResultsEnd=omp_get_wtime();
	
	printf("\nTime to launch kernel and execute all of the previous part of the method and get the results back: %f",tKernelResultsEnd-tKernelResultsStart);






	//SORTING FOR TESTING ONLY
	//XXXXXXXXX
	//XXXXXXXXX
	// std::sort(results, results+(*cnt),compResults);
	// printf("\n**** GPU\n");
	// for (int i=0; i<(*cnt); i++)
	// {
	// 	printf("\n%d,%d",results[i].pointID, results[i].pointInDist);
	// }

	//XXXXXXXXX
	//XXXXXXXXX
	//XXXXXXXXX


	//get debug information (optional)
	// unsigned int * debug1;
	// debug1=(unsigned int*)malloc(sizeof(unsigned int));
	// *debug1=0;
	// unsigned int * debug2;
	// debug2=(unsigned int*)malloc(sizeof(unsigned int));
	// *debug2=0;

	
	double tStartdebug=omp_get_wtime();

	errCode=cudaMemcpy(debug1, dev_debug1, sizeof(unsigned int), cudaMemcpyDeviceToHost );
	
	if(errCode != cudaSuccess) {
	cout << "\nError: getting debug1 from GPU Got error with code " << errCode << endl; 
	}
	else
	{
		printf("\nDebug1 value: %u",*debug1);
	}

	errCode=cudaMemcpy(debug2, dev_debug2, sizeof(unsigned int), cudaMemcpyDeviceToHost );
	
	if(errCode != cudaSuccess) {
	cout << "\nError: getting debug1 from GPU Got error with code " << errCode << endl; 
	}
	else
	{
		printf("\nDebug2 value: %u",*debug2);
	}	

	double tEnddebug=omp_get_wtime();
	printf("\nTime to retrieve debug values: %f", tEnddebug - tStartdebug);
	

	///////////////////////////////////
	//END GET RESULT SET
	///////////////////////////////////


	///////////////////////////////////
	//FREE MEMORY FROM THE GPU
	///////////////////////////////////
    //free:
	
	

	


	double tFreeStart=omp_get_wtime();



	//destroy streams
	// for (int i=0; i<GPUSTREAMS; i++)
	// {
	// cudaStreamDestroy(stream[i]);
	// }

	for (int i=0; i<GPUSTREAMS; i++)
	{
		errCode=cudaStreamDestroy(stream[i]);

		if(errCode != cudaSuccess) {
		cout << "\nError: destroying stream" << errCode << endl; 
		}
	}


	







	//free the data on the device
	cudaFree(dev_pointIDKey);
	cudaFree(dev_pointInDistValue);

    
	cudaFree(dev_database);
	cudaFree(dev_debug1);
	cudaFree(dev_debug2);
	
	cudaFree(dev_epsilon);
	cudaFree(dev_grid);
	cudaFree(dev_lookupArr);
	cudaFree(dev_gridNumXCells);
	cudaFree(dev_gridNumYCells);
	cudaFree(dev_gridMin_x);
	cudaFree(dev_gridMin_y);
	
	cudaFree(dev_N); 	
	cudaFree(dev_cnt); 
	cudaFree(dev_offset); 
	cudaFree(dev_batchNumber); 

	
		//free data related to the individual streams for each batch
		for (int i=0; i<GPUSTREAMS; i++)
		{
			//free the data on the device
			cudaFree(dev_pointIDKey[i]);
			cudaFree(dev_pointInDistValue[i]);

			//free on the host
			cudaFreeHost(pointIDKey[i]);
			cudaFreeHost(pointInDistValue[i]);

		// errCode=cudaMalloc((void **)&dev_pointIDKey[i], sizeof(int)*GPUBufferSize);
		// if(errCode != cudaSuccess) {
		// cout << "CUDA: Got error with code " << errCode << endl; //2 means not enough memory
		// }

		// errCode=cudaMalloc((void **)&dev_pointInDistValue[i], sizeof(int)*GPUBufferSize);
		// if(errCode != cudaSuccess) {
		// cout << "CUDA: Got error with code " << errCode << endl; //2 means not enough memory
		// }

		}

	

	cudaFree(dev_pointIDKey);
	cudaFree(dev_pointInDistValue);
	//free pinned memory on host
	cudaFreeHost(pointIDKey);
	cudaFreeHost(pointInDistValue);

	
	
	

	 

	


	double tFreeEnd=omp_get_wtime();

	printf("\nTime freeing memory: %f", tFreeEnd - tFreeStart);

	

		

	// printf("\nreturning before constructing table (which is commented)");
	// return;


	////////////////////////////////////
	//CONSTRUCT TABLE
	////////////////////////////////////	

	//NOW CONSTRUCT THE TABLE PARTIALLY WHEN MAKING THE BATCHES


	// double tStartTableConstruct=omp_get_wtime();
	// constructNeighborTableKeyValue(pointIDKey, pointInDistValue, neighborTable, totalResultSetCnt);
	// double tEndTableConstruct=omp_get_wtime();	
	// printf("\nTime constructing table: %f",tEndTableConstruct - tStartTableConstruct);	
	

	//print table:
	
	/*
	int tmpcnt=0;
	printf("\nGrid GPU Table**********");
	for (int i=0; i<(*DBSIZE); i++)
	{
		printf("\nPoint id: %d In distance: ", neighborTable[i].pointID);

		//sort so it has the same output:
		std::sort(neighborTable[i].neighbors.begin(),neighborTable[i].neighbors.end());
		for (int j=0; j<neighborTable[i].neighbors.size();j++)
		{
			printf("%d, ",neighborTable[i].neighbors[j]);
			tmpcnt++;
		}
	}

	printf("\n count elems: %d", tmpcnt);
	*/
	
	
	////////////////////////////////////
	//END CONSTRUCT TABLE
	////////////////////////////////////	

	
	
	//printf("\ntotal neighbors (in batched fn): %d", *totalNeighbors);

	cout<<"\n** last error at end of fn batches: "<<cudaGetLastError();
	// printf("\nExiting function early..");
	// return;


}




























//In this function we batch the results off of the GPU to accomodate larger epsilon values
//The results that come from the GPU are in the form of key/value pairs (in two arrays)
//Key-a point, Value-a point within epsilon of the key 
//The batches are mapped to differing streams
//Each batch requires its own memory space for the result set
//So the number of buffers on the GPU for the results is the number of streams (GPUSTREAMS)
//On the host, we use the same size buffers, and number of them, and then build part of the neighbor table with the batch

//This is an alternative to making one large array from all of the batches, which would require a large
//pinned cuda malloc which is very expensive.  It also allows for multiple threads to concurrently build the 
//neighbor table and interleave GPU work with work on the CPU

//Also, the number of batches is estimated by calling a kernel that samples the number of neighbours (1%) and then
//estimates the total neighbors, which is used to calculate the total number of batches
//To make sure each batch doesn't vary much, we use a strided scheme for each batch

//Uses the grid index to compute the direct neighbor table
//NO SHARED MEMORY PAGING

void makeDistanceTableGPUGridIndexBatchesAlternateTest(std::vector<struct dataElem> * dataPoints, double * epsilon, struct grid * index, double * gridMin_x, double * gridMin_y, int * gridNumXCells, int * gridNumYCells, unsigned int * lookupArr, struct neighborTableLookup * neighborTable, std::vector<struct neighborDataPtrs> * pointersToNeighbors, unsigned int * totalNeighbors)
{

	


	double tKernelResultsStart=omp_get_wtime();

	//CUDA error code:
	cudaError_t errCode;


	cout<<"\n** last error start of fn: "<<cudaGetLastError();


	




	///////////////////////////////////
	//COPY THE DATABASE TO THE GPU
	///////////////////////////////////


	unsigned int * DBSIZE;
	DBSIZE=(unsigned int*)malloc(sizeof(unsigned int));
	*DBSIZE=dataPoints->size();
	
	printf("\n in main GPU method: DBSIZE is: %u",*DBSIZE);cout.flush();

	struct point * database;
	
	//pinned memory for the database:
	database=(struct point*)malloc(sizeof(struct point)*(*DBSIZE));
	//dont use pinned memory for the database, its slower than using cudaMalloc
	//cudaMallocHost((void **) &database, sizeof(struct point)*(*DBSIZE));


	struct point * dev_database;
	dev_database=(struct point*)malloc(sizeof(struct point)*(*DBSIZE));

	//allocate memory on device:
	
	errCode=cudaMalloc( (void**)&dev_database, sizeof(struct point)*(*DBSIZE));

	printf("\n !!in main GPU method: DBSIZE is: %u",*DBSIZE);cout.flush();	
		
	if(errCode != cudaSuccess) {
	cout << "\nError: database Got error with code " << errCode << endl; cout.flush(); 
	}


	



	
	//first, we copy the x and y values from dataPoints to the database
	//we do this because the data points struct may contain other values than x and y
	for (int i=0; i<(*DBSIZE); i++)
	{
		database[i].x=(*dataPoints)[i].x;
		database[i].y=(*dataPoints)[i].y;
	}







	//printf("\n size of database: %d",N);

	//copy database to the device:
	errCode=cudaMemcpy(dev_database, database, sizeof(struct point)*(*DBSIZE), cudaMemcpyHostToDevice);	
	if(errCode != cudaSuccess) {
	cout << "\nError: database Got error with code " << errCode << endl; 
	}

	///////////////////////////////////
	//END COPY THE DATABASE TO THE GPU
	///////////////////////////////////




	///////////////////////////////////
	//COPY THE INDEX TO THE GPU
	///////////////////////////////////

	


	int totalGridCells=(*gridNumXCells)*(*gridNumYCells);

	struct grid * dev_grid;
	dev_grid=(struct grid*)malloc(sizeof(struct grid)*totalGridCells);

	//allocate memory on device:
	errCode=cudaMalloc( (void**)&dev_grid, sizeof(struct grid)*totalGridCells);
	
		
	if(errCode != cudaSuccess) {
	cout << "\nError: grid index Got error with code " << errCode << endl; cout.flush(); 
	}

	//copy grid index to the device:
	errCode=cudaMemcpy(dev_grid, index, sizeof(struct grid)*totalGridCells, cudaMemcpyHostToDevice);	
	if(errCode != cudaSuccess) {
	cout << "\nError: grid index allocation Got error with code " << errCode << endl; 
	}	

	printf("\nSize of index sent to GPU (GiB): %f", (double)sizeof(struct grid)*totalGridCells/(1024.0*1024.0*1024.0));

	///////////////////////////////////
	//END COPY THE INDEX TO THE GPU
	///////////////////////////////////


	///////////////////////////////////
	//COPY THE LOOKUP ARRAY TO THE GPU
	///////////////////////////////////

	


	int * dev_lookupArr;
	dev_lookupArr=(int*)malloc(sizeof(int)*(*DBSIZE));

	//allocate memory on device:
	errCode=cudaMalloc( (void**)&dev_lookupArr, sizeof(int)*(*DBSIZE));

	if(errCode != cudaSuccess) {
	cout << "\nError: lookup array Got error with code " << errCode << endl; cout.flush(); 
	}

	//copy lookup array to the device:
	errCode=cudaMemcpy(dev_lookupArr, lookupArr, sizeof(int)*(*DBSIZE), cudaMemcpyHostToDevice);	
	if(errCode != cudaSuccess) {
	cout << "\nError: lookup array allocation Got error with code " << errCode << endl; 
	}	


	///////////////////////////////////
	//END COPY THE LOOKUP ARRAY TO THE GPU
	///////////////////////////////////


	///////////////////////////////////
	//COPY GRID DIMENSIONS TO THE GPU
	//THIS INCLUDES THE NUMBER OF CELLS IN EACH X AND Y DIMENSION, 
	//AND THE STARTING POINT IN THE X AND Y DIMENSIONS THAT THE GRID STARTS AT
	///////////////////////////////////

	//The minimum x boundary of the grid:
	//gridMin_x
	double * dev_gridMin_x;
	dev_gridMin_x=(double*)malloc(sizeof( double ));

	//Allocate on the device
	errCode=cudaMalloc((void**)&dev_gridMin_x, sizeof(double));
	if(errCode != cudaSuccess) {
	cout << "\nError: gridMin_x Got error with code " << errCode << endl; 
	}

	errCode=cudaMemcpy( dev_gridMin_x, gridMin_x, sizeof(double), cudaMemcpyHostToDevice );
	if(errCode != cudaSuccess) {
	cout << "\nError: gridMin_x Got error with code " << errCode << endl; 
	}	


	//The minimum y boundary of the grid:
	//gridMin_y
	double * dev_gridMin_y;
	dev_gridMin_y=(double*)malloc(sizeof( double ));

	//Allocate on the device
	errCode=cudaMalloc((void**)&dev_gridMin_y, sizeof(double));
	if(errCode != cudaSuccess) {
	cout << "\nError: gridMin_y Got error with code " << errCode << endl; 
	}

	errCode=cudaMemcpy( dev_gridMin_y, gridMin_y, sizeof(double), cudaMemcpyHostToDevice );
	if(errCode != cudaSuccess) {
	cout << "\nError: gridMin_y Got error with code " << errCode << endl; 
	}	


	//The number of cells in the x dimension:
	//gridNumXCells
	int * dev_gridNumXCells; 
	dev_gridNumXCells=(int*)malloc(sizeof(int));
	*dev_gridNumXCells=0;

	//allocate on the device
	errCode=cudaMalloc((int**)&dev_gridNumXCells, sizeof(int));	
	if(errCode != cudaSuccess) {
	cout << "\nError: dev_gridNumXCells Got error with code " << errCode << endl; 
	}


	errCode=cudaMemcpy( dev_gridNumXCells, gridNumXCells, sizeof(int), cudaMemcpyHostToDevice );
	if(errCode != cudaSuccess) {
	cout << "\nError: dev_gridNumXCells memcpy Got error with code " << errCode << endl; 
	}


	//The number of cells in the y dimension:
	//gridNumYCells
	int * dev_gridNumYCells; 
	dev_gridNumYCells=(int*)malloc(sizeof(int));
	*dev_gridNumYCells=0;

	//allocate on the device
	errCode=cudaMalloc((int**)&dev_gridNumYCells, sizeof(int));	
	if(errCode != cudaSuccess) {
	cout << "\nError: dev_gridNumYCells Got error with code " << errCode << endl; 
	}


	errCode=cudaMemcpy( dev_gridNumYCells, gridNumYCells, sizeof(int), cudaMemcpyHostToDevice );
	if(errCode != cudaSuccess) {
	cout << "\nError: dev_gridNumYCells memcpy Got error with code " << errCode << endl; 
	}



	///////////////////////////////////
	//END COPY GRID DIMENSIONS TO THE GPU
	///////////////////////////////////






	









	///////////////////////////////////
	//SET OTHER KERNEL PARAMETERS
	///////////////////////////////////

	//total size of the result set as it's batched
	//this isnt sent to the GPU
	unsigned int * totalResultSetCnt;
	totalResultSetCnt=(unsigned int*)malloc(sizeof(unsigned int));
	*totalResultSetCnt=0;

	//count values - for an individual kernel launch
	//need different count values for each stream
	unsigned int * cnt;
	cnt=(unsigned int*)malloc(sizeof(unsigned int)*GPUSTREAMS);
	*cnt=0;

	unsigned int * dev_cnt; 
	dev_cnt=(unsigned int*)malloc(sizeof(unsigned int)*GPUSTREAMS);
	*dev_cnt=0;

	//allocate on the device
	errCode=cudaMalloc((unsigned int**)&dev_cnt, sizeof(unsigned int)*GPUSTREAMS);	
	if(errCode != cudaSuccess) {
	cout << "\nError: cnt Got error with code " << errCode << endl; 
	}


	// errCode=cudaMemcpy( dev_cnt, cnt, sizeof(unsigned int), cudaMemcpyHostToDevice );
	// if(errCode != cudaSuccess) {
	// cout << "\nError: dev_cnt memcpy Got error with code " << errCode << endl; 
	// }

	
	//Epsilon
	double * dev_epsilon;
	dev_epsilon=(double*)malloc(sizeof( double ));
	//*dev_epsilon=*epsilon;

	//Allocate on the device
	errCode=cudaMalloc((void**)&dev_epsilon, sizeof(double));
	if(errCode != cudaSuccess) {
	cout << "\nError: epsilon Got error with code " << errCode << endl; 
	}

	//copy to device
	errCode=cudaMemcpy( dev_epsilon, epsilon, sizeof(double), cudaMemcpyHostToDevice );
	if(errCode != cudaSuccess) {
	cout << "\nError: epsilon Got error with code " << errCode << endl; 
	}		




	
	//number of threads per gpu stream

	//THE NUMBER OF THREADS THAT ARE LAUNCHED IN A SINGLE KERNEL INVOCATION
	//CAN BE FEWER THAN THE NUMBER OF ELEMENTS IN THE DATABASE IF MORE THAN 1 BATCH
	unsigned int * N;
	N=(unsigned int*)malloc(sizeof(unsigned int)*GPUSTREAMS);
	


	unsigned int * dev_N; 
	dev_N=(unsigned int*)malloc(sizeof(unsigned int)*GPUSTREAMS);

	//allocate on the device
	errCode=cudaMalloc((void**)&dev_N, sizeof(unsigned int)*GPUSTREAMS);
	if(errCode != cudaSuccess) {
	cout << "\nError: dev_N Got error with code " << errCode << endl; 
	}	


	//offset into the database when batching the results
	unsigned int * batchOffset; 
	batchOffset=(unsigned int*)malloc(sizeof(unsigned int)*GPUSTREAMS);
	//*batchOffset=0;


	unsigned int * dev_offset; 
	dev_offset=(unsigned int*)malloc(sizeof(unsigned int)*GPUSTREAMS);

	//allocate on the device
	errCode=cudaMalloc((void**)&dev_offset, sizeof(unsigned int)*GPUSTREAMS);
	if(errCode != cudaSuccess) {
	cout << "\nError: offset Got error with code " << errCode << endl; 
	}

	//Batch number to calculate the point to process (in conjunction with the offset)

	//offset into the database when batching the results
	unsigned int * batchNumber; 
	batchNumber=(unsigned int*)malloc(sizeof(unsigned int)*GPUSTREAMS);
	//*batchOffset=0;


	unsigned int * dev_batchNumber; 
	dev_batchNumber=(unsigned int*)malloc(sizeof(unsigned int)*GPUSTREAMS);

	//allocate on the device
	errCode=cudaMalloc((void**)&dev_batchNumber, sizeof(unsigned int)*GPUSTREAMS);
	if(errCode != cudaSuccess) {
	cout << "\nError: batchNumber Got error with code " << errCode << endl; 
	}


			

	//debug values
	unsigned int * dev_debug1; 
	dev_debug1=(unsigned int *)malloc(sizeof(unsigned int ));
	*dev_debug1=0;

	unsigned int * dev_debug2; 
	dev_debug2=(unsigned int *)malloc(sizeof(unsigned int ));
	*dev_debug2=0;

	unsigned int * debug1; 
	debug1=(unsigned int *)malloc(sizeof(unsigned int ));
	*debug1=0;

	unsigned int * debug2; 
	debug2=(unsigned int *)malloc(sizeof(unsigned int ));
	*debug2=0;



	//allocate on the device
	errCode=cudaMalloc( (unsigned int **)&dev_debug1, sizeof(unsigned int ) );
	if(errCode != cudaSuccess) {
	cout << "\nError: debug1 Got error with code " << errCode << endl; 
	}		
	errCode=cudaMalloc( (unsigned int **)&dev_debug2, sizeof(unsigned int ) );
	if(errCode != cudaSuccess) {
	cout << "\nError: debug2 Got error with code " << errCode << endl; 
	}		

	//set to 0
	//copy debug to device
	errCode=cudaMemcpy( dev_debug1, debug1, sizeof(unsigned int), cudaMemcpyHostToDevice );
	if(errCode != cudaSuccess) {
	cout << "\nError: dev_debug1 Got error with code " << errCode << endl; 
	}

	errCode=cudaMemcpy( dev_debug2, debug2, sizeof(unsigned int), cudaMemcpyHostToDevice );
	if(errCode != cudaSuccess) {
	cout << "\nError: dev_debug2 Got error with code " << errCode << endl; 
	}



	//////////////////////////////////////////////////////////
	//ESTIMATE THE BUFFER SIZE AND NUMBER OF BATCHES ETC BY COUNTING THE NUMBER OF RESULTS
	//TAKE A SAMPLE OF THE DATA POINTS, NOT ALL OF THEM
	//Use sampleRate for this
	/////////////////////////////////////////////////////////

	printf("\n\n***********************************\nEstimating Batches:");
	//Parameters for the batch size estimation.
	double sampleRate=0.01; //sample 1% of the points in the dataset sampleRate=0.01. 
						//Sample the entire dataset(no sampling) sampleRate=1
	int offsetRate=1.0/sampleRate;
	printf("\nOffset: %d", offsetRate);


	/////////////////
	//N-threads
	////////////////

	
	double tstartbatchest=omp_get_wtime();

	unsigned int * dev_N_batchEst; 
	dev_N_batchEst=(unsigned int*)malloc(sizeof(unsigned int));

	unsigned int * N_batchEst; 
	N_batchEst=(unsigned int*)malloc(sizeof(unsigned int));
	*N_batchEst=*DBSIZE*sampleRate;


	//allocate on the device
	errCode=cudaMalloc((void**)&dev_N_batchEst, sizeof(unsigned int));
	if(errCode != cudaSuccess) {
	cout << "\nError: dev_N_batchEst Got error with code " << errCode << endl; 
	}	

	//copy N to device 
	//N IS THE NUMBER OF THREADS
	errCode=cudaMemcpy( dev_N_batchEst, N_batchEst, sizeof(unsigned int), cudaMemcpyHostToDevice);
	if(errCode != cudaSuccess) {
	cout << "\nError: N batchEST Got error with code " << errCode << endl; 
	}


	/////////////
	//count the result set size 
	////////////

	unsigned int * dev_cnt_batchEst; 
	dev_cnt_batchEst=(unsigned int*)malloc(sizeof(unsigned int));

	unsigned int * cnt_batchEst; 
	cnt_batchEst=(unsigned int*)malloc(sizeof(unsigned int));
	*cnt_batchEst=0;


	//allocate on the device
	errCode=cudaMalloc((void**)&dev_cnt_batchEst, sizeof(unsigned int));
	if(errCode != cudaSuccess) {
	cout << "\nError: dev_cnt_batchEst Got error with code " << errCode << endl; 
	}	

	//copy cnt to device 
	errCode=cudaMemcpy( dev_cnt_batchEst, cnt_batchEst, sizeof(unsigned int), cudaMemcpyHostToDevice);
	if(errCode != cudaSuccess) {
	cout << "\nError: dev_cnt_batchEst Got error with code " << errCode << endl; 
	}

	
	//////////////////
	//SAMPLE OFFSET - TO SAMPLE THE DATA TO ESTIMATE THE TOTAL NUMBER OF KEY VALUE PAIRS
	/////////////////

	//offset into the database when batching the results
	unsigned int * sampleOffset; 
	sampleOffset=(unsigned int*)malloc(sizeof(unsigned int));
	*sampleOffset=offsetRate;


	unsigned int * dev_sampleOffset; 
	dev_sampleOffset=(unsigned int*)malloc(sizeof(unsigned int));

	//allocate on the device
	errCode=cudaMalloc((void**)&dev_sampleOffset, sizeof(unsigned int));
	if(errCode != cudaSuccess) {
	cout << "\nError: sample offset Got error with code " << errCode << endl; 
	}

	//copy offset to device 
	errCode=cudaMemcpy( dev_sampleOffset, sampleOffset, sizeof(unsigned int), cudaMemcpyHostToDevice);
	if(errCode != cudaSuccess) {
	cout << "\nError: dev_cnt_batchEst Got error with code " << errCode << endl; 
	}



	const int TOTALBLOCKSBATCHEST=ceil((1.0*(*DBSIZE)*sampleRate)/(1.0*BLOCKSIZE));	
	printf("\ntotal blocks: %d",TOTALBLOCKSBATCHEST);

	kernelGridIndexBatchEstimator<<< TOTALBLOCKSBATCHEST, BLOCKSIZE>>>(dev_N_batchEst, dev_sampleOffset, dev_debug1, dev_debug2, dev_epsilon, dev_grid, dev_gridMin_x, dev_gridMin_y, dev_gridNumXCells, dev_gridNumYCells, dev_lookupArr, dev_cnt_batchEst, dev_database);
	cout<<"\n** ERROR FROM KERNEL LAUNCH OF BATCH ESTIMATOR: "<<cudaGetLastError();
	// find the size of the number of results
		errCode=cudaMemcpy( cnt_batchEst, dev_cnt_batchEst, sizeof(unsigned int), cudaMemcpyDeviceToHost);
		if(errCode != cudaSuccess) {
		cout << "\nError: getting cnt for batch estimate from GPU Got error with code " << errCode << endl; 
		}
		else
		{
			printf("\nGPU: result set size for estimating the number of batches (sampled): %u",*cnt_batchEst);
		}




	cudaFree(dev_cnt_batchEst);	
	cudaFree(dev_N_batchEst);
	cudaFree(dev_sampleOffset);



	double tendbatchest=omp_get_wtime();

	printf("\nTime to get the total result set size from batch estimator: %f",tendbatchest-tstartbatchest);

	




	//WE CALCULATE THE BUFFER SIZES AND NUMBER OF BATCHES

	unsigned int GPUBufferSize=100000000;
	double alpha=0.05; //overestimation factor

	unsigned long long estimatedTotalSize=(unsigned long long)(*cnt_batchEst)*(unsigned long long)offsetRate;
	unsigned long long estimatedTotalSizeWithAlpha=(unsigned long long)(*cnt_batchEst)*(unsigned long long)offsetRate*(1.0+(alpha));
	printf("\nEstimated total result set size: %llu", estimatedTotalSize);
	printf("\nEstimated total result set size (with Alpha %f): %llu", alpha,estimatedTotalSizeWithAlpha);
	

	//to accomodate small datasets, we need smaller buffers because the pinned memory malloc is expensive
	if (estimatedTotalSize<(GPUBufferSize*GPUSTREAMS))
	{
		GPUBufferSize=estimatedTotalSize*(1.0+(alpha*2.0))/(GPUSTREAMS);		//we do 2*alpha for small datasets because the
																		//sampling will be worse for small datasets
																		//but we fix the 3 streams still (thats why divide by 3).			
	}

	unsigned int numBatches=ceil(((1.0+alpha)*estimatedTotalSize*1.0)/(GPUBufferSize*1.0));
	printf("\nNumber of batches: %d, buffer size: %d", numBatches, GPUBufferSize);


		

	printf("\nEnd Batch Estimator\n***********************************\n");

	/////////////////////////////////////////////////////////	
	//END BATCH ESTIMATOR	
	/////////////////////////////////////////////////////////












	///////////////////
	//ALLOCATE POINTERS TO INTEGER ARRAYS FOR THE VALUES FOR THE NEIGHBORTABLES
	///////////////////

	//THE NUMBER OF POINTERS IS EQUAL TO THE NUMBER OF BATCHES
	for (int i=0; i<numBatches; i++)
	{
		int *ptr;
		struct neighborDataPtrs tmpStruct;
		tmpStruct.dataPtr=ptr;
		tmpStruct.sizeOfDataArr=0;
		
		pointersToNeighbors->push_back(tmpStruct);
	}

	///////////////////
	//END ALLOCATE POINTERS TO INTEGER ARRAYS FOR THE VALUES FOR THE NEIGHBORTABLES
	///////////////////











	///////////////////////////////////
	//ALLOCATE MEMORY FOR THE RESULT SET USING THE BATCH ESTIMATOR
	///////////////////////////////////
	

	//NEED BUFFERS ON THE GPU AND THE HOST FOR THE NUMBER OF CONCURRENT STREAMS	
	//GPU BUFFER ON THE DEVICE
	//BUFFER ON THE HOST WITH PINNED MEMORY FOR FAST MEMCPY
	//BUFFER ON THE HOST TO DUMP THE RESULTS OF BATCHES SO THAT GPU THREADS CAN CONTINUE
	//EXECUTING STREAMS ON THE HOST



	//GPU MEMORY ALLOCATION:

	//CHANGING THE RESULTS TO KEY VALUE PAIR SORT, WHICH IS TWO ARRAYS
	//KEY IS THE POINT ID
	//THE VALUE IS THE POINT ID WITHIN THE DISTANCE OF KEY

	int * dev_pointIDKey[GPUSTREAMS]; //key
	int * dev_pointInDistValue[GPUSTREAMS]; //value

	

	for (int i=0; i<GPUSTREAMS; i++)
	{
		errCode=cudaMalloc((void **)&dev_pointIDKey[i], sizeof(int)*GPUBufferSize);
		if(errCode != cudaSuccess) {
		cout << "CUDA: Got error with code " << errCode << endl; //2 means not enough memory
		}

		errCode=cudaMalloc((void **)&dev_pointInDistValue[i], sizeof(int)*GPUBufferSize);
		if(errCode != cudaSuccess) {
		cout << "CUDA: Got error with code " << errCode << endl; //2 means not enough memory
		}


	}


	

	//HOST RESULT ALLOCATION FOR THE GPU TO COPY THE DATA INTO A PINNED MEMORY ALLOCATION
	//ON THE HOST
	//pinned result set memory for the host
	//the number of elements are recorded for that batch in resultElemCountPerBatch
	//NEED PINNED MEMORY ALSO BECAUSE YOU NEED IT TO USE STREAMS IN THRUST FOR THE MEMCOPY OF THE SORTED RESULTS	

	//PINNED MEMORY TO COPY FROM THE GPU	
	int * pointIDKey[GPUSTREAMS]; //key
	int * pointInDistValue[GPUSTREAMS]; //value
	

	double tstartpinnedresults=omp_get_wtime();
	
	for (int i=0; i<GPUSTREAMS; i++)
	{
	cudaMallocHost((void **) &pointIDKey[i], sizeof(int)*GPUBufferSize);
	cudaMallocHost((void **) &pointInDistValue[i], sizeof(int)*GPUBufferSize);
	}

	double tendpinnedresults=omp_get_wtime();
	printf("\nTime to allocate pinned memory for results: %f", tendpinnedresults - tstartpinnedresults);
	

	// cudaMalloc((void **) &pointIDKey, sizeof(int)*GPUBufferSize*NUMBATCHES);
	// cudaMalloc((void **) &pointInDistValue, sizeof(int)*GPUBufferSize*NUMBATCHES);




	printf("\nmemory requested for results ON GPU (GiB): %f",(double)(sizeof(int)*2*GPUBufferSize*GPUSTREAMS)/(1024*1024*1024));
	printf("\nmemory requested for results in MAIN MEMORY (GiB): %f",(double)(sizeof(int)*2*GPUBufferSize*GPUSTREAMS)/(1024*1024*1024));

	
	///////////////////////////////////
	//END ALLOCATE MEMORY FOR THE RESULT SET
	///////////////////////////////////











	



	/////////////////////////////////
	//SET OPENMP ENVIRONMENT VARIABLES
	////////////////////////////////

	omp_set_num_threads(GPUSTREAMS);
	

	/////////////////////////////////
	//END SET OPENMP ENVIRONMENT VARIABLES
	////////////////////////////////
	
	

	/////////////////////////////////
	//CREATE STREAMS
	////////////////////////////////

	cudaStream_t stream[GPUSTREAMS];
	
	for (int i=0; i<GPUSTREAMS; i++){

    //cudaStreamCreate(&stream[i]);
	cudaStreamCreateWithFlags(&stream[i], cudaStreamNonBlocking);

	}	

	/////////////////////////////////
	//END CREATE STREAMS
	////////////////////////////////
	
	

	///////////////////////////////////
	//LAUNCH KERNEL IN BATCHES
	///////////////////////////////////
		
	//since we use the strided scheme, some of the batch sizes
	//are off by 1 of each other, a first group of batches will
	//have 1 extra data point to process, and we calculate which batch numbers will 
	//have that.  The batchSize is the lower value (+1 is added to the first ones)


	unsigned int batchSize=(*DBSIZE)/numBatches;
	unsigned int batchesThatHaveOneMore=(*DBSIZE)-(batchSize*numBatches); //batch number 0- < this value have one more
	printf("\n\n***Batches that have one more: %u batchSize(N): %u, \n\n***",batchSize, batchesThatHaveOneMore);

	unsigned int totalResultsLoop=0;


		
		//FOR LOOP OVER THE NUMBER OF BATCHES STARTS HERE
		#pragma omp parallel for schedule(static,1) reduction(+:totalResultsLoop) num_threads(GPUSTREAMS)
		for (int i=0; i<numBatches; i++)
		{	

			int tid=omp_get_thread_num();
			
			printf("\ntid: %d, starting iteration: %d",tid,i);

			//N NOW BECOMES THE NUMBER OF POINTS TO PROCESS PER BATCH
			//AS ONE THREAD PROCESSES A SINGLE POINT
			



			
			if (i<batchesThatHaveOneMore)
			{
				N[tid]=batchSize+1;	
				printf("\nN: %d, tid: %d",N[tid], tid);
			}
			else
			{
				N[tid]=batchSize;	
				printf("\nN (1 less): %d tid: %d",N[tid], tid);
			}

			//set relevant parameters for the batched execution that get reset
			
			//copy N to device 
			//N IS THE NUMBER OF THREADS
			errCode=cudaMemcpyAsync( &dev_N[tid], &N[tid], sizeof(unsigned int), cudaMemcpyHostToDevice, stream[tid] );
			if(errCode != cudaSuccess) {
			cout << "\nError: N Got error with code " << errCode << endl; 
			}

			//the batched result set size (reset to 0):
			cnt[tid]=0;
			errCode=cudaMemcpyAsync( &dev_cnt[tid], &cnt[tid], sizeof(unsigned int), cudaMemcpyHostToDevice, stream[tid] );
			if(errCode != cudaSuccess) {
			cout << "\nError: dev_cnt memcpy Got error with code " << errCode << endl; 
			}

			//the offset for batching, which keeps track of where to start processing at each batch
			//batchOffset[tid]=i*batchSize; //original
			batchOffset[tid]=numBatches; //for the strided
			errCode=cudaMemcpyAsync( &dev_offset[tid], &batchOffset[tid], sizeof(unsigned int), cudaMemcpyHostToDevice, stream[tid] );
			if(errCode != cudaSuccess) {
			cout << "\nError: dev_offset memcpy Got error with code " << errCode << endl; 
			}

			//the batch number for batching with strided
			batchNumber[tid]=i;
			errCode=cudaMemcpyAsync( &dev_batchNumber[tid], &batchNumber[tid], sizeof(unsigned int), cudaMemcpyHostToDevice, stream[tid] );
			if(errCode != cudaSuccess) {
			cout << "\nError: dev_batchNumber memcpy Got error with code " << errCode << endl; 
			}

			const int TOTALBLOCKS=ceil((1.0*(N[tid]))/(1.0*BLOCKSIZE));	
			printf("\ntotal blocks: %d",TOTALBLOCKS);

			//execute kernel	
			//0 is shared memory pool
			kernelGridIndex<<< TOTALBLOCKS, BLOCKSIZE, 0, stream[tid] >>>(&dev_N[tid], &dev_offset[tid], &dev_batchNumber[tid], dev_debug1, dev_debug2, dev_epsilon, dev_grid, dev_gridMin_x, dev_gridMin_y, dev_gridNumXCells, dev_gridNumYCells, dev_lookupArr, &dev_cnt[tid], dev_database, dev_pointIDKey[tid], dev_pointInDistValue[tid]);
			
			// errCode=cudaDeviceSynchronize();
			// cout <<"\n\nError from device synchronize: "<<errCode;

			cout <<"\n\nKERNEL LAUNCH RETURN: "<<cudaGetLastError()<<endl<<endl;
			if ( cudaSuccess != cudaGetLastError() ){
		    	cout <<"\n\nERROR IN KERNEL LAUNCH. ERROR: "<<cudaSuccess<<endl<<endl;
		    }

		    

		   
			// find the size of the number of results
			errCode=cudaMemcpyAsync( &cnt[tid], &dev_cnt[tid], sizeof(unsigned int), cudaMemcpyDeviceToHost, stream[tid] );
			if(errCode != cudaSuccess) {
			cout << "\nError: getting cnt from GPU Got error with code " << errCode << endl; 
			}
			else
			{
				printf("\nGPU: result set size within epsilon (GPU grid): %d",cnt[tid]);
			}


			
			


			////////////////////////////////////
			//SORT THE TABLE DATA ON THE GPU
			//THERE IS NO ORDERING BETWEEN EACH POINT AND THE ONES THAT IT'S WITHIN THE DISTANCE OF
			////////////////////////////////////

			/////////////////////////////
			//ONE PROBLEM WITH NOT TRANSFERING THE RESULT OFF OF THE DEVICE IS THAT
			//YOU CAN'T RESIZE THE RESULTS TO BE THE SIZE OF *CNT
			//SO THEN YOU HAVE POTENTIALLY LOTS OF WASTED SPACE
			/////////////////////////////

			//sort by key with the data already on the device:
			//wrap raw pointer with a device_ptr to use with Thrust functions
			thrust::device_ptr<int> dev_keys_ptr(dev_pointIDKey[tid]);
			thrust::device_ptr<int> dev_data_ptr(dev_pointInDistValue[tid]);

			//XXXXXXXXXXXXXXXX
			//THRUST USING STREAMS REQUIRES THRUST V1.8 
			//SEEMS TO BE WORKING :)
			//XXXXXXXXXXXXXXXX

			try{
			thrust::sort_by_key(thrust::cuda::par.on(stream[tid]), dev_keys_ptr, dev_keys_ptr + cnt[tid], dev_data_ptr);
			}
			catch(std::bad_alloc &e)
			  {
			    std::cerr << "Ran out of memory while sorting" << std::endl;
			    exit(-1);
			  }
			

	  		//thrust with streams into individual buffers for each batch
			cudaMemcpyAsync(thrust::raw_pointer_cast(pointIDKey[tid]), thrust::raw_pointer_cast(dev_keys_ptr), cnt[tid]*sizeof(int), cudaMemcpyDeviceToHost, stream[tid]);
			cudaMemcpyAsync(thrust::raw_pointer_cast(pointInDistValue[tid]), thrust::raw_pointer_cast(dev_data_ptr), cnt[tid]*sizeof(int), cudaMemcpyDeviceToHost, stream[tid]);	

			//need to make sure the data is copied before constructing portion of the neighbor table
			cudaStreamSynchronize(stream[tid]);


			
			double tableconstuctstart=omp_get_wtime();
			//set the number of neighbors in the pointer struct:
			(*pointersToNeighbors)[i].sizeOfDataArr=cnt[tid];    
			(*pointersToNeighbors)[i].dataPtr=new int[cnt[tid]]; 
			constructNeighborTableKeyValueAlternateTest(pointIDKey[tid], pointInDistValue[tid], neighborTable, (*pointersToNeighbors)[i].dataPtr, &cnt[tid]);
			
			//cout <<"\nIn make neighbortable. Data array ptr: "<<(*pointersToNeighbors)[i].dataPtr<<" , size of data array: "<<(*pointersToNeighbors)[i].sizeOfDataArr;cout.flush();

			double tableconstuctend=omp_get_wtime();	
			
			printf("\nTable construct time: %f", tableconstuctend - tableconstuctstart);

			
			//add the batched result set size to the total count
			totalResultsLoop+=cnt[tid];

			printf("\nRunning total of total size of result array, tid: %d: %u", tid, totalResultsLoop);
			//}



		

		} //END LOOP OVER THE GPU BATCHES


	
	
	printf("\nTOTAL RESULT SET SIZE ON HOST:  %u", totalResultsLoop);
	*totalNeighbors=totalResultsLoop;


	double tKernelResultsEnd=omp_get_wtime();
	
	printf("\nTime to launch kernel and execute all of the previous part of the method and get the results back: %f",tKernelResultsEnd-tKernelResultsStart);








	//SORTING FOR TESTING ONLY
	//XXXXXXXXX
	//XXXXXXXXX
	// std::sort(results, results+(*cnt),compResults);
	// printf("\n**** GPU\n");
	// for (int i=0; i<(*cnt); i++)
	// {
	// 	printf("\n%d,%d",results[i].pointID, results[i].pointInDist);
	// }

	//XXXXXXXXX
	//XXXXXXXXX
	//XXXXXXXXX


	//get debug information (optional)
	// unsigned int * debug1;
	// debug1=(unsigned int*)malloc(sizeof(unsigned int));
	// *debug1=0;
	// unsigned int * debug2;
	// debug2=(unsigned int*)malloc(sizeof(unsigned int));
	// *debug2=0;

	
	double tStartdebug=omp_get_wtime();

	errCode=cudaMemcpy(debug1, dev_debug1, sizeof(unsigned int), cudaMemcpyDeviceToHost );
	
	if(errCode != cudaSuccess) {
	cout << "\nError: getting debug1 from GPU Got error with code " << errCode << endl; 
	}
	else
	{
		printf("\nDebug1 value: %u",*debug1);
	}

	errCode=cudaMemcpy(debug2, dev_debug2, sizeof(unsigned int), cudaMemcpyDeviceToHost );
	
	if(errCode != cudaSuccess) {
	cout << "\nError: getting debug1 from GPU Got error with code " << errCode << endl; 
	}
	else
	{
		printf("\nDebug2 value: %u",*debug2);
	}	

	double tEnddebug=omp_get_wtime();
	printf("\nTime to retrieve debug values: %f", tEnddebug - tStartdebug);
	

	///////////////////////////////////
	//END GET RESULT SET
	///////////////////////////////////


	///////////////////////////////////
	//FREE MEMORY FROM THE GPU
	///////////////////////////////////
    //free:
	
	

	


	double tFreeStart=omp_get_wtime();



	//destroy streams
	// for (int i=0; i<GPUSTREAMS; i++)
	// {
	// cudaStreamDestroy(stream[i]);
	// }

	for (int i=0; i<GPUSTREAMS; i++)
	{
		errCode=cudaStreamDestroy(stream[i]);

		if(errCode != cudaSuccess) {
		cout << "\nError: destroying stream" << errCode << endl; 
		}
	}


	







	//free the data on the device
	cudaFree(dev_pointIDKey);
	cudaFree(dev_pointInDistValue);

    
	cudaFree(dev_database);
	cudaFree(dev_debug1);
	cudaFree(dev_debug2);
	
	cudaFree(dev_epsilon);
	cudaFree(dev_grid);
	cudaFree(dev_lookupArr);
	cudaFree(dev_gridNumXCells);
	cudaFree(dev_gridNumYCells);
	cudaFree(dev_gridMin_x);
	cudaFree(dev_gridMin_y);
	
	cudaFree(dev_N); 	
	cudaFree(dev_cnt); 
	cudaFree(dev_offset); 
	cudaFree(dev_batchNumber); 

	
		//free data related to the individual streams for each batch
		for (int i=0; i<GPUSTREAMS; i++)
		{
			//free the data on the device
			cudaFree(dev_pointIDKey[i]);
			cudaFree(dev_pointInDistValue[i]);

			//free on the host
			cudaFreeHost(pointIDKey[i]);
			cudaFreeHost(pointInDistValue[i]);

		

		}

	

	cudaFree(dev_pointIDKey);
	cudaFree(dev_pointInDistValue);
	//free pinned memory on host
	cudaFreeHost(pointIDKey);
	cudaFreeHost(pointInDistValue);

	
	
	

	 

	


	double tFreeEnd=omp_get_wtime();

	printf("\nTime freeing memory: %f", tFreeEnd - tFreeStart);

	

		
	
	
	//printf("\ntotal neighbors (in batched fn): %d", *totalNeighbors);

	cout<<"\n** last error at end of fn batches: "<<cudaGetLastError();
	// printf("\nExiting function early..");
	// return;


}







































//void constructNeighborTableKeyValueAlternateTest(int * pointIDKey, int * pointInDistValue, struct neighborTableLookup * neighborTable, int * pointersToNeighbors, unsigned int * cnt);
void constructNeighborTableKeyValueAlternateTest(int * pointIDKey, int * pointInDistValue, struct neighborTableLookup * neighborTable, int * pointersToNeighbors, unsigned int * cnt)
{

	



	//need to take as input a pointer to an array of integers that has not been allocated yet (neighborTableData), 1 per batch (2D array)

	// int * ptrToData;
	// pointersToNeighbors.push_back()
		
	//allocate memory to the array that holds all of the direct neighbors:	 
	
	//pointersToNeighbors= new int[*cnt];
	

	//record the size of the array
//	pointersToNeighbors->sizeOfDataArr=*cnt;


	//copy the value data:
	std::copy(pointInDistValue, pointInDistValue+(*cnt), pointersToNeighbors);


	// printf("\nTest copy: ");
	// int sample=ceil((*cnt)*0.001);
	// for (int i=0; i<sample; i++)
	// {
	// 	printf("\nval: %d",pointersToNeighbors[i]);
	// }


	//Step 1: find all of the unique keys and their positions in the key array
	
	//double tstart=omp_get_wtime();

	unsigned int numUniqueKeys=0;

	struct keyData{
		int key;
		int position;
	};

	std::vector<keyData> uniqueKeyData;

	keyData tmp;
	tmp.key=pointIDKey[0];
	tmp.position=0;
	uniqueKeyData.push_back(tmp);

	//we assign the ith data item when iterating over i+1th data item,
	//so we go 1 loop iteration beyond the number (*cnt)
	for (int i=1; i<(*cnt)+1; i++)
	{
		if (pointIDKey[i-1]!=pointIDKey[i])
		{
			numUniqueKeys++;
			tmp.key=pointIDKey[i];
			tmp.position=i;
			uniqueKeyData.push_back(tmp);
		}
	}


	
	//insert into the neighbor table the values based on the positions of 
	//the unique keys obtained above. 
	for (int i=0; i<uniqueKeyData.size()-1; i++) 
	{
		int keyElem=uniqueKeyData[i].key;
		neighborTable[keyElem].pointID=keyElem;
		neighborTable[keyElem].indexmin=uniqueKeyData[i].position;
		neighborTable[keyElem].indexmax=uniqueKeyData[i+1].position-1;
	
		//update the pointer to the data array for the values
		neighborTable[keyElem].dataPtr=pointersToNeighbors;	

	}








	/*	
	//newer multithreaded way:
	//Step 1: find all of the unique keys and their positions in the key array
	
	//double tstart=omp_get_wtime();

	unsigned int numUniqueKeys=0;
	unsigned int count=0;

	struct keyData{
		int key;
		int position;
	};

	std::vector<keyData> uniqueKeyData;

	keyData tmp;
	tmp.key=pointIDKey[0];
	tmp.position=0;
	uniqueKeyData.push_back(tmp);

	//we assign the ith data item when iterating over i+1th data item,
	//so we go 1 loop iteration beyond the number (*cnt)
	for (int i=1; i<(*cnt)+1; i++)
	{
		if (pointIDKey[i-1]!=pointIDKey[i])
		{
			numUniqueKeys++;
			tmp.key=pointIDKey[i];
			tmp.position=i;
			uniqueKeyData.push_back(tmp);
		}
	}

	//Step 2: In parallel, insert into the neighbor table the values based on the positions of 
	//the unique keys obtained above. Since multiple threads access this function, we don't want to oversubscribe the
	//machine with nested parallelism, so limit to 2 threads
	omp_set_nested(1);
	#pragma omp parallel for reduction(+:count) num_threads(2) schedule(static,1)
	for (int i=0; i<uniqueKeyData.size()-1; i++) 
	{
		int keyElem=uniqueKeyData[i].key;
		int valStart=uniqueKeyData[i].position;
		int valEnd=uniqueKeyData[i+1].position-1;
		int size=valEnd-valStart+1;
		//printf("\nval: start:%d, end: %d", valStart,valEnd);
		neighborTable[keyElem].pointID=keyElem;
		neighborTable[keyElem].neighbors.insert(neighborTable[keyElem].neighbors.begin(),&pointInDistValue[valStart],&pointInDistValue[valStart+size]);
		//printf("\ni: %d, keyElem: %d, position start: %d, position end: %d, size: %d", i,keyElem,valStart, valEnd,size);	


		count+=size;

	}
	
	*/

}

















void constructNeighborTableKeyValue(int * pointIDKey, int * pointInDistValue, struct table * neighborTable, unsigned int * cnt)
{
	
	//newer multithreaded way:
	//Step 1: find all of the unique keys and their positions in the key array
	
	//double tstart=omp_get_wtime();

	unsigned int numUniqueKeys=0;
	unsigned int count=0;

	struct keyData{
		int key;
		int position;
	};

	std::vector<keyData> uniqueKeyData;

	keyData tmp;
	tmp.key=pointIDKey[0];
	tmp.position=0;
	uniqueKeyData.push_back(tmp);

	//we assign the ith data item when iterating over i+1th data item,
	//so we go 1 loop iteration beyond the number (*cnt)
	for (int i=1; i<(*cnt)+1; i++)
	{
		if (pointIDKey[i-1]!=pointIDKey[i])
		{
			numUniqueKeys++;
			tmp.key=pointIDKey[i];
			tmp.position=i;
			uniqueKeyData.push_back(tmp);
		}
	}

	//Step 2: In parallel, insert into the neighbor table the values based on the positions of 
	//the unique keys obtained above. Since multiple threads access this function, we don't want to oversubscribe the
	//machine with nested parallelism, so limit to 2 threads
	omp_set_nested(1);
	#pragma omp parallel for reduction(+:count) num_threads(2) schedule(static,1)
	for (int i=0; i<uniqueKeyData.size()-1; i++) 
	{
		int keyElem=uniqueKeyData[i].key;
		int valStart=uniqueKeyData[i].position;
		int valEnd=uniqueKeyData[i+1].position-1;
		int size=valEnd-valStart+1;
		//printf("\nval: start:%d, end: %d", valStart,valEnd);
		neighborTable[keyElem].pointID=keyElem;
		neighborTable[keyElem].neighbors.insert(neighborTable[keyElem].neighbors.begin(),&pointInDistValue[valStart],&pointInDistValue[valStart+size]);
		//printf("\ni: %d, keyElem: %d, position start: %d, position end: %d, size: %d", i,keyElem,valStart, valEnd,size);	


		count+=size;

	}
	

}









void constructNeighborTable(thrust::host_vector<structresults> * hVectResults, struct table * neighborTable, unsigned int * cnt)
{
	//original way:
	
	// for (unsigned int i=0; i<(*cnt); i++)
	// {
	// 	unsigned int elemID=hVectResults[i].pointID;
	// 	neighborTable[elemID].pointID=elemID;
	// 	neighborTable[elemID].neighbors.push_back(hVectResults[i].pointInDist);
	// }
	
	//end original way


	//new way: loop over and find the ranges of the different point ids
	//then make one insert into the vector
	

	unsigned int lastElemID=(*hVectResults)[0].pointID;
	unsigned int lastIndex=0;


	//we assign the ith data item when iterating over i+1th data item,
	//so we go 1 loop iteration beyond the number (*cnt)
	for (unsigned int i=1; i<(*cnt)+1; i++)
	{
		
	

		if ((*hVectResults)[i].pointID!=lastElemID)
		{
			unsigned int rangemax=i-1;
			
			int tmpSize=rangemax-lastIndex+1;
			unsigned int tmp[tmpSize];

			for (int j=lastIndex; j<=rangemax; j++)
			{
				tmp[j-lastIndex]=(*hVectResults)[j].pointInDist;
			}	

			neighborTable[lastElemID].pointID=lastElemID;	
			neighborTable[lastElemID].neighbors.insert(neighborTable[lastElemID].neighbors.begin(),tmp, tmp+tmpSize);
			
			//update the new last elem id
			lastElemID=(*hVectResults)[i].pointID;
			lastIndex=i;
		}
	}

	//print table:
	/*
	int tmpcnt=0;
	printf("\nGrid GPU Table**********");
	for (int i=0; i<(*N); i++)
	{
		printf("\nPoint id: %d In distance: ", neighborTable[i].pointID);
		for (int j=0; j<neighborTable[i].neighbors.size();j++)
		{
			printf("%d, ",neighborTable[i].neighbors[j]);
			tmpcnt++;
		}
	}

	printf("\n count elems: %d", tmpcnt);
	*/


}




























//Uses a brute force kernel to calculate the direct neighbors of the points in the database
void makeDistanceTableGPUBruteForce(std::vector<struct dataElem> * dataPoints, double * epsilon, struct table * neighborTable, int * totalNeighbors)
{
	//CUDA error code:
	cudaError_t errCode;


	///////////////////////////////////
	//COPY THE DATABASE TO THE GPU
	///////////////////////////////////


	unsigned int * N;
	N=(unsigned int*)malloc(sizeof(unsigned int));
	*N=dataPoints->size();
	
	printf("\n in main GPU method: N is: %u",*N);cout.flush();

	struct point * database;
	database=(struct point*)malloc(sizeof(struct point)*(*N));
	


	struct point * dev_database;
	dev_database=(struct point*)malloc(sizeof(struct point)*(*N));

	//allocate memory on device:
	
	errCode=cudaMalloc( (void**)&dev_database, sizeof(struct point)*(*N));

	printf("\n !!in main GPU method: N is: %u",*N);cout.flush();	
		
	if(errCode != cudaSuccess) {
	cout << "\nError: database Got error with code " << errCode << endl; cout.flush(); 
	}


	



	//first, we copy the x and y values from dataPoints to the database
	for (int i=0; i<(*N); i++)
	{
		database[i].x=(*dataPoints)[i].x;
		database[i].y=(*dataPoints)[i].y;
	}



	//printf("\n size of database: %d",N);

	//copy database to the device:
	errCode=cudaMemcpy(dev_database, database, sizeof(struct point)*(*N), cudaMemcpyHostToDevice);	
	if(errCode != cudaSuccess) {
	cout << "\nError: database Got error with code " << errCode << endl; 
	}

	///////////////////////////////////
	//END COPY THE DATABASE TO THE GPU
	///////////////////////////////////





	///////////////////////////////////
	//ALLOCATE MEMORY FOR THE RESULT SET
	///////////////////////////////////
	//NON-PINNED MEMORY FOR SINGLE KERNEL INVOCATION (NO BATCHING)


	//CHANGING THE RESULTS TO KEY VALUE PAIR SORT, WHICH IS TWO ARRAYS
	//KEY IS THE POINT ID
	//THE VALUE IS THE POINT ID WITHIN THE DISTANCE OF KEY

	int * dev_pointIDKey; //key
	int * dev_pointInDistValue; //value

	int * pointIDKey; //key
	int * pointInDistValue; //value


	errCode=cudaMalloc((void **)&dev_pointIDKey, sizeof(int)*BUFFERELEM);
	if(errCode != cudaSuccess) {
	cout << "CUDA: Got error with code " << errCode << endl; //2 means not enough memory
	}

	errCode=cudaMalloc((void **)&dev_pointInDistValue, sizeof(int)*BUFFERELEM);
	if(errCode != cudaSuccess) {
	cout << "CUDA: Got error with code " << errCode << endl; //2 means not enough memory
	}


	printf("\nmemory requested for results (GiB): %f",(double)(sizeof(int)*2*BUFFERELEM)/(1024*1024*1024));

	double tstartalloc=omp_get_wtime();

	//host result allocation:
	//pinned result set memory for the host
	// cudaMallocHost((void **) &pointIDKey, sizeof(int)*BUFFERELEM);
	// cudaMallocHost((void **) &pointInDistValue, sizeof(int)*BUFFERELEM);



	//PAGED MEMORY ALLOCATION FOR SMALL RESULT SET WITH SINGLE KERNEL EXECUTION?
	pointIDKey=(int*)malloc(sizeof(int)*BUFFERELEM);
	pointInDistValue=(int*)malloc(sizeof(int)*BUFFERELEM);

	double tendalloc=omp_get_wtime();


	//printf("\nTime to allocate pinned memory on the host: %f", tendalloc - tstartalloc);
	printf("\nTime to allocate (non-pinned) memory on the host: %f", tendalloc - tstartalloc);

	

	///////////////////////////////////
	//END ALLOCATE MEMORY FOR THE RESULT SET
	///////////////////////////////////















	///////////////////////////////////
	//SET OTHER KERNEL PARAMETERS
	///////////////////////////////////

	

	//count values
	unsigned int * cnt;
	cnt=(unsigned int*)malloc(sizeof(unsigned int));
	*cnt=0;

	unsigned int * dev_cnt; 
	dev_cnt=(unsigned int*)malloc(sizeof(unsigned int));
	*dev_cnt=0;

	//allocate on the device
	errCode=cudaMalloc((unsigned int**)&dev_cnt, sizeof(unsigned int));	
	if(errCode != cudaSuccess) {
	cout << "\nError: cnt Got error with code " << errCode << endl; 
	}


	errCode=cudaMemcpy( dev_cnt, cnt, sizeof(unsigned int), cudaMemcpyHostToDevice );
	if(errCode != cudaSuccess) {
	cout << "\nError: dev_cnt memcpy Got error with code " << errCode << endl; 
	}

	
	//Epsilon
	double * dev_epsilon;
	dev_epsilon=(double*)malloc(sizeof( double ));
	//*dev_epsilon=*epsilon;

	//Allocate on the device
	errCode=cudaMalloc((void**)&dev_epsilon, sizeof(double));
	if(errCode != cudaSuccess) {
	cout << "\nError: epsilon Got error with code " << errCode << endl; 
	}


	
	//size of the database:
	unsigned int * dev_N; 
	dev_N=(unsigned int*)malloc(sizeof( unsigned int ));

	//allocate on the device
	errCode=cudaMalloc((void**)&dev_N, sizeof(unsigned int));
	if(errCode != cudaSuccess) {
	cout << "\nError: N Got error with code " << errCode << endl; 
	}	

	//debug values
	unsigned int * dev_debug1; 
	unsigned int * dev_debug2; 
	dev_debug1=(unsigned int *)malloc(sizeof(unsigned int ));
	*dev_debug1=0;
	dev_debug2=(unsigned int *)malloc(sizeof(unsigned int ));
	*dev_debug2=0;




	//allocate on the device
	errCode=cudaMalloc( (unsigned int **)&dev_debug1, sizeof(unsigned int ) );
	if(errCode != cudaSuccess) {
	cout << "\nError: debug1 Got error with code " << errCode << endl; 
	}		
	errCode=cudaMalloc( (unsigned int **)&dev_debug2, sizeof(unsigned int ) );
	if(errCode != cudaSuccess) {
	cout << "\nError: debug2 Got error with code " << errCode << endl; 
	}		


	//copy N, epsilon and cnt to the device
	//epsilon
	errCode=cudaMemcpy( dev_epsilon, epsilon, sizeof(double), cudaMemcpyHostToDevice );
	if(errCode != cudaSuccess) {
	cout << "\nError: epsilon Got error with code " << errCode << endl; 
	}		



	

	//N (DATASET SIZE)
	errCode=cudaMemcpy( dev_N, N, sizeof(unsigned int), cudaMemcpyHostToDevice );
	if(errCode != cudaSuccess) {
	cout << "\nError: N Got error with code " << errCode << endl; 
	}		




	///////////////////////////////////
	//END SET OTHER KERNEL PARAMETERS
	///////////////////////////////////


	


	///////////////////////////////////
	//LAUNCH KERNEL
	///////////////////////////////////

	const int TOTALBLOCKS=ceil((1.0*(*N))/(1.0*BLOCKSIZE));	
	printf("\ntotal blocks: %d",TOTALBLOCKS);

	//execute kernel	
	
	kernelBruteForce<<< TOTALBLOCKS, BLOCKSIZE >>>(dev_N, dev_debug1, dev_debug2, dev_epsilon, dev_cnt, dev_database, dev_pointIDKey, dev_pointInDistValue);
	if ( cudaSuccess != cudaGetLastError() ){
    	printf( "Error in kernel launch!\n" );
    }

    ///////////////////////////////////
	//END LAUNCH KERNEL
	///////////////////////////////////



    ///////////////////////////////////
	//GET RESULT SET
	///////////////////////////////////

	//first find the size of the number of results
	errCode=cudaMemcpy( cnt, dev_cnt, sizeof(unsigned int), cudaMemcpyDeviceToHost );
	if(errCode != cudaSuccess) {
	cout << "\nError: getting cnt from GPU Got error with code " << errCode << endl; 
	}
	else
	{
		printf("\nGPU: result set size on within epsilon: %d",*cnt);
	}

	//copy the results, but only transfer the number of results, not the entire buffer
	// errCode=cudaMemcpy(results, dev_results, sizeof(struct structresults)*(*cnt), cudaMemcpyDeviceToHost );
	
	// if(errCode != cudaSuccess) {
	// cout << "\nError: getting results from GPU Got error with code " << errCode << endl; 
	// }

	*totalNeighbors=(*cnt);

	//SORTING FOR TESTING ONLY
	//XXXXXXXXX
	//XXXXXXXXX
	// std::sort(results, results+(*cnt),compResults);
	// printf("\n**** GPU\n");
	// for (int i=0; i<(*cnt); i++)
	// {
	// 	printf("\n%d,%d",results[i].pointID, results[i].pointInDist);
	// }

	//XXXXXXXXX
	//XXXXXXXXX
	//XXXXXXXXX


	//get debug information (optional)
	unsigned int * debug1;
	debug1=(unsigned int*)malloc(sizeof(unsigned int));
	*debug1=0;
	unsigned int * debug2;
	debug2=(unsigned int*)malloc(sizeof(unsigned int));
	*debug2=0;

	errCode=cudaMemcpy(debug1, dev_debug1, sizeof(unsigned int), cudaMemcpyDeviceToHost );
	
	if(errCode != cudaSuccess) {
	cout << "\nError: getting debug1 from GPU Got error with code " << errCode << endl; 
	}
	else
	{
		printf("\nDebug1 value: %u",*debug1);
	}

	errCode=cudaMemcpy(debug2, dev_debug2, sizeof(unsigned int), cudaMemcpyDeviceToHost );
	
	if(errCode != cudaSuccess) {
	cout << "\nError: getting debug1 from GPU Got error with code " << errCode << endl; 
	}
	else
	{
		printf("\nDebug2 value: %u",*debug2);
	}	


	///////////////////////////////////
	//END GET RESULT SET
	///////////////////////////////////


	///////////////////////////////////
	//FREE MEMORY FROM THE GPU
	///////////////////////////////////
    //free:
	cudaFree(dev_database);
	cudaFree(dev_debug1);
	cudaFree(dev_debug2);
	cudaFree(dev_cnt);
	cudaFree(dev_epsilon);
	//cudaFree(dev_results);

	////////////////////////////////////


	////////////////////////////////////
	//SORT THE TABLE DATA ON THE GPU
	//THERE IS NO ORDERING BETWEEN EACH POINT AND THE ONES THAT IT'S WITHIN THE DISTANCE OF
	////////////////////////////////////

	/////////////////////////////
	//ONE PROBLEM WITH NOT TRANSFERING THE RESULT OFF OF THE DEVICE IS THAT
	//YOU CAN'T RESIZE THE RESULTS TO BE THE SIZE OF *CNT
	//SO THEN YOU HAVE POTENTIALLY LOTS OF WASTED SPACE
	/////////////////////////////

	//sort by key with the data already on the device:
	//wrap raw pointer with a device_ptr to use with Thrust functions
	thrust::device_ptr<int> dev_keys_ptr(dev_pointIDKey);
	thrust::device_ptr<int> dev_data_ptr(dev_pointInDistValue);

	


	try{
	thrust::sort_by_key(dev_keys_ptr, dev_keys_ptr + (*cnt), dev_data_ptr);
	}
	catch(std::bad_alloc &e)
	  {
	    std::cerr << "Ran out of memory while sorting" << std::endl;
	    exit(-1);
	  }
	
	


	//copy the sorted arays back to the host
	thrust::copy(dev_keys_ptr,dev_keys_ptr+(*cnt),pointIDKey);
	thrust::copy(dev_data_ptr,dev_data_ptr+(*cnt),pointInDistValue);
	  
	

	//free the data on the device
	cudaFree(dev_pointIDKey);
	cudaFree(dev_pointInDistValue);

	
	////////////////////////////////////
	//END SORT THE DATA ON THE GPU
	////////////////////////////////////


	////////////////////////////////////
	//CONSTRUCT TABLE
	////////////////////////////////////	

	double tStartTableConstruct=omp_get_wtime();
	constructNeighborTableKeyValue(pointIDKey, pointInDistValue, neighborTable, cnt);
	double tEndTableConstruct=omp_get_wtime();	
	printf("\nTime constructing table: %f",tEndTableConstruct - tStartTableConstruct);	

	
		//print table:

	// for (int i=0; i<(*N); i++)
	// {
	// 	printf("\nPoint id: %d In distance: ", neighborTable[i].pointID);
	// 	for (int j=0; j<neighborTable[i].neighbors.size();j++)
	// 	{
	// 		printf("%d, ",neighborTable[i].neighbors[j]);
	// 	}
	// }




	////////////////////////////////////
	//END CONSTRUCT TABLE
	////////////////////////////////////	







}










//generates an array and lookup table for the GPU, from individual pointers to the neighbors from a previous table's results
//This one uses the new implementation that doesn't use vectors
//Input:
//numPoints
//inputNeighborTable
//OLD: dataPtr - vector of pointers to the arrays containing the neighbors across the previous batches in the neighborTableLookup //std::vector<struct neighborDataPtrs> *dataPtr,

//Outputs:
//directNeighborArray -- the ids of the points within epsilon of the input table
//gpuLookupArray -- points to the indices of the elements in directNeighborArray
void generateNeighborArrayForGPUAlternative(unsigned int numPoints, struct neighborTableLookup * inputNeighborTable, int * directNeighborArray, struct gpulookuptable * gpuLookupArray)
{

	
	//populate the direct neighboer array
	//and the lookup array at the same time
	//This is because the neighbors for each data point are stored across the various arrays allocated for each batch

	int startIndex=0;

	for (int i=0; i<numPoints; i++)
	{
		int indexmin=inputNeighborTable[i].indexmin;
		int indexmax=inputNeighborTable[i].indexmax;
		int * data=  (inputNeighborTable[i].dataPtr)+indexmin;
		int sizeRange=indexmax-indexmin+1;

		//printf("\nIteration: %d, Start index: %d, sizeRange: %d",i,startIndex,sizeRange);

		 //std::copy(data+indexmin, data+sizeRange, directNeighborArray+startIndex);
		std::copy(data, data+sizeRange, directNeighborArray+startIndex);




		gpuLookupArray[i].indexmin=startIndex;
		gpuLookupArray[i].indexmax=startIndex+sizeRange-1;

		

		startIndex+=sizeRange;

		

	}




}
























//generates an array and lookup table for the GPU. This is because we can't use vectors on the GPU.
void generateNeighborArrayForGPU(unsigned int numPoints, struct table * inputNeighborTable, int * directNeighborArray, struct gpulookuptable * gpuLookupArray)
{
	int startIndex=0;
	unsigned int cnt=0;

	for (int i=0; i<numPoints; i++)
	{
		startIndex=cnt;
		for (int j=0; j<inputNeighborTable[i].neighbors.size(); j++)
		{
			directNeighborArray[cnt]=inputNeighborTable[i].neighbors[j];
			cnt++;
		}	


		gpuLookupArray[i].indexmin=startIndex;
		gpuLookupArray[i].indexmax=cnt-1;

	}

}



void generateDistanceTableFromPreviousTable(std::vector<struct dataElem> * dataPoints, struct gpulookuptable * gpuLookupArray, int * directNeighborArray, int * totalDirectNeighbors, double * epsilon,  struct table * neighborTable)
{
	printf("\nIn generate from previous table:\nDatapoints: %lu, \nTotal direct neighbors: %d\n",dataPoints->size(), *totalDirectNeighbors);

	cout<<"\n** Last CUDA error start of fn: "<<cudaGetLastError();

	//CUDA error code:
	cudaError_t errCode;

	unsigned int * N;
	N=(unsigned int*)malloc(sizeof(unsigned int));
	*N=dataPoints->size();
	
	printf("\n in generate previous table GPU method: N is: %u",*N);cout.flush();


	///////////////////////////////////
	//COPY THE DATABASE TO THE GPU
	///////////////////////////////////

	struct point * database;
	database=(struct point*)malloc(sizeof(struct point)*(*N));
	


	struct point * dev_database;
	dev_database=(struct point*)malloc(sizeof(struct point)*(*N));

	//allocate memory on device:
	
	errCode=cudaMalloc( (void**)&dev_database, sizeof(struct point)*(*N));	
	if(errCode != cudaSuccess) {
	cout << "\nError: database (in previous table method) Got error with code " << errCode << endl; cout.flush(); 
	}

	//first, we copy the x and y values from dataPoints to the database
	for (int i=0; i<(*N); i++)
	{
		database[i].x=(*dataPoints)[i].x;
		database[i].y=(*dataPoints)[i].y;
	}

	//copy database to the device:
	errCode=cudaMemcpy(dev_database, database, sizeof(struct point)*(*N), cudaMemcpyHostToDevice);	
	if(errCode != cudaSuccess) {
	cout << "\nError: database memcopy (in previous table method) Got error with code " << errCode << endl; 
	}

	///////////////////////////////////
	//END COPY THE DATABASE TO THE GPU
	///////////////////////////////////


	///////////////////////////////
	//Copy the lookup array struct to the GPU:
	///////////////////////////////

	struct gpulookuptable * dev_gpuLookupArray;
	dev_gpuLookupArray=(struct gpulookuptable*)malloc(sizeof(struct gpulookuptable)*(*N));

	//allocate memory on device:
	errCode=cudaMalloc( (void**)&dev_gpuLookupArray, sizeof(struct gpulookuptable)*(*N));
		
	if(errCode != cudaSuccess) {
	cout << "\nError: gpu lookup array Got error with code " << errCode << endl; cout.flush(); 
	}

		


	//copy lookup array to the device:
	errCode=cudaMemcpy(dev_gpuLookupArray, gpuLookupArray, sizeof(struct gpulookuptable)*(*N), cudaMemcpyHostToDevice);	
	if(errCode != cudaSuccess) {
	cout << "\nError: lookup array memcpy Got error with code " << errCode << endl; 
	}


	///////////////////////////////
	//END Copy the lookup array struct to the GPU:
	///////////////////////////////
	

	///////////////////////////////
	//Copy direct neighbor array to the GPU:
	///////////////////////////////

	int * dev_directNeighborArray;
	dev_directNeighborArray=(int*)malloc(sizeof(int)*(*totalDirectNeighbors));

	//allocate memory on device for the direct neighbor array:
	errCode=cudaMalloc( (void**)&dev_directNeighborArray, sizeof(int)*(*totalDirectNeighbors));
		
	if(errCode != cudaSuccess) {
	cout << "\nError: gpu direct neighbor array Got error with code " << errCode << endl; cout.flush(); 
	}

	//copy direct neighbor array to the device:
	errCode=cudaMemcpy(dev_directNeighborArray, directNeighborArray, sizeof(int)*(*totalDirectNeighbors), cudaMemcpyHostToDevice);	
	if(errCode != cudaSuccess) {
	cout << "\nError: direct neighbor array memcpy Got error with code " << errCode << endl; 
	}


	///////////////////////////////
	//END Copy direct neighbor array to the GPU:
	///////////////////////////////


	///////////////////////////////
	//copy the size of the database
	///////////////////////////////
	unsigned int * dev_N; 
	dev_N=(unsigned int*)malloc(sizeof( unsigned int ));

	//allocate on the device
	errCode=cudaMalloc((void**)&dev_N, sizeof(unsigned int));
	if(errCode != cudaSuccess) {
	cout << "\nError: N Got error with code " << errCode << endl; 
	}



	errCode=cudaMemcpy( dev_N, N, sizeof(unsigned int), cudaMemcpyHostToDevice );
	if(errCode != cudaSuccess) {
	cout << "\nError: N Got error with code " << errCode << endl; 
	}	

	///////////////////////////////
	//END copy the size of the database
	///////////////////////////////



	///////////////////////////////
	//copy the newer (smaller) epsilon
	///////////////////////////////

	double * dev_epsilon;
	dev_epsilon=(double*)malloc(sizeof( double ));
	
	

	//Allocate on the device
	errCode=cudaMalloc((void**)&dev_epsilon, sizeof(double));
	if(errCode != cudaSuccess) {
	cout << "\nError: epsilon Got error with code " << errCode << endl; 
	}

	

	errCode=cudaMemcpy( dev_epsilon, epsilon, sizeof(double), cudaMemcpyHostToDevice );
	if(errCode != cudaSuccess) {
	cout << "\nError: epsilon memcpy Got error with code " << errCode << endl; 
	}


	///////////////////////////////
	//END copy the newer (smaller) epsilon
	///////////////////////////////


	///////////////////////////////////
	//ALLOCATE COUNT ON THE DEVICE (THE NUMBER OF RESULT ITEMS)
	///////////////////////////////////


	//count values
	unsigned int * cnt;
	cnt=(unsigned int*)malloc(sizeof(unsigned int));
	*cnt=0;

	unsigned int * dev_cnt; 
	dev_cnt=(unsigned int*)malloc(sizeof(unsigned int));
	*dev_cnt=0;

	

	//allocate on the device
	errCode=cudaMalloc((unsigned int**)&dev_cnt, sizeof(unsigned int));	
	if(errCode != cudaSuccess) {
	cout << "\nError: cnt Got error with code " << errCode << endl; 
	}

	

	errCode=cudaMemcpy( dev_cnt, cnt, sizeof(unsigned int), cudaMemcpyHostToDevice );
	if(errCode != cudaSuccess) {
	cout << "\nError: dev_cnt memcpy Got error with code " << errCode << endl; 
	}



	///////////////////////////////////
	//END ALLOCATE COUNT ON THE DEVICE (THE NUMBER OF RESULT ITEMS)
	///////////////////////////////////


	///////////////////////////////////
	//ALLOCATE MEMORY FOR THE RESULT SET
	///////////////////////////////////
	// struct structresults * dev_results;
	// struct structresults * results;

	// errCode=cudaMalloc((void **)&dev_results, sizeof(struct structresults)*BUFFERELEM);
	// if(errCode != cudaSuccess) {
	// cout << "CUDA: Got error with code " << errCode << endl; //2 means not enough memory
	// }

	// printf("\nmemory requested for results from previous table (GiB): %f",(double)(sizeof(struct structresults)*BUFFERELEM)/(1024*1024*1024));

	// //host result allocation:
	// results=(struct structresults*)malloc(sizeof(struct structresults)*BUFFERELEM);


	int * dev_pointIDKey; //key
	int * dev_pointInDistValue; //value

	int * pointIDKey; //key
	int * pointInDistValue; //value


	

	errCode=cudaMalloc((void **)&dev_pointIDKey, sizeof(int)*BUFFERELEM);
	if(errCode != cudaSuccess) {
	cout << "CUDA: Got error with code " << errCode << endl; //2 means not enough memory
	}

	

	errCode=cudaMalloc((void **)&dev_pointInDistValue, sizeof(int)*BUFFERELEM);
	if(errCode != cudaSuccess) {
	cout << "CUDA: Got error with code " << errCode << endl; //2 means not enough memory
	}


	printf("\nmemory requested for results (GiB): %f",(double)(sizeof(int)*2*BUFFERELEM)/(1024*1024*1024));

	double tstartalloc=omp_get_wtime();

	//host result allocation:
	//pinned result set memory for the host
	// cudaMallocHost((void **) &pointIDKey, sizeof(int)*BUFFERELEM);
	// cudaMallocHost((void **) &pointInDistValue, sizeof(int)*BUFFERELEM);



	//PAGED MEMORY ALLOCATION FOR SMALL RESULT SET WITH SINGLE KERNEL EXECUTION?
	pointIDKey=(int*)malloc(sizeof(int)*BUFFERELEM);
	pointInDistValue=(int*)malloc(sizeof(int)*BUFFERELEM);

	double tendalloc=omp_get_wtime();


	//printf("\nTime to allocate pinned memory on the host: %f", tendalloc - tstartalloc);
	printf("\nTime to allocate (non-pinned) memory on the host: %f", tendalloc - tstartalloc);

	
		
	
	

	///////////////////////////////////
	//END ALLOCATE MEMORY FOR THE RESULT SET
	///////////////////////////////////



	///////////////////////////////
	//EXECUTE KERNEL
	///////////////////////////////

	const int TOTALBLOCKS=ceil((1.0*(*N))/(1.0*BLOCKSIZE));	
	printf("\ntotal blocks (from previous table method): %d",TOTALBLOCKS);

	//execute kernel	
	calcNeighborsFromTableKernel<<< TOTALBLOCKS, BLOCKSIZE >>>(dev_N, dev_gpuLookupArray, dev_directNeighborArray, dev_cnt, dev_epsilon, dev_database, dev_pointIDKey, dev_pointInDistValue);
	cout <<endl<<"After kernel launch, Error code: "<<cudaGetLastError()<<endl;
	if ( cudaSuccess != cudaGetLastError() ){
    	printf( "\nError in kernel launch (previous table method)!" );
    	// cout <<endl<<"Error code: "<<cudaGetLastError()<<endl;
    }

    


	///////////////////////////////
	//END EXECUTE KERNEL
	///////////////////////////////

    ///////////////////////////////////
	//GET RESULT SET
	///////////////////////////////////

	//first find the size of the number of results
	//errCode=cudaMemcpy( cnt, dev_cnt, sizeof(unsigned int), cudaMemcpyDeviceToHost );
	// if(errCode != cudaSuccess) {
	// cout << "\nError: getting cnt from GPU Got error with code " << errCode << endl; 
	// }
	// else
	// {
	// 	printf("\nGPU: result set size on within epsilon: %d",*cnt);
	// }


	errCode=cudaMemcpy( cnt, dev_cnt, sizeof(unsigned int), cudaMemcpyDeviceToHost );
	if(errCode != cudaSuccess) {
	cout << "\nError: getting cnt from GPU Got error with code " << errCode << endl; 
	}
	else
	{
		printf("\nGPU: result set size on GPU within epsilon (from precomputed table method): %d",*cnt);
	}

	

	
	//copy the results, but only transfer the number of results, not the entire buffer
	// errCode=cudaMemcpy(results, dev_results, sizeof(struct structresults)*(*cnt), cudaMemcpyDeviceToHost );
	// if(errCode != cudaSuccess) {
	// cout << "\nError: getting results from GPU (from the precomputed table) Got error with code " << errCode << endl; 
	// }

	//*totalNeighbors=(*cnt);

	//SORTING FOR TESTING ONLY
	//XXXXXXXXX
	//XXXXXXXXX
	// std::sort(results, results+(*cnt),compResults);
	// printf("\n**** GPU\n");
	// for (int i=0; i<(*cnt); i++)
	// {
	// 	printf("\n%d,%d",results[i].pointID, results[i].pointInDist);
	// }

	//XXXXXXXXX
	//XXXXXXXXX
	//XXXXXXXXX



	///////////////////////////////////
	//END GET RESULT SET
	///////////////////////////////////



	/////////////////
	//FREE
	/////////////////
	cudaFree(dev_directNeighborArray);
	cudaFree(dev_gpuLookupArray);
	//cudaFree(dev_results);
	cudaFree(dev_database);
	cudaFree(dev_epsilon);
	cudaFree(dev_N);
	cudaFree(dev_cnt);





	////////////////////////////////////
	//SORT THE TABLE DATA ON THE GPU
	//THERE IS NO ORDERING BETWEEN EACH POINT AND THE ONES THAT IT'S WITHIN THE DISTANCE OF
	////////////////////////////////////

	/////////////////////////////
	//ONE PROBLEM WITH NOT TRANSFERING THE RESULT OFF OF THE DEVICE IS THAT
	//YOU CAN'T RESIZE THE RESULTS TO BE THE SIZE OF *CNT
	//SO THEN YOU HAVE POTENTIALLY LOTS OF WASTED SPACE
	/////////////////////////////

	//sort by key with the data already on the device:
	//wrap raw pointer with a device_ptr to use with Thrust functions
	thrust::device_ptr<int> dev_keys_ptr(dev_pointIDKey);
	thrust::device_ptr<int> dev_data_ptr(dev_pointInDistValue);


	// allocate space for the output
	//thrust::device_vector<int> sortedKeys(*cnt);
	//thrust::device_vector<int> sortedVals(*cnt);
	


	try{
	thrust::sort_by_key(dev_keys_ptr, dev_keys_ptr + (*cnt), dev_data_ptr);
	}
	catch(std::bad_alloc &e)
	  {
	    std::cerr << "Ran out of memory while sorting" << std::endl;
	    exit(-1);
	  }
	
	
	//copy the sorted arays back to the host
	thrust::copy(dev_keys_ptr,dev_keys_ptr+(*cnt),pointIDKey);
	thrust::copy(dev_data_ptr,dev_data_ptr+(*cnt),pointInDistValue);
	  


	//free the data on the device
	cudaFree(dev_pointIDKey);
	cudaFree(dev_pointInDistValue);



	////////////////////////////////////
	//CONSTRUCT TABLE
	////////////////////////////////////

	double tStartTableConstruct=omp_get_wtime();
	constructNeighborTableKeyValue(pointIDKey, pointInDistValue, neighborTable, cnt);
	double tEndTableConstruct=omp_get_wtime();	
	printf("\nTime constructing table: %f",tEndTableConstruct - tStartTableConstruct);	

	////////////////////////////////////
	//END CONSTRUCT TABLE
	////////////////////////////////////




	/*
	////////////////////////////////////
	//SORT THE TABLE DATA ON THE GPU
	////////////////////////////////////

	double tstartsort=omp_get_wtime();
	//make a host vector initialized with the results that have been transfered from the GPU
	thrust::host_vector<structresults> hVectResults(results,results+(*cnt));



	// for (int i=0;i<numResults;i++)
	// {
	// 	printf("\n host vector: %d, %d",hVectResults[i].pointID,hVectResults[i].pointInDist);
	// }

	// for (int i=0; i<numResults; i++)
	// {
	// 	structresults tmp;
	// 	tmp.pointID=0;
	// 	tmp.pointInDist=0;
	// 	hVectResults.push_back(tmp);
	// }

	//Now transfer the hostvector to the device:
	thrust::device_vector<structresults> dVectResults=hVectResults;


	//sort the device vector on the GPU

	try{
	thrust::sort(dVectResults.begin(), dVectResults.end(),compareThrust());
	}
	catch(std::bad_alloc &e)
	  {
	    std::cerr << "Ran out of memory while sorting" << std::endl;
	    exit(-1);
	  }

	// transfer the sorted results back to host
	thrust::copy(dVectResults.begin(), dVectResults.end(), hVectResults.begin());

	double tendsort=omp_get_wtime();

	printf("\nTime to sort on the GPU (from precompute table): %f",tendsort-tstartsort);


	//print GPU:
	// for (int i=0; i<(*cnt);i++)
	// {	
	// 	printf("\nPrecompute GPU elem: %d, data: %d",hVectResults[i].pointID,hVectResults[i].pointInDist);
	// }


	////////////////////////////////////
	//END SORT THE DATA ON THE GPU
	////////////////////////////////////


	////////////////////////////////////
	//CONSTRUCT TABLE
	////////////////////////////////////	


	
	for (unsigned int i=0; i<(*cnt); i++)
	{
		unsigned int index=hVectResults[i].pointID;
		neighborTable[index].pointID=index;
		neighborTable[index].neighbors.push_back(hVectResults[i].pointInDist);
	}
	*/

	//print table:
	/*
	printf("\n****Precompute table: ");
	for (int i=0; i<(*N); i++)
	{
		printf("\nPoint id: %d In distance: ", neighborTable[i].pointID);
		for (int j=0; j<neighborTable[i].neighbors.size();j++)
		{
			printf("%d, ",neighborTable[i].neighbors[j]);
		}
	}
	*/
	////////////////////////////////////
	//END CONSTRUCT TABLE
	////////////////////////////////////	

}










//USE THIS TO MAKE A TABLE FROM A PREVIOUS TABLE WITH A HIGHER EPSILON
//TAKES AS INPUT:
//The data points (Database) gpuLookupArray
//A lookup array that points to an array with the neighbors of each data point (directNeighborArray)
//The total number of direct neighbors: totalDirectNeighbors
//epsilon
//previousEpsilon- the epsilon that made the input direct neighbors: used to estimate batch sizes for the new epsilon
//The resulting neighborTable to be passed into DBSCAN
//The total number of neighbors in the table
//It batches the results off of the GPU.
//However, if the number of direct neighbors are too large, we don't batch these on and off in addition to the resultset
//We return false and generate a new neighborTable using the index and not another neighborTable)
bool generateDistanceTableFromPreviousTableBatches(std::vector<struct dataElem> * dataPoints, struct gpulookuptable * gpuLookupArray, int * directNeighborArray, unsigned int * totalDirectNeighbors, double * epsilon, double * previousEpsilon,  struct table * neighborTable, unsigned int * totalNeighbors)
{

	

	double tKernelResultsStart=omp_get_wtime();
	printf("\nIn generate from previous table:\nDatapoints: %lu, \nTotal direct neighbors: %d\n",dataPoints->size(), *totalDirectNeighbors);

	cout<<"\n** Last CUDA error start of fn: "<<cudaGetLastError();

	printf("\n\nNOTE THAT SEG FAULTS ARE TYPICALLY DUE TO INSUFFICIENT BUFFER SPACE FOR THE RESULTS WHEN BATCHING\n\n");

	//CUDA error code:
	cudaError_t errCode;

	unsigned int * DBSIZE;
	DBSIZE=(unsigned int*)malloc(sizeof(unsigned int));
	*DBSIZE=dataPoints->size();
	
	
	 

	printf("\n in generate previous table GPU method: DNSIZE is: %u",*DBSIZE);cout.flush();


	///////////////////////////////////
	//COPY THE DATABASE TO THE GPU
	///////////////////////////////////

	struct point * database;
	database=(struct point*)malloc(sizeof(struct point)*(*DBSIZE));
	


	struct point * dev_database;
	dev_database=(struct point*)malloc(sizeof(struct point)*(*DBSIZE));

	//allocate memory on device:
	
	errCode=cudaMalloc( (void**)&dev_database, sizeof(struct point)*(*DBSIZE));	
	if(errCode != cudaSuccess) {
	cout << "\nError: database (in previous table method) Got error with code " << errCode << endl; cout.flush(); 
	}

	//first, we copy the x and y values from dataPoints to the database
	for (int i=0; i<*DBSIZE; i++)
	{
		database[i].x=(*dataPoints)[i].x;
		database[i].y=(*dataPoints)[i].y;
	}

	//copy database to the device:
	errCode=cudaMemcpy(dev_database, database, sizeof(struct point)*(*DBSIZE), cudaMemcpyHostToDevice);	
	if(errCode != cudaSuccess) {
	cout << "\nError: database memcopy (in previous table method) Got error with code " << errCode << endl; 
	}

	///////////////////////////////////
	//END COPY THE DATABASE TO THE GPU
	///////////////////////////////////


	///////////////////////////////
	//Copy the lookup array struct to the GPU:
	///////////////////////////////

	struct gpulookuptable * dev_gpuLookupArray;
	dev_gpuLookupArray=(struct gpulookuptable*)malloc(sizeof(struct gpulookuptable)*(*DBSIZE));

	//allocate memory on device:
	errCode=cudaMalloc( (void**)&dev_gpuLookupArray, sizeof(struct gpulookuptable)*(*DBSIZE));
		
	if(errCode != cudaSuccess) {
	cout << "\nError: gpu lookup array Got error with code " << errCode << endl; cout.flush(); 
	}

	printf("\nSize of lookup table: %f (GiB)", (double)sizeof(struct gpulookuptable)*(*DBSIZE)/(1024*1024*1024));
		


	//copy lookup array to the device:
	errCode=cudaMemcpy(dev_gpuLookupArray, gpuLookupArray, sizeof(struct gpulookuptable)*(*DBSIZE), cudaMemcpyHostToDevice);	
	if(errCode != cudaSuccess) {
	cout << "\nError: lookup array memcpy Got error with code " << errCode << endl; 
	}


	///////////////////////////////
	//END Copy the lookup array struct to the GPU:
	///////////////////////////////
	

	///////////////////////////////
	//Copy direct neighbor array to the GPU:
	///////////////////////////////

	int * dev_directNeighborArray;
	dev_directNeighborArray=(int*)malloc(sizeof(int)*(*totalDirectNeighbors));

	//allocate memory on device for the direct neighbor array:
	errCode=cudaMalloc( (void**)&dev_directNeighborArray, sizeof(int)*(*totalDirectNeighbors));
		
	if(errCode != cudaSuccess) {
	cout << "\nError: gpu direct neighbor array Got error with code " << errCode << endl; cout.flush(); 
	}

	//copy direct neighbor array to the device:
	errCode=cudaMemcpy(dev_directNeighborArray, directNeighborArray, sizeof(int)*(*totalDirectNeighbors), cudaMemcpyHostToDevice);	
	if(errCode != cudaSuccess) {
	cout << "\nError: direct neighbor array memcpy Got error with code " << errCode << endl; 
	}

	printf("\nSize of direct neighbor array: %f (GiB)", (double)sizeof(int)*(*totalDirectNeighbors)/(1024*1024*1024));

	///////////////////////////////
	//END Copy direct neighbor array to the GPU:
	///////////////////////////////


	///////////////////////////////
	//copy the size of the database
	///////////////////////////////
		//number of threads per gpu stream

	//THE NUMBER OF THREADS THAT ARE LAUNCHED IN A SINGLE KERNEL INVOCATION
	//CAN BE FEWER THAN THE NUMBER OF ELEMENTS IN THE DATABASE IF MORE THAN 1 BATCH
	unsigned int * N;
	N=(unsigned int*)malloc(sizeof(unsigned int)*GPUSTREAMS);
	


	unsigned int * dev_N; 
	dev_N=(unsigned int*)malloc(sizeof(unsigned int)*GPUSTREAMS);

	//allocate on the device
	errCode=cudaMalloc((void**)&dev_N, sizeof(unsigned int)*GPUSTREAMS);
	if(errCode != cudaSuccess) {
	cout << "\nError: dev_N Got error with code " << errCode << endl; 
	}	


	//offset into the database when batching the results
	unsigned int * batchOffset; 
	batchOffset=(unsigned int*)malloc(sizeof(unsigned int)*GPUSTREAMS);
	//*batchOffset=0;


	unsigned int * dev_offset; 
	dev_offset=(unsigned int*)malloc(sizeof(unsigned int)*GPUSTREAMS);

	//allocate on the device
	errCode=cudaMalloc((void**)&dev_offset, sizeof(unsigned int)*GPUSTREAMS);
	if(errCode != cudaSuccess) {
	cout << "\nError: offset Got error with code " << errCode << endl; 
	}

	//Batch number to calculate the point to process (in conjunction with the offset)

	//offset into the database when batching the results
	unsigned int * batchNumber; 
	batchNumber=(unsigned int*)malloc(sizeof(unsigned int)*GPUSTREAMS);
	//*batchOffset=0;


	unsigned int * dev_batchNumber; 
	dev_batchNumber=(unsigned int*)malloc(sizeof(unsigned int)*GPUSTREAMS);

	//allocate on the device
	errCode=cudaMalloc((void**)&dev_batchNumber, sizeof(unsigned int)*GPUSTREAMS);
	if(errCode != cudaSuccess) {
	cout << "\nError: batchNumber Got error with code " << errCode << endl; 
	}


	///////////////////////////////
	//END copy the size of the database
	///////////////////////////////



	///////////////////////////////
	//copy the newer (smaller) epsilon
	///////////////////////////////

	double * dev_epsilon;
	dev_epsilon=(double*)malloc(sizeof( double ));
	
	

	//Allocate on the device
	errCode=cudaMalloc((void**)&dev_epsilon, sizeof(double));
	if(errCode != cudaSuccess) {
	cout << "\nError: epsilon Got error with code " << errCode << endl; 
	}

	

	errCode=cudaMemcpy( dev_epsilon, epsilon, sizeof(double), cudaMemcpyHostToDevice );
	if(errCode != cudaSuccess) {
	cout << "\nError: epsilon memcpy Got error with code " << errCode << endl; 
	}


	///////////////////////////////
	//END copy the newer (smaller) epsilon
	///////////////////////////////


	///////////////////////////////////
	//ALLOCATE COUNT ON THE DEVICE (THE NUMBER OF RESULT ITEMS)
	///////////////////////////////////


	//count values - for an individual kernel launch
	//need different count values for each stream
	unsigned int * cnt;
	cnt=(unsigned int*)malloc(sizeof(unsigned int)*GPUSTREAMS);
	*cnt=0;

	unsigned int * dev_cnt; 
	dev_cnt=(unsigned int*)malloc(sizeof(unsigned int)*GPUSTREAMS);
	*dev_cnt=0;

	//allocate on the device
	errCode=cudaMalloc((unsigned int**)&dev_cnt, sizeof(unsigned int)*GPUSTREAMS);	
	if(errCode != cudaSuccess) {
	cout << "\nError: cnt Got error with code " << errCode << endl; 
	}	

	// errCode=cudaMemcpy( dev_cnt, cnt, sizeof(unsigned int), cudaMemcpyHostToDevice );
	// if(errCode != cudaSuccess) {
	// cout << "\nError: dev_cnt memcpy Got error with code " << errCode << endl; 
	// }



	///////////////////////////////////
	//END ALLOCATE COUNT ON THE DEVICE (THE NUMBER OF RESULT ITEMS)
	///////////////////////////////////





	///////////////////////////////////
	//ALLOCATE MEMORY FOR THE RESULT SET USING PREVIOUS SIZE OF NEIGHBORTABLE
	///////////////////////////////////
	

	//NEED BUFFERS ON THE GPU AND THE HOST FOR THE NUMBER OF CONCURRENT STREAMS	
	//GPU BUFFER ON THE DEVICE
	//BUFFER ON THE HOST WITH PINNED MEMORY FOR FAST MEMCPY
	//BUFFER ON THE HOST TO DUMP THE RESULTS OF BATCHES SO THAT GPU THREADS CAN CONTINUE
	//EXECUTING STREAMS ON THE HOST

	unsigned int GPUBufferSize=100000000;
	double alpha=1; //overestimation factor is greater for the table because as epsilon increases, the 
						//total number of neighbors within the epsilon neighborhood increases at a lower rate
						//i.e., as epsilon approaches infinity, the total number of neighbors within epsilon
						//becomes constant.
	int numBatches=0;
		

	double areaRatioNewOldEpsilon=(M_PI*(*epsilon)*(*epsilon))/(M_PI*(*previousEpsilon)*(*previousEpsilon));	

	unsigned int estimatedTotalSize=(*totalDirectNeighbors)*areaRatioNewOldEpsilon*(1.0+alpha);

	printf("\nPrevious table size: %u, area ratio of epsilons: %f, estimated total size (incl. alpha): %u", *totalDirectNeighbors, areaRatioNewOldEpsilon, estimatedTotalSize);	



	//to accomodate small datasets, we need smaller buffers because the pinned memory malloc is expensive
	if (estimatedTotalSize<(GPUBufferSize*GPUSTREAMS))
	{
		GPUBufferSize=estimatedTotalSize/GPUSTREAMS;	//but we fix the 3 streams still (thats why divide by 3).			
															
		
	}
	
	numBatches=ceil(estimatedTotalSize*1.0/GPUBufferSize*1.0);


	printf("\n\nNumber of batches: %d, buffer size: %d\n\n", numBatches, GPUBufferSize);

	//GPU MEMORY ALLOCATION:

	//CHANGING THE RESULTS TO KEY VALUE PAIR SORT, WHICH IS TWO ARRAYS
	//KEY IS THE POINT ID
	//THE VALUE IS THE POINT ID WITHIN THE DISTANCE OF KEY

	int * dev_pointIDKey[GPUSTREAMS]; //key
	int * dev_pointInDistValue[GPUSTREAMS]; //value

	

	for (int i=0; i<GPUSTREAMS; i++)
	{
		errCode=cudaMalloc((void **)&dev_pointIDKey[i], sizeof(int)*GPUBufferSize);
		if(errCode != cudaSuccess) {
		cout << "CUDA: Got error with code " << errCode << endl; //2 means not enough memory
		}

		errCode=cudaMalloc((void **)&dev_pointInDistValue[i], sizeof(int)*GPUBufferSize);
		if(errCode != cudaSuccess) {
		cout << "CUDA: Got error with code " << errCode << endl; //2 means not enough memory
		}


	}


	

	//HOST RESULT ALLOCATION FOR THE GPU TO COPY THE DATA INTO A PINNED MEMORY ALLOCATION
	//ON THE HOST
	//pinned result set memory for the host
	//the number of elements are recorded for that batch in resultElemCountPerBatch
	//NEED PINNED MEMORY ALSO BECAUSE YOU NEED IT TO USE STREAMS IN THRUST FOR THE MEMCOPY OF THE SORTED RESULTS	

	//PINNED MEMORY TO COPY FROM THE GPU	
	int * pointIDKey[GPUSTREAMS]; //key
	int * pointInDistValue[GPUSTREAMS]; //value
	

	double tstartpinnedresults=omp_get_wtime();
	
	for (int i=0; i<GPUSTREAMS; i++)
	{
	cudaMallocHost((void **) &pointIDKey[i], sizeof(int)*GPUBufferSize);
	cudaMallocHost((void **) &pointInDistValue[i], sizeof(int)*GPUBufferSize);
	}

	double tendpinnedresults=omp_get_wtime();
	printf("\nTime to allocate pinned memory for results: %f", tendpinnedresults - tstartpinnedresults);
	

	// cudaMalloc((void **) &pointIDKey, sizeof(int)*GPUBufferSize*NUMBATCHES);
	// cudaMalloc((void **) &pointInDistValue, sizeof(int)*GPUBufferSize*NUMBATCHES);




	printf("\nmemory requested for results ON GPU (GiB): %f",(double)(sizeof(int)*2*GPUBufferSize*GPUSTREAMS)/(1024*1024*1024));
	printf("\nmemory requested for results in MAIN MEMORY (GiB): %f",(double)(sizeof(int)*2*GPUBufferSize*GPUSTREAMS)/(1024*1024*1024));

	
	///////////////////////////////////
	//END ALLOCATE MEMORY FOR THE RESULT SET
	///////////////////////////////////





	/////////////////////////////////
	//SET OPENMP ENVIRONMENT VARIABLES
	////////////////////////////////

	omp_set_num_threads(GPUSTREAMS);
	

	/////////////////////////////////
	//END SET OPENMP ENVIRONMENT VARIABLES
	////////////////////////////////
	
	

	/////////////////////////////////
	//CREATE STREAMS
	////////////////////////////////

	cudaStream_t stream[GPUSTREAMS];
	
	for (int i=0; i<GPUSTREAMS; i++){

    //cudaStreamCreate(&stream[i]);
	cudaStreamCreateWithFlags(&stream[i], cudaStreamNonBlocking);

	}	

	/////////////////////////////////
	//END CREATE STREAMS
	////////////////////////////////



	///////////////////////////////////
	//LAUNCH KERNEL IN BATCHES
	///////////////////////////////////
		
	//since we use the strided scheme, some of the batch sizes
	//are off by 1 of each other, a first group of batches will
	//have 1 extra data point to process, and we calculate which batch numbers will 
	//have that.  The batchSize is the lower value (+1 is added to the first ones)


	unsigned int batchSize=(*DBSIZE)/numBatches;
	unsigned int batchesThatHaveOneMore=(*DBSIZE)-(batchSize*numBatches); //batch number 0- < this value have one more
	printf("\n\n***Batches that have one more: %u batchSize(N): %u, \n\n***",batchSize, batchesThatHaveOneMore);

	unsigned int totalResultsLoop=0;


	/*
	//////OLD NON-BATCHED
	const int TOTALBLOCKS=ceil((1.0*(*N))/(1.0*BLOCKSIZE));	
	printf("\ntotal blocks (from previous table method): %d",TOTALBLOCKS);

	//execute kernel	
	
	calcNeighborsFromTableKernel<<< TOTALBLOCKS, BLOCKSIZE >>>(dev_N, dev_gpuLookupArray, dev_directNeighborArray, dev_cnt, dev_epsilon, dev_database, dev_pointIDKey, dev_pointInDistValue);
	cout <<endl<<"After kernel launch, Error code: "<<cudaGetLastError()<<endl;
	if ( cudaSuccess != cudaGetLastError() ){
    	printf( "\nError in kernel launch (previous table method)!" );
    	// cout <<endl<<"Error code: "<<cudaGetLastError()<<endl;
    }
	*/

		//FOR LOOP OVER THE NUMBER OF BATCHES STARTS HERE
	#pragma omp parallel for schedule(static,1) reduction(+:totalResultsLoop) num_threads(GPUSTREAMS)
	for (int i=0; i<numBatches; i++)
	{	

		int tid=omp_get_thread_num();
		
		printf("\ntid: %d, starting iteration: %d",tid,i);

		//N NOW BECOMES THE NUMBER OF POINTS TO PROCESS PER BATCH
		//AS ONE THREAD PROCESSES A SINGLE POINT
			
    	
			
		if (i<batchesThatHaveOneMore)
		{
			N[tid]=batchSize+1;	
			printf("\nN: %d, tid: %d",N[tid], tid);
		}
		else
		{
			N[tid]=batchSize;	
			printf("\nN (1 less): %d tid: %d",N[tid], tid);
		}

		//set relevant parameters for the batched execution that get reset
			
		//copy N to device 
		//N IS THE NUMBER OF THREADS
		errCode=cudaMemcpyAsync( &dev_N[tid], &N[tid], sizeof(unsigned int), cudaMemcpyHostToDevice, stream[tid] );
		if(errCode != cudaSuccess) {
		cout << "\nError: N Got error with code " << errCode << endl; 
		}

		//the batched result set size (reset to 0):
		cnt[tid]=0;
		errCode=cudaMemcpyAsync( &dev_cnt[tid], &cnt[tid], sizeof(unsigned int), cudaMemcpyHostToDevice, stream[tid] );
		if(errCode != cudaSuccess) {
		cout << "\nError: dev_cnt memcpy Got error with code " << errCode << endl; 
		}

		//the offset for batching, which keeps track of where to start processing at each batch
		//batchOffset[tid]=i*batchSize; //original
		batchOffset[tid]=numBatches; //for the strided
		errCode=cudaMemcpyAsync( &dev_offset[tid], &batchOffset[tid], sizeof(unsigned int), cudaMemcpyHostToDevice, stream[tid] );
		if(errCode != cudaSuccess) {
		cout << "\nError: dev_offset memcpy Got error with code " << errCode << endl; 
		}

		//the batch number for batching with strided
		batchNumber[tid]=i;
		errCode=cudaMemcpyAsync( &dev_batchNumber[tid], &batchNumber[tid], sizeof(unsigned int), cudaMemcpyHostToDevice, stream[tid] );
		if(errCode != cudaSuccess) {
		cout << "\nError: dev_batchNumber memcpy Got error with code " << errCode << endl; 
		}

		const int TOTALBLOCKS=ceil((1.0*(N[tid]))/(1.0*BLOCKSIZE));	
		printf("\ntotal blocks: %d",TOTALBLOCKS);

		//execute kernel	
		//0 is shared memory pool
		calcNeighborsFromTableKernelBatches<<< TOTALBLOCKS, BLOCKSIZE, 0, stream[tid] >>>(&dev_N[tid], &dev_offset[tid], &dev_batchNumber[tid], dev_gpuLookupArray, dev_directNeighborArray, &dev_cnt[tid], dev_epsilon, dev_database, dev_pointIDKey[tid], dev_pointInDistValue[tid]);

		cout <<"\n\nKERNEL LAUNCH RETURN: "<<cudaGetLastError()<<endl<<endl;
			if ( cudaSuccess != cudaGetLastError() ){
		    	cout <<"\n\nERROR IN KERNEL LAUNCH. ERROR: "<<cudaSuccess<<endl<<endl;
		    }

	   	// find the size of the number of results
		errCode=cudaMemcpyAsync( &cnt[tid], &dev_cnt[tid], sizeof(unsigned int), cudaMemcpyDeviceToHost, stream[tid] );
		if(errCode != cudaSuccess) {
		cout << "\nError: getting cnt from GPU Got error with code " << errCode << endl; 
		}
		else
		{
			printf("\n\nGPU: result set size within epsilon (CONSTRUCT FROM PREVIOUS NEIGHBORTABLE BATCHES): %d\n\n",cnt[tid]);
		} 



		////////////////////////////////////
		//SORT THE TABLE DATA ON THE GPU
		//THERE IS NO ORDERING BETWEEN EACH POINT AND THE ONES THAT IT'S WITHIN THE DISTANCE OF
		////////////////////////////////////

		/////////////////////////////
		//ONE PROBLEM WITH NOT TRANSFERING THE RESULT OFF OF THE DEVICE IS THAT
		//YOU CAN'T RESIZE THE RESULTS TO BE THE SIZE OF *CNT
		//SO THEN YOU HAVE POTENTIALLY LOTS OF WASTED SPACE
		/////////////////////////////

		//sort by key with the data already on the device:
		//wrap raw pointer with a device_ptr to use with Thrust functions
		thrust::device_ptr<int> dev_keys_ptr(dev_pointIDKey[tid]);
		thrust::device_ptr<int> dev_data_ptr(dev_pointInDistValue[tid]);

		//XXXXXXXXXXXXXXXX
		//THRUST USING STREAMS REQUIRES THRUST V1.8 
		//SEEMS TO BE WORKING :)
		//XXXXXXXXXXXXXXXX

		try{
		thrust::sort_by_key(thrust::cuda::par.on(stream[tid]), dev_keys_ptr, dev_keys_ptr + cnt[tid], dev_data_ptr);
		}
		catch(std::bad_alloc &e)
		  {
		    std::cerr << "Ran out of memory while sorting" << std::endl;
		    exit(-1);
		  }


		//thrust with streams into individual buffers for each batch
			cudaMemcpyAsync(thrust::raw_pointer_cast(pointIDKey[tid]), thrust::raw_pointer_cast(dev_keys_ptr), cnt[tid]*sizeof(int), cudaMemcpyDeviceToHost, stream[tid]);
			cudaMemcpyAsync(thrust::raw_pointer_cast(pointInDistValue[tid]), thrust::raw_pointer_cast(dev_data_ptr), cnt[tid]*sizeof(int), cudaMemcpyDeviceToHost, stream[tid]);	

			//need to make sure the data is copied before constructing portion of the neighbor table
			cudaStreamSynchronize(stream[tid]);


			//construct portion of the table:
			double tableconstuctstart=omp_get_wtime();
			constructNeighborTableKeyValue(pointIDKey[tid], pointInDistValue[tid], neighborTable, &cnt[tid]);
			double tableconstuctend=omp_get_wtime();	
			
			printf("\nTable construct time: %f", tableconstuctend - tableconstuctstart);

			//add the batched result set size to the total count
			totalResultsLoop+=cnt[tid];


			printf("\nRunning total of total size of result array, tid: %d: %u", tid, totalResultsLoop);
			//}




		

		} //END LOOP OVER THE GPU BATCHES  


	printf("\nTOTAL RESULT SET SIZE ON HOST:  %d", totalResultsLoop);
	*totalNeighbors=totalResultsLoop;
	

	double tKernelResultsEnd=omp_get_wtime();
	
	printf("\nTime to launch kernel and execute all of the previous part of the method and get the results back: %f",tKernelResultsEnd-tKernelResultsStart);



	

	///////////////////////////////
	//END EXECUTE KERNEL
	///////////////////////////////

    ///////////////////////////////////
	//GET RESULT SET
	///////////////////////////////////

	//first find the size of the number of results
	//errCode=cudaMemcpy( cnt, dev_cnt, sizeof(unsigned int), cudaMemcpyDeviceToHost );
	// if(errCode != cudaSuccess) {
	// cout << "\nError: getting cnt from GPU Got error with code " << errCode << endl; 
	// }
	// else
	// {
	// 	printf("\nGPU: result set size on within epsilon: %d",*cnt);
	// }


	/*
	errCode=cudaMemcpy( cnt, dev_cnt, sizeof(unsigned int), cudaMemcpyDeviceToHost );
	if(errCode != cudaSuccess) {
	cout << "\nError: getting cnt from GPU Got error with code " << errCode << endl; 
	}
	else
	{
		printf("\nGPU: result set size on GPU within epsilon (from precomputed table method): %d",*cnt);
	}

	*/

	
	//copy the results, but only transfer the number of results, not the entire buffer
	// errCode=cudaMemcpy(results, dev_results, sizeof(struct structresults)*(*cnt), cudaMemcpyDeviceToHost );
	// if(errCode != cudaSuccess) {
	// cout << "\nError: getting results from GPU (from the precomputed table) Got error with code " << errCode << endl; 
	// }

	//*totalNeighbors=(*cnt);

	//SORTING FOR TESTING ONLY
	//XXXXXXXXX
	//XXXXXXXXX
	// std::sort(results, results+(*cnt),compResults);
	// printf("\n**** GPU\n");
	// for (int i=0; i<(*cnt); i++)
	// {
	// 	printf("\n%d,%d",results[i].pointID, results[i].pointInDist);
	// }

	//XXXXXXXXX
	//XXXXXXXXX
	//XXXXXXXXX



	///////////////////////////////////
	//END GET RESULT SET
	///////////////////////////////////

	double tFreeStart=omp_get_wtime();

	/////////////////
	//FREE
	/////////////////

	for (int i=0; i<GPUSTREAMS; i++)
	{
		errCode=cudaStreamDestroy(stream[i]);

		if(errCode != cudaSuccess) {
		cout << "\nError: destroying stream" << errCode << endl; 
		}
	}


	cudaFree(dev_directNeighborArray);
	cudaFree(dev_gpuLookupArray);
	//cudaFree(dev_results);
	cudaFree(dev_database);
	cudaFree(dev_epsilon);
	cudaFree(dev_N);
	cudaFree(dev_cnt);
	cudaFree(dev_offset);
	cudaFree(dev_batchNumber);

	//free data related to the individual streams for each batch
	for (int i=0; i<GPUSTREAMS; i++)
	{
		//free the data on the device
		cudaFree(dev_pointIDKey[i]);
		cudaFree(dev_pointInDistValue[i]);

		//free on the host
		cudaFreeHost(pointIDKey[i]);
		cudaFreeHost(pointInDistValue[i]);

	

	}

	
	cudaFree(dev_pointIDKey);
	cudaFree(dev_pointInDistValue);
	//free pinned memory on host
	cudaFreeHost(pointIDKey);
	cudaFreeHost(pointInDistValue);

		


	double tFreeEnd=omp_get_wtime();

	printf("\nTime freeing memory: %f", tFreeEnd - tFreeStart);

	cout<<"\n** last error at end of fn construct table batches: "<<cudaGetLastError();


	return true;

}










//USE THIS TO MAKE A TABLE FROM A PREVIOUS TABLE WITH A HIGHER EPSILON
//TAKES AS INPUT:
//The data points (Database) gpuLookupArray
//A lookup array that points to an array with the neighbors of each data point (directNeighborArray)
//The total number of direct neighbors: totalDirectNeighbors
//epsilon
//previousEpsilon- the epsilon that made the input direct neighbors: used to estimate batch sizes for the new epsilon
//The resulting neighborTable to be passed into DBSCAN
//The total number of neighbors in the table
//It batches the results off of the GPU.
//However, if the number of direct neighbors are too large, we don't batch these on and off in addition to the resultset
//We return false and generate a new neighborTable using the index and not another neighborTable)
bool generateDistanceTableFromPreviousTableBatchesAlternate(std::vector<struct dataElem> * dataPoints, struct gpulookuptable * gpuLookupArray, int * directNeighborArray, unsigned int * totalDirectNeighbors, double * epsilon, double * previousEpsilon,  struct neighborTableLookup * neighborTable, std::vector<struct neighborDataPtrs> * pointersToNeighbors,unsigned int * totalNeighbors)
{

	

	double tKernelResultsStart=omp_get_wtime();
	printf("\nIn generate from previous table:\nDatapoints: %lu, \nTotal direct neighbors: %d\n",dataPoints->size(), *totalDirectNeighbors);

	cout<<"\n** Last CUDA error start of fn: "<<cudaGetLastError();

	printf("\n\nNOTE THAT SEG FAULTS ARE TYPICALLY DUE TO INSUFFICIENT BUFFER SPACE FOR THE RESULTS WHEN BATCHING\n\n");

	//CUDA error code:
	cudaError_t errCode;

	unsigned int * DBSIZE;
	DBSIZE=(unsigned int*)malloc(sizeof(unsigned int));
	*DBSIZE=dataPoints->size();
	
	
	 

	printf("\n in generate previous table GPU method: DNSIZE is: %u",*DBSIZE);cout.flush();


	///////////////////////////////////
	//COPY THE DATABASE TO THE GPU
	///////////////////////////////////

	struct point * database;
	database=(struct point*)malloc(sizeof(struct point)*(*DBSIZE));
	


	struct point * dev_database;
	dev_database=(struct point*)malloc(sizeof(struct point)*(*DBSIZE));

	//allocate memory on device:
	
	errCode=cudaMalloc( (void**)&dev_database, sizeof(struct point)*(*DBSIZE));	
	if(errCode != cudaSuccess) {
	cout << "\nError: database (in previous table method) Got error with code " << errCode << endl; cout.flush(); 
	}

	//first, we copy the x and y values from dataPoints to the database
	for (int i=0; i<*DBSIZE; i++)
	{
		database[i].x=(*dataPoints)[i].x;
		database[i].y=(*dataPoints)[i].y;
	}

	//copy database to the device:
	errCode=cudaMemcpy(dev_database, database, sizeof(struct point)*(*DBSIZE), cudaMemcpyHostToDevice);	
	if(errCode != cudaSuccess) {
	cout << "\nError: database memcopy (in previous table method) Got error with code " << errCode << endl; 
	}

	///////////////////////////////////
	//END COPY THE DATABASE TO THE GPU
	///////////////////////////////////


	///////////////////////////////
	//Copy the lookup array struct to the GPU:
	///////////////////////////////

	struct gpulookuptable * dev_gpuLookupArray;
	dev_gpuLookupArray=(struct gpulookuptable*)malloc(sizeof(struct gpulookuptable)*(*DBSIZE));

	//allocate memory on device:
	errCode=cudaMalloc( (void**)&dev_gpuLookupArray, sizeof(struct gpulookuptable)*(*DBSIZE));
		
	if(errCode != cudaSuccess) {
	cout << "\nError: gpu lookup array Got error with code " << errCode << endl; cout.flush(); 
	}

	printf("\nSize of lookup table: %f (GiB)", (double)sizeof(struct gpulookuptable)*(*DBSIZE)/(1024*1024*1024));
		


	//copy lookup array to the device:
	errCode=cudaMemcpy(dev_gpuLookupArray, gpuLookupArray, sizeof(struct gpulookuptable)*(*DBSIZE), cudaMemcpyHostToDevice);	
	if(errCode != cudaSuccess) {
	cout << "\nError: lookup array memcpy Got error with code " << errCode << endl; 
	}


	///////////////////////////////
	//END Copy the lookup array struct to the GPU:
	///////////////////////////////
	

	///////////////////////////////
	//Copy direct neighbor array to the GPU:
	///////////////////////////////

	int * dev_directNeighborArray;
	dev_directNeighborArray=(int*)malloc(sizeof(int)*(*totalDirectNeighbors));

	//allocate memory on device for the direct neighbor array:
	errCode=cudaMalloc( (void**)&dev_directNeighborArray, sizeof(int)*(*totalDirectNeighbors));
		
	if(errCode != cudaSuccess) {
	cout << "\nError: gpu direct neighbor array Got error with code " << errCode << endl; cout.flush(); 
	}

	//copy direct neighbor array to the device:
	errCode=cudaMemcpy(dev_directNeighborArray, directNeighborArray, sizeof(int)*(*totalDirectNeighbors), cudaMemcpyHostToDevice);	
	if(errCode != cudaSuccess) {
	cout << "\nError: direct neighbor array memcpy Got error with code " << errCode << endl; 
	}

	printf("\nSize of direct neighbor array: %f (GiB)", (double)sizeof(int)*(*totalDirectNeighbors)/(1024*1024*1024));

	///////////////////////////////
	//END Copy direct neighbor array to the GPU:
	///////////////////////////////


	///////////////////////////////
	//copy the size of the database
	///////////////////////////////
		//number of threads per gpu stream

	//THE NUMBER OF THREADS THAT ARE LAUNCHED IN A SINGLE KERNEL INVOCATION
	//CAN BE FEWER THAN THE NUMBER OF ELEMENTS IN THE DATABASE IF MORE THAN 1 BATCH
	unsigned int * N;
	N=(unsigned int*)malloc(sizeof(unsigned int)*GPUSTREAMS);
	


	unsigned int * dev_N; 
	dev_N=(unsigned int*)malloc(sizeof(unsigned int)*GPUSTREAMS);

	//allocate on the device
	errCode=cudaMalloc((void**)&dev_N, sizeof(unsigned int)*GPUSTREAMS);
	if(errCode != cudaSuccess) {
	cout << "\nError: dev_N Got error with code " << errCode << endl; 
	}	


	//offset into the database when batching the results
	unsigned int * batchOffset; 
	batchOffset=(unsigned int*)malloc(sizeof(unsigned int)*GPUSTREAMS);
	//*batchOffset=0;


	unsigned int * dev_offset; 
	dev_offset=(unsigned int*)malloc(sizeof(unsigned int)*GPUSTREAMS);

	//allocate on the device
	errCode=cudaMalloc((void**)&dev_offset, sizeof(unsigned int)*GPUSTREAMS);
	if(errCode != cudaSuccess) {
	cout << "\nError: offset Got error with code " << errCode << endl; 
	}

	//Batch number to calculate the point to process (in conjunction with the offset)

	//offset into the database when batching the results
	unsigned int * batchNumber; 
	batchNumber=(unsigned int*)malloc(sizeof(unsigned int)*GPUSTREAMS);
	//*batchOffset=0;


	unsigned int * dev_batchNumber; 
	dev_batchNumber=(unsigned int*)malloc(sizeof(unsigned int)*GPUSTREAMS);

	//allocate on the device
	errCode=cudaMalloc((void**)&dev_batchNumber, sizeof(unsigned int)*GPUSTREAMS);
	if(errCode != cudaSuccess) {
	cout << "\nError: batchNumber Got error with code " << errCode << endl; 
	}


	///////////////////////////////
	//END copy the size of the database
	///////////////////////////////



	///////////////////////////////
	//copy the newer (smaller) epsilon
	///////////////////////////////

	double * dev_epsilon;
	dev_epsilon=(double*)malloc(sizeof( double ));
	
	

	//Allocate on the device
	errCode=cudaMalloc((void**)&dev_epsilon, sizeof(double));
	if(errCode != cudaSuccess) {
	cout << "\nError: epsilon Got error with code " << errCode << endl; 
	}

	

	errCode=cudaMemcpy( dev_epsilon, epsilon, sizeof(double), cudaMemcpyHostToDevice );
	if(errCode != cudaSuccess) {
	cout << "\nError: epsilon memcpy Got error with code " << errCode << endl; 
	}


	///////////////////////////////
	//END copy the newer (smaller) epsilon
	///////////////////////////////


	///////////////////////////////////
	//ALLOCATE COUNT ON THE DEVICE (THE NUMBER OF RESULT ITEMS)
	///////////////////////////////////


	//count values - for an individual kernel launch
	//need different count values for each stream
	unsigned int * cnt;
	cnt=(unsigned int*)malloc(sizeof(unsigned int)*GPUSTREAMS);
	*cnt=0;

	unsigned int * dev_cnt; 
	dev_cnt=(unsigned int*)malloc(sizeof(unsigned int)*GPUSTREAMS);
	*dev_cnt=0;

	//allocate on the device
	errCode=cudaMalloc((unsigned int**)&dev_cnt, sizeof(unsigned int)*GPUSTREAMS);	
	if(errCode != cudaSuccess) {
	cout << "\nError: cnt Got error with code " << errCode << endl; 
	}	

	// errCode=cudaMemcpy( dev_cnt, cnt, sizeof(unsigned int), cudaMemcpyHostToDevice );
	// if(errCode != cudaSuccess) {
	// cout << "\nError: dev_cnt memcpy Got error with code " << errCode << endl; 
	// }



	///////////////////////////////////
	//END ALLOCATE COUNT ON THE DEVICE (THE NUMBER OF RESULT ITEMS)
	///////////////////////////////////





	///////////////////////////////////
	//ALLOCATE MEMORY FOR THE RESULT SET USING PREVIOUS SIZE OF NEIGHBORTABLE
	///////////////////////////////////
	

	//NEED BUFFERS ON THE GPU AND THE HOST FOR THE NUMBER OF CONCURRENT STREAMS	
	//GPU BUFFER ON THE DEVICE
	//BUFFER ON THE HOST WITH PINNED MEMORY FOR FAST MEMCPY
	//BUFFER ON THE HOST TO DUMP THE RESULTS OF BATCHES SO THAT GPU THREADS CAN CONTINUE
	//EXECUTING STREAMS ON THE HOST

	unsigned int GPUBufferSize=100000000;
	double alpha=0.6; //overestimation factor is greater for the table because as epsilon increases, the 
						//total number of neighbors within the epsilon neighborhood increases at a lower rate
						//i.e., as epsilon approaches infinity, the total number of neighbors within epsilon
						//becomes constant.
	int numBatches=0;
		

	double areaRatioNewOldEpsilon=(M_PI*(*epsilon)*(*epsilon))/(M_PI*(*previousEpsilon)*(*previousEpsilon));	

	unsigned int estimatedTotalSize=(*totalDirectNeighbors)*areaRatioNewOldEpsilon*(1.0+alpha);

	printf("\nPrevious table size: %u, area ratio of epsilons: %f, estimated total size (incl. alpha): %u", *totalDirectNeighbors, areaRatioNewOldEpsilon, estimatedTotalSize);	



	//to accomodate small datasets, we need smaller buffers because the pinned memory malloc is expensive
	if (estimatedTotalSize<(GPUBufferSize*GPUSTREAMS))
	{
		GPUBufferSize=estimatedTotalSize/GPUSTREAMS;	//but we fix the 3 streams still (thats why divide by 3).			
															
		
	}
	
	numBatches=ceil(estimatedTotalSize*1.0/GPUBufferSize*1.0);


	printf("\n\nNumber of batches: %d, buffer size: %d\n\n", numBatches, GPUBufferSize);

	//GPU MEMORY ALLOCATION:

	//CHANGING THE RESULTS TO KEY VALUE PAIR SORT, WHICH IS TWO ARRAYS
	//KEY IS THE POINT ID
	//THE VALUE IS THE POINT ID WITHIN THE DISTANCE OF KEY

	int * dev_pointIDKey[GPUSTREAMS]; //key
	int * dev_pointInDistValue[GPUSTREAMS]; //value

	

	for (int i=0; i<GPUSTREAMS; i++)
	{
		errCode=cudaMalloc((void **)&dev_pointIDKey[i], sizeof(int)*GPUBufferSize);
		if(errCode != cudaSuccess) {
		cout << "CUDA: Got error with code " << errCode << endl; //2 means not enough memory
		}

		errCode=cudaMalloc((void **)&dev_pointInDistValue[i], sizeof(int)*GPUBufferSize);
		if(errCode != cudaSuccess) {
		cout << "CUDA: Got error with code " << errCode << endl; //2 means not enough memory
		}


	}


	

	//HOST RESULT ALLOCATION FOR THE GPU TO COPY THE DATA INTO A PINNED MEMORY ALLOCATION
	//ON THE HOST
	//pinned result set memory for the host
	//the number of elements are recorded for that batch in resultElemCountPerBatch
	//NEED PINNED MEMORY ALSO BECAUSE YOU NEED IT TO USE STREAMS IN THRUST FOR THE MEMCOPY OF THE SORTED RESULTS	

	//PINNED MEMORY TO COPY FROM THE GPU	
	int * pointIDKey[GPUSTREAMS]; //key
	int * pointInDistValue[GPUSTREAMS]; //value
	

	double tstartpinnedresults=omp_get_wtime();
	
	for (int i=0; i<GPUSTREAMS; i++)
	{
	cudaMallocHost((void **) &pointIDKey[i], sizeof(int)*GPUBufferSize);
	cudaMallocHost((void **) &pointInDistValue[i], sizeof(int)*GPUBufferSize);
	}

	double tendpinnedresults=omp_get_wtime();
	printf("\nTime to allocate pinned memory for results: %f", tendpinnedresults - tstartpinnedresults);
	

	// cudaMalloc((void **) &pointIDKey, sizeof(int)*GPUBufferSize*NUMBATCHES);
	// cudaMalloc((void **) &pointInDistValue, sizeof(int)*GPUBufferSize*NUMBATCHES);




	printf("\nmemory requested for results ON GPU (GiB): %f",(double)(sizeof(int)*2*GPUBufferSize*GPUSTREAMS)/(1024*1024*1024));
	printf("\nmemory requested for results in MAIN MEMORY (GiB): %f",(double)(sizeof(int)*2*GPUBufferSize*GPUSTREAMS)/(1024*1024*1024));

	
	///////////////////////////////////
	//END ALLOCATE MEMORY FOR THE RESULT SET
	///////////////////////////////////




	///////////////////
	//ALLOCATE POINTERS TO INTEGER ARRAYS FOR THE VALUES FOR THE NEIGHBORTABLES
	///////////////////

	//THE NUMBER OF POINTERS IS EQUAL TO THE NUMBER OF BATCHES
	for (int i=0; i<numBatches; i++)
	{
		int *ptr;
		struct neighborDataPtrs tmpStruct;
		tmpStruct.dataPtr=ptr;
		tmpStruct.sizeOfDataArr=0;
		
		pointersToNeighbors->push_back(tmpStruct);
	}

	///////////////////
	//END ALLOCATE POINTERS TO INTEGER ARRAYS FOR THE VALUES FOR THE NEIGHBORTABLES
	///////////////////








	/////////////////////////////////
	//SET OPENMP ENVIRONMENT VARIABLES
	////////////////////////////////

	omp_set_nested(1);
	omp_set_num_threads(GPUSTREAMS);
	

	/////////////////////////////////
	//END SET OPENMP ENVIRONMENT VARIABLES
	////////////////////////////////
	
	

	/////////////////////////////////
	//CREATE STREAMS
	////////////////////////////////

	cudaStream_t stream[GPUSTREAMS];
	
	for (int i=0; i<GPUSTREAMS; i++){

    //cudaStreamCreate(&stream[i]);
	cudaStreamCreateWithFlags(&stream[i], cudaStreamNonBlocking);

	}	

	/////////////////////////////////
	//END CREATE STREAMS
	////////////////////////////////



	///////////////////////////////////
	//LAUNCH KERNEL IN BATCHES
	///////////////////////////////////
		
	//since we use the strided scheme, some of the batch sizes
	//are off by 1 of each other, a first group of batches will
	//have 1 extra data point to process, and we calculate which batch numbers will 
	//have that.  The batchSize is the lower value (+1 is added to the first ones)


	unsigned int batchSize=(*DBSIZE)/numBatches;
	unsigned int batchesThatHaveOneMore=(*DBSIZE)-(batchSize*numBatches); //batch number 0- < this value have one more
	printf("\n\n***Batches that have one more: %u batchSize(N): %u, \n\n***",batchSize, batchesThatHaveOneMore);

	unsigned int totalResultsLoop=0;


	/*
	//////OLD NON-BATCHED
	const int TOTALBLOCKS=ceil((1.0*(*N))/(1.0*BLOCKSIZE));	
	printf("\ntotal blocks (from previous table method): %d",TOTALBLOCKS);

	//execute kernel	
	
	calcNeighborsFromTableKernel<<< TOTALBLOCKS, BLOCKSIZE >>>(dev_N, dev_gpuLookupArray, dev_directNeighborArray, dev_cnt, dev_epsilon, dev_database, dev_pointIDKey, dev_pointInDistValue);
	cout <<endl<<"After kernel launch, Error code: "<<cudaGetLastError()<<endl;
	if ( cudaSuccess != cudaGetLastError() ){
    	printf( "\nError in kernel launch (previous table method)!" );
    	// cout <<endl<<"Error code: "<<cudaGetLastError()<<endl;
    }
	*/

		//FOR LOOP OVER THE NUMBER OF BATCHES STARTS HERE
	#pragma omp parallel for schedule(static,1) reduction(+:totalResultsLoop) 
	for (int i=0; i<numBatches; i++)
	{	

		int tid=omp_get_thread_num();
		
		printf("\nMaking table from previous, tid: %d, starting iteration: %d",tid,i);

		//N NOW BECOMES THE NUMBER OF POINTS TO PROCESS PER BATCH
		//AS ONE THREAD PROCESSES A SINGLE POINT
			
    	
			
		if (i<batchesThatHaveOneMore)
		{
			N[tid]=batchSize+1;	
			printf("\nN: %d, tid: %d",N[tid], tid);
		}
		else
		{
			N[tid]=batchSize;	
			printf("\nN (1 less): %d tid: %d",N[tid], tid);
		}

		//printf("\nN is: %d, tid: %d", N[tid], tid);

		//set relevant parameters for the batched execution that get reset
			
		//copy N to device 
		//N IS THE NUMBER OF THREADS
		errCode=cudaMemcpyAsync( &dev_N[tid], &N[tid], sizeof(unsigned int), cudaMemcpyHostToDevice, stream[tid] );
		if(errCode != cudaSuccess) {
		cout << "\nError: N Got error with code " << errCode << endl; 
		}

		//the batched result set size (reset to 0):
		cnt[tid]=0;
		errCode=cudaMemcpyAsync( &dev_cnt[tid], &cnt[tid], sizeof(unsigned int), cudaMemcpyHostToDevice, stream[tid] );
		if(errCode != cudaSuccess) {
		cout << "\nError: dev_cnt memcpy Got error with code " << errCode << endl; 
		}

		//the offset for batching, which keeps track of where to start processing at each batch
		//batchOffset[tid]=i*batchSize; //original
		batchOffset[tid]=numBatches; //for the strided
		errCode=cudaMemcpyAsync( &dev_offset[tid], &batchOffset[tid], sizeof(unsigned int), cudaMemcpyHostToDevice, stream[tid] );
		if(errCode != cudaSuccess) {
		cout << "\nError: dev_offset memcpy Got error with code " << errCode << endl; 
		}

		//the batch number for batching with strided
		batchNumber[tid]=i;
		errCode=cudaMemcpyAsync( &dev_batchNumber[tid], &batchNumber[tid], sizeof(unsigned int), cudaMemcpyHostToDevice, stream[tid] );
		if(errCode != cudaSuccess) {
		cout << "\nError: dev_batchNumber memcpy Got error with code " << errCode << endl; 
		}

		const int TOTALBLOCKS=ceil((1.0*(N[tid]))/(1.0*BLOCKSIZE));	
		printf("\ntotal blocks: %d",TOTALBLOCKS);

		//execute kernel	
		//0 is shared memory pool
		calcNeighborsFromTableKernelBatches<<< TOTALBLOCKS, BLOCKSIZE, 0, stream[tid] >>>(&dev_N[tid], &dev_offset[tid], &dev_batchNumber[tid], dev_gpuLookupArray, dev_directNeighborArray, &dev_cnt[tid], dev_epsilon, dev_database, dev_pointIDKey[tid], dev_pointInDistValue[tid]);

		cout <<"\n\nKERNEL LAUNCH RETURN: "<<cudaGetLastError()<<endl<<endl;
			if ( cudaSuccess != cudaGetLastError() ){
		    	cout <<"\n\nERROR IN KERNEL LAUNCH. ERROR: "<<cudaSuccess<<endl<<endl;
		    }

	   	// find the size of the number of results
		errCode=cudaMemcpyAsync( &cnt[tid], &dev_cnt[tid], sizeof(unsigned int), cudaMemcpyDeviceToHost, stream[tid] );
		if(errCode != cudaSuccess) {
		cout << "\nError: getting cnt from GPU Got error with code " << errCode << endl; 
		}
		else
		{
			printf("\n\nGPU: result set size within epsilon (CONSTRUCT FROM PREVIOUS NEIGHBORTABLE BATCHES): %d\n\n",cnt[tid]);
		} 



		////////////////////////////////////
		//SORT THE TABLE DATA ON THE GPU
		//THERE IS NO ORDERING BETWEEN EACH POINT AND THE ONES THAT IT'S WITHIN THE DISTANCE OF
		////////////////////////////////////

		/////////////////////////////
		//ONE PROBLEM WITH NOT TRANSFERING THE RESULT OFF OF THE DEVICE IS THAT
		//YOU CAN'T RESIZE THE RESULTS TO BE THE SIZE OF *CNT
		//SO THEN YOU HAVE POTENTIALLY LOTS OF WASTED SPACE
		/////////////////////////////

		//sort by key with the data already on the device:
		//wrap raw pointer with a device_ptr to use with Thrust functions
		thrust::device_ptr<int> dev_keys_ptr(dev_pointIDKey[tid]);
		thrust::device_ptr<int> dev_data_ptr(dev_pointInDistValue[tid]);

		//XXXXXXXXXXXXXXXX
		//THRUST USING STREAMS REQUIRES THRUST V1.8 
		//SEEMS TO BE WORKING :)
		//XXXXXXXXXXXXXXXX

		try{
		thrust::sort_by_key(thrust::cuda::par.on(stream[tid]), dev_keys_ptr, dev_keys_ptr + cnt[tid], dev_data_ptr);
		}
		catch(std::bad_alloc &e)
		  {
		    std::cerr << "Ran out of memory while sorting" << std::endl;
		    exit(-1);
		  }


		//thrust with streams into individual buffers for each batch
			cudaMemcpyAsync(thrust::raw_pointer_cast(pointIDKey[tid]), thrust::raw_pointer_cast(dev_keys_ptr), cnt[tid]*sizeof(int), cudaMemcpyDeviceToHost, stream[tid]);
			cudaMemcpyAsync(thrust::raw_pointer_cast(pointInDistValue[tid]), thrust::raw_pointer_cast(dev_data_ptr), cnt[tid]*sizeof(int), cudaMemcpyDeviceToHost, stream[tid]);	

			//need to make sure the data is copied before constructing portion of the neighbor table
			cudaStreamSynchronize(stream[tid]);


			//construct portion of the table:
			double tableconstuctstart=omp_get_wtime();
			//constructNeighborTableKeyValue(pointIDKey[tid], pointInDistValue[tid], neighborTable, &cnt[tid]);
			
			
			//set the number of neighbors in the pointer struct:
			(*pointersToNeighbors)[i].sizeOfDataArr=cnt[tid];    
			(*pointersToNeighbors)[i].dataPtr=new int[cnt[tid]]; 
			constructNeighborTableKeyValueAlternateTest(pointIDKey[tid], pointInDistValue[tid], neighborTable, (*pointersToNeighbors)[i].dataPtr, &cnt[tid]);
			double tableconstuctend=omp_get_wtime();	
			
			//cout <<"\nIn neighbortable from previous table. Data array ptr: "<<(*pointersToNeighbors)[i].dataPtr<<" , size of data array: "<<(*pointersToNeighbors)[i].sizeOfDataArr;cout.flush();

			printf("\nTable construct time: %f", tableconstuctend - tableconstuctstart);

			//add the batched result set size to the total count
			totalResultsLoop+=cnt[tid];


			printf("\nRunning total of total size of result array, tid: %d: %u", tid, totalResultsLoop);
			//}





		

		} //END LOOP OVER THE GPU BATCHES  


	printf("\nTOTAL RESULT SET SIZE ON HOST:  %d", totalResultsLoop);
	*totalNeighbors=totalResultsLoop;
	

	double tKernelResultsEnd=omp_get_wtime();
	
	printf("\nTime to launch kernel and execute all of the previous part of the method and get the results back: %f",tKernelResultsEnd-tKernelResultsStart);



	

	///////////////////////////////
	//END EXECUTE KERNEL
	///////////////////////////////

    ///////////////////////////////////
	//GET RESULT SET
	///////////////////////////////////

	//first find the size of the number of results
	//errCode=cudaMemcpy( cnt, dev_cnt, sizeof(unsigned int), cudaMemcpyDeviceToHost );
	// if(errCode != cudaSuccess) {
	// cout << "\nError: getting cnt from GPU Got error with code " << errCode << endl; 
	// }
	// else
	// {
	// 	printf("\nGPU: result set size on within epsilon: %d",*cnt);
	// }


	/*
	errCode=cudaMemcpy( cnt, dev_cnt, sizeof(unsigned int), cudaMemcpyDeviceToHost );
	if(errCode != cudaSuccess) {
	cout << "\nError: getting cnt from GPU Got error with code " << errCode << endl; 
	}
	else
	{
		printf("\nGPU: result set size on GPU within epsilon (from precomputed table method): %d",*cnt);
	}

	*/

	
	//copy the results, but only transfer the number of results, not the entire buffer
	// errCode=cudaMemcpy(results, dev_results, sizeof(struct structresults)*(*cnt), cudaMemcpyDeviceToHost );
	// if(errCode != cudaSuccess) {
	// cout << "\nError: getting results from GPU (from the precomputed table) Got error with code " << errCode << endl; 
	// }

	//*totalNeighbors=(*cnt);

	//SORTING FOR TESTING ONLY
	//XXXXXXXXX
	//XXXXXXXXX
	// std::sort(results, results+(*cnt),compResults);
	// printf("\n**** GPU\n");
	// for (int i=0; i<(*cnt); i++)
	// {
	// 	printf("\n%d,%d",results[i].pointID, results[i].pointInDist);
	// }

	//XXXXXXXXX
	//XXXXXXXXX
	//XXXXXXXXX



	///////////////////////////////////
	//END GET RESULT SET
	///////////////////////////////////

	double tFreeStart=omp_get_wtime();

	/////////////////
	//FREE
	/////////////////

	for (int i=0; i<GPUSTREAMS; i++)
	{
		errCode=cudaStreamDestroy(stream[i]);

		if(errCode != cudaSuccess) {
		cout << "\nError: destroying stream" << errCode << endl; 
		}
	}


	cudaFree(dev_directNeighborArray);
	cudaFree(dev_gpuLookupArray);
	//cudaFree(dev_results);
	cudaFree(dev_database);
	cudaFree(dev_epsilon);
	cudaFree(dev_N);
	cudaFree(dev_cnt);
	cudaFree(dev_offset);
	cudaFree(dev_batchNumber);

	//free data related to the individual streams for each batch
	for (int i=0; i<GPUSTREAMS; i++)
	{
		//free the data on the device
		cudaFree(dev_pointIDKey[i]);
		cudaFree(dev_pointInDistValue[i]);

		//free on the host
		cudaFreeHost(pointIDKey[i]);
		cudaFreeHost(pointInDistValue[i]);

	

	}

	
	cudaFree(dev_pointIDKey);
	cudaFree(dev_pointInDistValue);
	//free pinned memory on host
	cudaFreeHost(pointIDKey);
	cudaFreeHost(pointInDistValue);

		


	double tFreeEnd=omp_get_wtime();

	printf("\nTime freeing memory: %f", tFreeEnd - tFreeStart);

	cout<<"\n** last error at end of fn construct table batches: "<<cudaGetLastError();


	return true;

}
















/*

//METHOD TO COPY THE DATABASE TO THE GPU:
//takes as input: 
//the imported points, but which include extraneous information (tec, time)
//a pointer to the database on the GPU
void copyDatabaseToGPU(std::vector<struct dataElem> * dataPoints, struct point * dev_database)
{

	//CUDA error code:
	cudaError_t errCode;

	//
	unsigned int N=dataPoints->size();
	struct point * database;
	database=(struct point*)malloc(sizeof(struct point)*N);
	dev_database=(struct point*)malloc(sizeof(struct point)*N);

	//allocate memory on device:
	
	errCode=cudaMalloc( (void**)&dev_database, sizeof(struct point)*N );
	if(errCode != cudaSuccess) {
	cout << "\nError: database Got error with code " << errCode << endl; //2 means not enough memory
	}


	//first, we copy the x and y values from dataPoints to the database
	for (int i=0; i<N; i++)
	{
		database[i].x=(*dataPoints)[i].x;
		database[i].y=(*dataPoints)[i].y;
	}

	//printf("\n size of database: %d",N);

	//copy database to the device:
	errCode=cudaMemcpy(dev_database, database, sizeof(struct point)*N, cudaMemcpyHostToDevice);	
	if(errCode != cudaSuccess) {
	cout << "\nError: database Got error with code " << errCode << endl; //2 means not enough memory
	}

}


//METHOD TO SET THE KERNEL PARAMETERS
void setKernelParams(unsigned int * dev_N, unsigned int * N, unsigned int  * dev_debug1, unsigned int  * dev_debug2, unsigned int *dev_cnt, double * dev_epsilon, double * epsilon)
{

	//CUDA error code:
	cudaError_t errCode;
	
	//count values
	unsigned int * cnt;
	cnt=(unsigned int*)malloc(sizeof(unsigned int));
	*cnt=0;

	//unsigned int *dev_cnt;
	dev_cnt=(unsigned int*)malloc(sizeof(unsigned int));
	*dev_cnt=0;

	//printf("\ndev cnt in fn: %u",*dev_cnt);cout.flush();


	//allocate on the device
	errCode=cudaMalloc((unsigned int**)&dev_cnt, sizeof(unsigned int));	
	if(errCode != cudaSuccess) {
	cout << "\nError: cnt Got error with code " << errCode << endl; //2 means not enough memory
	}


	

	//double * dev_epsilon;
	dev_epsilon=(double*)malloc(sizeof( double ));
	*dev_epsilon=*epsilon;

	//allocate on the device
	errCode=cudaMalloc((void**)&dev_epsilon, sizeof(double));
	if(errCode != cudaSuccess) {
	cout << "\nError: epsilon Got error with code " << errCode << endl; //2 means not enough memory
	}

	
	//size of the database:
	dev_N=(unsigned int*)malloc(sizeof( unsigned int ));
	//*dev_N=N;

	//allocate on the device
	errCode=cudaMalloc((void**)&dev_N, sizeof(unsigned int));
	if(errCode != cudaSuccess) {
	cout << "\nError: N Got error with code " << errCode << endl; //2 means not enough memory
	}	

	//debug values
	//unsigned int  * dev_debug1;
	//unsigned int  * dev_debug2;
	dev_debug1=(unsigned int *)malloc(sizeof(unsigned int ));
	*dev_debug1=0;
	dev_debug2=(unsigned int *)malloc(sizeof(unsigned int ));
	*dev_debug2=0;

	//allocate on the device
	errCode=cudaMalloc( (unsigned int **)&dev_debug1, sizeof(unsigned int ) );
	if(errCode != cudaSuccess) {
	cout << "\nError: debug1 Got error with code " << errCode << endl; //2 means not enough memory
	}		
	errCode=cudaMalloc( (unsigned int **)&dev_debug2, sizeof(unsigned int ) );
	if(errCode != cudaSuccess) {
	cout << "\nError: debug2 Got error with code " << errCode << endl; //2 means not enough memory
	}		


	//copy N, epsilon and cnt to the device
	//epsilon
	errCode=cudaMemcpy( dev_epsilon, epsilon, sizeof(double), cudaMemcpyHostToDevice );
	if(errCode != cudaSuccess) {
	cout << "\nError: epsilon Got error with code " << errCode << endl; //2 means not enough memory
	}		

	//cnt
	// errCode=cudaMemcpy( dev_cnt, cnt, sizeof(unsigned int), cudaMemcpyHostToDevice );
	// if(errCode != cudaSuccess) {
	// cout << "\nError: cnt Got error with code " << errCode << endl; //2 means not enough memory
	// }		

	//N
	errCode=cudaMemcpy( dev_N, N, sizeof(unsigned int), cudaMemcpyHostToDevice );
	if(errCode != cudaSuccess) {
	cout << "\nError: N Got error with code " << errCode << endl; //2 means not enough memory
	}		

	//printf("\nnumber of elements: %u,%u",*dev_N,N);




}


void allocateResultSet(struct structresults * dev_results, struct structresults * results)
{
	
//dev_results=(struct structresults*)malloc(sizeof(struct structresults)*BUFFERELEM);
cudaError_t errCode=cudaMalloc((void **)&dev_results, sizeof(struct structresults)*BUFFERELEM);
if(errCode != cudaSuccess) {
cout << "CUDA: Got error with code " << errCode << endl; //2 means not enough memory
}

printf("\nmemory requested for results (GiB): %f",(double)(sizeof(struct structresults)*BUFFERELEM)/(1024*1024*1024));

//host result allocation:
results=(struct structresults*)malloc(sizeof(struct structresults)*BUFFERELEM);

}

*/

bool compResults(structresults const& lhs, structresults const& rhs) {
    if (lhs.pointID != rhs.pointID)
        return (lhs.pointID < rhs.pointID);
    if (lhs.pointInDist != rhs.pointInDist)
    {
        return (lhs.pointInDist < rhs.pointInDist);
    }
    return (lhs.pointInDist > rhs.pointInDist); 
}



