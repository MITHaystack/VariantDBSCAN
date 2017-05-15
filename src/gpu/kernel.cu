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


#include "kernel.h"
#include "structs.h"
#include <math.h>	

/////////////////////////////////////////
//THE RESULTS GET GENERATED AS KEY/VALUE PAIRS IN TWO ARRAYS
//KEY- THE POINT ID BEING SEARCHED
//VALUE- A POINT ID WITHIN EPSILON OF THE KEY POINT THAT WAS SEARCHED
//THE RESULTS ARE SORTED IN SITU ON THE DEVICE BY THRUST AFTER THE KERNEL FINISHES
/////////////////////////////////////////


// __global__ void testkernel(unsigned int * cnt) {
// unsigned int tid=threadIdx.x+ (blockIdx.x*BLOCKSIZE); 
// if (tid>=N2)
// 	return;
	
// unsigned int idx=atomicAdd(cnt,int(1));	
// return;
// }


//this kernel takes as input a direct neighbors table for a greater epsilon value and then calculates the table for a 
//smaller epsilon value 
//N is the total nunber of data points
//lookup is a lookup table that points to indices in the directNeighborArray
//Thats tells where the location of the direct neighbours for each data point are located.
//epsilon is the smaller epsilon value from the lookup table
__global__ void calcNeighborsFromTableKernel(unsigned int *N, struct gpulookuptable * lookup, int * directNeighborArray, unsigned int * cnt, double * epsilon, struct point * database, int * pointIDKey, int * pointInDistVal)
{
unsigned int tid=threadIdx.x+ (blockIdx.x*BLOCKSIZE); 



if (tid>=*N){
	return;
}




// if (tid<*N){
// 	//*cnt=*N;
// 	//*cnt=0;
// 	unsigned int idx=atomicAdd(cnt,int(1));
// }

// return;
//unsigned int idx=atomicAdd(cnt,int(1));



int indexmin=lookup[tid].indexmin;
int indexmax=lookup[tid].indexmax;
double pntX=database[tid].x;
double pntY=database[tid].y;

//uses purely global memory, no optimization here.
for (int i=indexmin; i<=indexmax; i++)
{

	int index=directNeighborArray[i];


	if (sqrt(((pntX-database[index].x)*(pntX-database[index].x))+((pntY-database[index].y)*(pntY-database[index].y)))<=*epsilon)
				{	
					unsigned int idx=atomicAdd(cnt,int(1));
					// results[idx].pointID=tid;
					// results[idx].pointInDist=index;
					pointIDKey[idx]=tid;
					pointInDistVal[idx]=index;


				}		
}





}







//this kernel takes as input a direct neighbors table for a greater epsilon value and then calculates the table for a 
//smaller epsilon value 
//N is the total nunber of data points
//lookup is a lookup table that points to indices in the directNeighborArray
//Thats tells where the location of the direct neighbours for each data point are located.
//epsilon is the smaller epsilon value from the lookup table
__global__ void calcNeighborsFromTableKernelBatches(unsigned int *N, unsigned int *offset, unsigned int *batchNum, struct gpulookuptable * lookup, int * directNeighborArray, unsigned int * cnt, double * epsilon, struct point * database, int * pointIDKey, int * pointInDistVal)
{
unsigned int tid=threadIdx.x+ (blockIdx.x*BLOCKSIZE); 



if (tid>=*N){
	return;
}

unsigned int t_elemID=tid*(*offset)+(*batchNum); //strided across the dataset for more consistent batch return sizes


// if (tid<*N){
// 	//*cnt=*N;
// 	//*cnt=0;
// 	unsigned int idx=atomicAdd(cnt,int(1));
// }

// return;
//unsigned int idx=atomicAdd(cnt,int(1));



int indexmin=lookup[t_elemID].indexmin;
int indexmax=lookup[t_elemID].indexmax;
double pntX=database[t_elemID].x;
double pntY=database[t_elemID].y;

//uses purely global memory, no optimization here.
for (int i=indexmin; i<=indexmax; i++)
{

	int index=directNeighborArray[i];


	if (sqrt(((pntX-database[index].x)*(pntX-database[index].x))+((pntY-database[index].y)*(pntY-database[index].y)))<=*epsilon)
				{	
					unsigned int idx=atomicAdd(cnt,int(1));
					// results[idx].pointID=tid;
					// results[idx].pointInDist=index;
					pointIDKey[idx]=t_elemID;
					pointInDistVal[idx]=index;


				}		
}





}








//kernel with grid to generate the neighbor table for each point in the database
//Each grid cell is assigned to a single block. 
//The threads in each block page the data points to shared memory
//DATA AWARE
//We first see the maximum amount of shared memory required for a single block
//and pass it in to store the overlapping points in adjacent grid cells
__global__ void kernelGridIndexSMBlockDataAware(unsigned int *numThreads, unsigned int *N, unsigned int *debug1, unsigned int *debug2, double *epsilon, struct grid * index, double * gridMin_x, double * gridMin_y, int * gridNumXCells, int * gridNumYCells, int * lookupArr, unsigned int * cnt, struct point * database, const unsigned int * sharedMemElemSize, int * pointIDKey, int * pointInDistVal)
{

unsigned int tid=threadIdx.x+ (blockIdx.x*BLOCKSIZE); 

if (tid>=*numThreads){
	return;
}

//if the cell is empty, then we return, since each cell is processed by a block
if (index[blockIdx.x].indexmin==-1)
{
	return;
}

// *debug1=*sharedMemElemSize;





//stores in shared memory the data points of the originating cells
__shared__ double xOriginCell[BLOCKSIZE];
__shared__ double yOriginCell[BLOCKSIZE];


//testing:
//pool of shared memory for 3 arrays
extern __shared__ double s[];

//using the pool of shared memory defined by s above for the 3 shared memory arrays
//the amount of memory is passed in through the kernel as the 3rd parameter
const int offset1=(*sharedMemElemSize);
const int offset2=2*(*sharedMemElemSize);



double *xAdjacentCell=s;
double *yAdjacentCell=&s[offset1];
int *idPointAdjacentCell=(int *)&s[offset2];




// const unsigned int SMsizeAdjData=*sharedMemElemSize;
// __shared__ double xAdjacentCell[SMsizeAdjData];
// __shared__ double yAdjacentCell[SMsizeAdjData];
// __shared__ int idPointAdjacentCell[SMsizeAdjData];




//the number of adjacent cells with data in them
__shared__ int cellCnt;	
//the maximum number of grid cells around a grid cell is 9 because it's constrained by epsilon
__shared__ int GridCellIDs[9];	



//only one thread calculates the 1D cell IDs of neighboring cells
//this is because each grid cell is assigned to a single block
if (threadIdx.x==0)
{

	//initialize the cell count to 0
	cellCnt=0;

/////////////////////////////
//Calculate the linear ids of the adjacent grid cells of the CELL
//The linearized CELL ID is the block ID
//only those that have points of them are used

	//copy the number of x and y grid cells to registers
	int reg_gridNumXCells=*gridNumXCells;
	int reg_gridNumYCells=*gridNumYCells;	
	

	//int xCellID=(pntX-(*gridMin_x))/(*epsilon);
	//int yCellID=(pntY-(*gridMin_y))/(*epsilon);

	//the block IDx is the 1D linearized grid cell
	int xCellID=blockIdx.x%reg_gridNumXCells;
	int yCellID=blockIdx.x/reg_gridNumXCells;

	

	int minXCellID=0;
	int maxXCellID=0;
	int minYCellID=0;
	int maxYCellID=0;
	

	//calculate the min and max x and y cell ids by adding and subtracting one from each value
	//deal with exception cases below.

	minXCellID=max(0,xCellID-1);
	maxXCellID=min(xCellID+1,reg_gridNumXCells-1);
	minYCellID=max(0,yCellID-1);
	maxYCellID=min(yCellID+1,reg_gridNumYCells-1);

	
	//enumerate the cells in 2D, then convert into 1D
	//only store the cells that have data in them

	
	
	#pragma unroll
	for (int i=minYCellID; i<=maxYCellID; i++){
		#pragma unroll
		for (int j=minXCellID; j<=maxXCellID; j++){
			int linearID=(i*reg_gridNumXCells)+j;			
			
			if(index[linearID].indexmin!=-1) 
			{
				GridCellIDs[cellCnt]=linearID;
				cellCnt++;
			} 
		}
	}


} //end if statement
//End calculate the linear ids of the grid cells
///////////////////////////////


__syncthreads(); //synchronize the threads in the block. Only the first thread in each block has done any work so far



//first, page all of the data elements into shared memory of the adjacent grid cells (and the originating cell itself)
int tmpElemCnt=0;

for (int h=0; h<cellCnt; h++)
{

	int adjCellID=GridCellIDs[h];
	const int numElemInAdjacentCell=index[adjCellID].indexmax-index[adjCellID].indexmin+1;	
	for (int k=0; k<numElemInAdjacentCell; k+=BLOCKSIZE)
	{

		
		if(((k*BLOCKSIZE)+threadIdx.x)<numElemInAdjacentCell)
		{
					int adjDataId=lookupArr[index[adjCellID].indexmin+k+threadIdx.x];	
					int threadOffset=threadIdx.x+k+tmpElemCnt;
					xAdjacentCell[threadOffset]=database[adjDataId].x;
					yAdjacentCell[threadOffset]=database[adjDataId].y;
					idPointAdjacentCell[threadOffset]=adjDataId;
		}

	}

	tmpElemCnt+=numElemInAdjacentCell;
	
}	


__syncthreads(); //sync the threads so that all of the adjacent cell data is in shared memory


//now we compare the adjacent data to the origin cell and perform the distance calculation
const int numElemInOriginCell=index[blockIdx.x].indexmax-index[blockIdx.x].indexmin+1;
	
for (int i=0; i<numElemInOriginCell; i+=BLOCKSIZE)
{
		//manually page data into shared memory		
		//for the origin cell	
		int dataId=lookupArr[index[blockIdx.x].indexmin+i+threadIdx.x];

		
		if(((i*BLOCKSIZE)+threadIdx.x)<numElemInOriginCell)
		{
			xOriginCell[threadIdx.x]=database[dataId].x;
			yOriginCell[threadIdx.x]=database[dataId].y;		
		}

		if(((i*BLOCKSIZE)+threadIdx.x)<numElemInOriginCell)
		{
			//int iterations=min(BLOCKSIZE,(tmpElemCnt-(i*BLOCKSIZE)));
			for (int l=0; l<tmpElemCnt; l++)
			{
				//distance calculation:
				if (sqrt(((xOriginCell[threadIdx.x]-xAdjacentCell[l])*(xOriginCell[threadIdx.x]-xAdjacentCell[l]))+
				((yOriginCell[threadIdx.x]-yAdjacentCell[l])*(yOriginCell[threadIdx.x]-yAdjacentCell[l])))<=(*epsilon))
				{
				unsigned int idx=atomicAdd(cnt,int(1));
				// results[idx].pointID=dataId;
				// results[idx].pointInDist=idPointAdjacentCell[l];
				pointIDKey[idx]=dataId;
				pointInDistVal[idx]=idPointAdjacentCell[l];
				}								
			}
		}

}		




} //end kernel












//kernel with grid to generate the neighbor table for each point in the database
//Each grid cell is assigned to a single block. 
//The threads in each block page the data points to shared memory
//DATA OBLIVIOUS
__global__ void kernelGridIndexSMBlock(unsigned int *numThreads, unsigned int *N, unsigned int *debug1, unsigned int *debug2, double *epsilon, struct grid * index, double * gridMin_x, double * gridMin_y, int * gridNumXCells, int * gridNumYCells, int * lookupArr, unsigned int * cnt, struct point * database, unsigned int * schedule, int * pointIDKey, int * pointInDistVal)
{

unsigned int tid=threadIdx.x+ (blockIdx.x*BLOCKSIZE); 


if (tid>=*numThreads){
	return;
}

//if the cell is empty, then we return, since each cell is processed by a block
// if (index[blockIdx.x].indexmin==-1)
// {
// 	return;
// }


//consult the schedule to find out what grid cell to process:
int cellToProcess=schedule[blockIdx.x];


////////////////////////
//DEBUG
// if (tid==0)
// {
// 	*cnt=*numThreads;
// }

// if (threadIdx.x==0)
// {
// 	atomicAdd(cnt,int(1));
// }
////////////////////////

//the number of adjacent cells with data in them
__shared__ int cellCnt;	
//the maximum number of grid cells around a grid cell is 9 because it's constrained by epsilon
__shared__ int GridCellIDs[9];	

//the number of data items in the originating cell
__shared__ int numElemInOriginCell;


//stores in shared memory the data points of the originating cells
__shared__ double xOriginCell[BLOCKSIZE];
__shared__ double yOriginCell[BLOCKSIZE];

//stores in shared memory the data points of the adjacent cells
__shared__ double xAdjacentCell[BLOCKSIZE];
__shared__ double yAdjacentCell[BLOCKSIZE];
__shared__ int idPointAdjacentCell[BLOCKSIZE];



//only one thread calculates the 1D cell IDs of neighboring cells
//this is because each grid cell is assigned to a single block
if (threadIdx.x==0)
{

	//the number of data items in the originating cell
	numElemInOriginCell=index[cellToProcess].indexmax-index[cellToProcess].indexmin+1;

	//initialize the cell count to 0
	cellCnt=0;

/////////////////////////////
//Calculate the linear ids of the adjacent grid cells of the CELL
//The linearized CELL ID is the block ID
//only those that have points of them are used

	//copy the number of x and y grid cells to registers
	int reg_gridNumXCells=*gridNumXCells;
	int reg_gridNumYCells=*gridNumYCells;	
	

	//int xCellID=(pntX-(*gridMin_x))/(*epsilon);
	//int yCellID=(pntY-(*gridMin_y))/(*epsilon);

	//the block IDx is the 1D linearized grid cell
	int xCellID=cellToProcess%reg_gridNumXCells;
	int yCellID=cellToProcess/reg_gridNumXCells;

	

	int minXCellID=0;
	int maxXCellID=0;
	int minYCellID=0;
	int maxYCellID=0;
	

	//calculate the min and max x and y cell ids by adding and subtracting one from each value
	//deal with exception cases below.

	minXCellID=max(0,xCellID-1);
	maxXCellID=min(xCellID+1,reg_gridNumXCells-1);
	minYCellID=max(0,yCellID-1);
	maxYCellID=min(yCellID+1,reg_gridNumYCells-1);

	
	//enumerate the cells in 2D, then convert into 1D
	//only store the cells that have data in them

	
	
	#pragma unroll
	for (int i=minYCellID; i<=maxYCellID; i++){
		#pragma unroll
		for (int j=minXCellID; j<=maxXCellID; j++){
			int linearID=(i*reg_gridNumXCells)+j;			
			
			if(index[linearID].indexmin!=-1) 
			{
				GridCellIDs[cellCnt]=linearID;
				cellCnt++;
			} 
		}
	}


} //end if statement
//End calculate the linear ids of the grid cells
///////////////////////////////


__syncthreads(); //synchronize the threads in the block. Only the first thread in each block has done any work so far



//loop over each adjacent cell, including the originating cell itself
for (int h=0; h<cellCnt; h++)
{


	 
	//manually page the data of the originating cell into shared memory
	

	for (int i=0; i<numElemInOriginCell; i+=BLOCKSIZE)
	{
			
			
			//manually page data into shared memory		
			//for the origin cell	
			int dataId=lookupArr[index[cellToProcess].indexmin+i+threadIdx.x];

			//if(((i*BLOCKSIZE)+threadIdx.x)<numElemInOriginCell) //CHANGED THIS TO BELOW
			if((i+threadIdx.x)<numElemInOriginCell)
			{
				xOriginCell[threadIdx.x]=database[dataId].x;
				yOriginCell[threadIdx.x]=database[dataId].y;		
			}

			
			//now we page the data of the adjacent cell and perform the distance calculation:

			int adjCellID=GridCellIDs[h];
			const int numElemInAdjacentCell=index[adjCellID].indexmax-index[adjCellID].indexmin+1;


			//do we need a sync threads here? YES
						

			for (int k=0; k<numElemInAdjacentCell; k+=BLOCKSIZE)
			{
				
				

				__syncthreads(); //this one is required


				//Page in data of the adjacent cell:
				//make sure that threads dont page in data that they shouldn't
				//if(((k*BLOCKSIZE)+threadIdx.x)<numElemInAdjacentCell) //CHANGED THIS BELOW
				if((k+threadIdx.x)<numElemInAdjacentCell)
				{
					int adjDataId=lookupArr[index[adjCellID].indexmin+k+threadIdx.x];	
					xAdjacentCell[threadIdx.x]=database[adjDataId].x;
					yAdjacentCell[threadIdx.x]=database[adjDataId].y;
					idPointAdjacentCell[threadIdx.x]=adjDataId;
				}
				

				//do we need a sync threads here? YES
				__syncthreads();

				//the shared memory contains the points of the originating cell and the adjacent cell.
				//each thread processes a single originating cell data point and loops over all of the
				//data points in the adjacent cell

				//make sure that threads dont try to access data out of bounds
				
				//if(((i*BLOCKSIZE)+threadIdx.x)<numElemInOriginCell) //CHANGED THIS TO BELOW
				if((i+threadIdx.x)<numElemInOriginCell) 
				{
					//int iterations=min(BLOCKSIZE,(numElemInAdjacentCell-(k*BLOCKSIZE))); //CHANGED THIS TO BELOW
					int iterations=min(BLOCKSIZE,(numElemInAdjacentCell-k));


					for (int l=0; l<iterations; l++)
					{
						//distance calculation:
						if (sqrt(((xOriginCell[threadIdx.x]-xAdjacentCell[l])*(xOriginCell[threadIdx.x]-xAdjacentCell[l]))+
						((yOriginCell[threadIdx.x]-yAdjacentCell[l])*(yOriginCell[threadIdx.x]-yAdjacentCell[l])))<=(*epsilon))
						{
						unsigned int idx=atomicAdd(cnt,int(1));
						// results[idx].pointID=dataId;
						// results[idx].pointInDist=idPointAdjacentCell[l];
						pointIDKey[idx]=dataId;
						pointInDistVal[idx]=idPointAdjacentCell[l];
						}								
					}
				}
				



			

			} //end adjacent cell paging and distance calculation

			

	}


}//end of outer loop that loops over all of the cells


} //end kernel

















//kernel with grid to generate the neighbor table for each point in the database
//the total number of threads are the number of data points
//each point finds its own surrounding grid cells and compares
//NO SHARED MEMORY

//N-the number of threads
//epislon- the epsilon value to search for between pairs of points
//index- the grid index
//gridMin_x- the minimum x value of the grid
//gridMin_y- the minimum y value of the grid
//gridNumXCells- the number of "x" cells (rows)
//gridNumYCells- the number of "y" cells (columns)
//lookupArr- a look up array that maps the range of ids in the index struct to the data points (database)
//database- the data base of points (their positions)
//offset- an offset into the database of points. Used when batching. Leave it when not batching as well, the offset will be 0.

//produces:
//cnt- global counter of the number of the total number of pairs of points within the the epsilon distance from each other
//results are as key value pairs, where the key is the point being searched, and value is another point within the distance
//pointIDKey- a point within the distance of a point in the pointInDistVal array
//pointInDistVal- a point within the distance of pointIDKey
__global__ void kernelGridIndex(unsigned int *N, unsigned int *offset, unsigned int *batchNum, unsigned int *debug1, unsigned int *debug2, double *epsilon, struct grid * index, double * gridMin_x, double * gridMin_y, int * gridNumXCells, int * gridNumYCells, int * lookupArr, unsigned int * cnt, struct point * database, int * pointIDKey, int * pointInDistVal)
{

unsigned int tid=threadIdx.x+ (blockIdx.x*BLOCKSIZE); 

if (tid>=*N){
	return;
}


//unsigned int t_elemID=tid+(*offset); //original way of batching
unsigned int t_elemID=tid*(*offset)+(*batchNum); //strided across the dataset for more consistent batch return sizes


//copy the x and y positions into registers
	double pntX=database[t_elemID].x;
	double pntY=database[t_elemID].y;

/////////////////////////////
//Calculate the linear ids of the adjacent grid cells of the point
//only those that have points of them are used

	//copy the number of x and y grid cells to registers
	int reg_gridNumXCells=*gridNumXCells;
	int reg_gridNumYCells=*gridNumYCells;	
	

	int xCellID=(pntX-(*gridMin_x))/(*epsilon);
	int yCellID=(pntY-(*gridMin_y))/(*epsilon);

	int GridCellIDs[9];	

	int minXCellID=0;
	int maxXCellID=0;
	int minYCellID=0;
	int maxYCellID=0;
	

	//calculate the min and max x and y cell ids by adding and subtracting one from each value
	//deal with exception cases below.

	minXCellID=max(0,xCellID-1);
	maxXCellID=min(xCellID+1,reg_gridNumXCells-1);
	minYCellID=max(0,yCellID-1);
	maxYCellID=min(yCellID+1,reg_gridNumYCells-1);

	
	//enumerate the cells in 2D, then convert into 1D
	//only store the cells that have data in them
	//the most number of cells that can be found with data is 9 because the grid is constrained by epsilon

	
	int cellCnt=0;	
	#pragma unroll
	for (int i=minYCellID; i<=maxYCellID; i++){
		#pragma unroll
		for (int j=minXCellID; j<=maxXCellID; j++){
			int linearID=(i*reg_gridNumXCells)+j;			
			
			if(index[linearID].indexmin!=-1) 
			{
				GridCellIDs[cellCnt]=linearID;
				cellCnt++;
			} 
		}
	}



//End calculate the linear ids of the grid cells
///////////////////////////////


//iterate over the grid cells with points in them 
for (int j=0; j<cellCnt; j++){
		
		int gridID=GridCellIDs[j];

			
			for (int k=index[gridID].indexmin; k<=index[gridID].indexmax; k++)
			{

				int elemid=lookupArr[k];
				double x2=database[elemid].x;
				double y2=database[elemid].y;	
				if (sqrt(((pntX-x2)*(pntX-x2))+((pntY-y2)*(pntY-y2)))<=(*epsilon))
					{
						unsigned int idx=atomicAdd(cnt,int(1));
						pointIDKey[idx]=t_elemID;
						pointInDistVal[idx]=elemid;
					}


			}

		}	



}



//BATCH ESTIMATOR KERNEL: DOESNT STORE THE RESULTS OR OFFSET
//runs once
//returns cnt
//Samples the total dataset using sampleOffset, so that we don't process the entire database
__global__ void kernelGridIndexBatchEstimator(unsigned int *N, unsigned int *sampleOffset, unsigned int *debug1, unsigned int *debug2, double *epsilon, struct grid * index, double * gridMin_x, double * gridMin_y, int * gridNumXCells, int * gridNumYCells, int * lookupArr, unsigned int * cnt, struct point * database)
{

unsigned int tid=threadIdx.x+ (blockIdx.x*BLOCKSIZE); 

if (tid>=*N){
	return;
}



//copy the x and y positions into registers
	double pntX=database[tid*(*sampleOffset)].x;
	double pntY=database[tid*(*sampleOffset)].y;

/////////////////////////////
//Calculate the linear ids of the adjacent grid cells of the point
//only those that have points of them are used

	//copy the number of x and y grid cells to registers
	int reg_gridNumXCells=*gridNumXCells;
	int reg_gridNumYCells=*gridNumYCells;	
	

	int xCellID=(pntX-(*gridMin_x))/(*epsilon);
	int yCellID=(pntY-(*gridMin_y))/(*epsilon);

	int GridCellIDs[9];	

	int minXCellID=0;
	int maxXCellID=0;
	int minYCellID=0;
	int maxYCellID=0;
	

	//calculate the min and max x and y cell ids by adding and subtracting one from each value
	//deal with exception cases below.

	minXCellID=max(0,xCellID-1);
	maxXCellID=min(xCellID+1,reg_gridNumXCells-1);
	minYCellID=max(0,yCellID-1);
	maxYCellID=min(yCellID+1,reg_gridNumYCells-1);

	
	//enumerate the cells in 2D, then convert into 1D
	//only store the cells that have data in them
	//the most number of cells that can be found with data is 9 because the grid is constrained by epsilon

	
	int cellCnt=0;	
	#pragma unroll
	for (int i=minYCellID; i<=maxYCellID; i++){
		#pragma unroll
		for (int j=minXCellID; j<=maxXCellID; j++){
			int linearID=(i*reg_gridNumXCells)+j;			
			
			if(index[linearID].indexmin!=-1) 
			{
				GridCellIDs[cellCnt]=linearID;
				cellCnt++;
			} 
		}
	}



//End calculate the linear ids of the grid cells
///////////////////////////////


//iterate over the grid cells with points in them 
for (int j=0; j<cellCnt; j++){
		
		int gridID=GridCellIDs[j];

			
			for (int k=index[gridID].indexmin; k<=index[gridID].indexmax; k++)
			{

				int elemid=lookupArr[k];
				double x2=database[elemid].x;
				double y2=database[elemid].y;	
				if (sqrt(((pntX-x2)*(pntX-x2))+((pntY-y2)*(pntY-y2)))<=(*epsilon))
					{
						unsigned int idx=atomicAdd(cnt,int(1));
					}


			}

		}	



}




//kernel with grid to generate the neighbor table for each point in the database
//the total number of threads are the number of data points
//each point finds its own surrounding grid cells and compares
//NO SHARED MEMORY

/////////////////
//FOR TESTING
//////////////////
/*
__global__ void kernelGridIndexKeyVal(unsigned int *N, unsigned int *debug1, unsigned int *debug2, double *epsilon, struct grid * index, double * gridMin_x, double * gridMin_y, int * gridNumXCells, int * gridNumYCells, int * lookupArr, unsigned int * cnt, struct point * database, int * pointIDKey, int * pointInDistVal)
{

unsigned int tid=threadIdx.x+ (blockIdx.x*BLOCKSIZE); 

if (tid>=*N){
	return;
}



//copy the x and y positions into registers
	double pntX=database[tid].x;
	double pntY=database[tid].y;

/////////////////////////////
//Calculate the linear ids of the adjacent grid cells of the point
//only those that have points of them are used

	//copy the number of x and y grid cells to registers
	int reg_gridNumXCells=*gridNumXCells;
	int reg_gridNumYCells=*gridNumYCells;	
	

	int xCellID=(pntX-(*gridMin_x))/(*epsilon);
	int yCellID=(pntY-(*gridMin_y))/(*epsilon);

	int GridCellIDs[9];	

	int minXCellID=0;
	int maxXCellID=0;
	int minYCellID=0;
	int maxYCellID=0;
	

	//calculate the min and max x and y cell ids by adding and subtracting one from each value
	//deal with exception cases below.

	minXCellID=max(0,xCellID-1);
	maxXCellID=min(xCellID+1,reg_gridNumXCells-1);
	minYCellID=max(0,yCellID-1);
	maxYCellID=min(yCellID+1,reg_gridNumYCells-1);

	
	//enumerate the cells in 2D, then convert into 1D
	//only store the cells that have data in them
	//the most number of cells that can be found with data is 9 because the grid is constrained by epsilon

	
	int cellCnt=0;	
	#pragma unroll
	for (int i=minYCellID; i<=maxYCellID; i++){
		#pragma unroll
		for (int j=minXCellID; j<=maxXCellID; j++){
			int linearID=(i*reg_gridNumXCells)+j;			
			
			if(index[linearID].indexmin!=-1) 
			{
				GridCellIDs[cellCnt]=linearID;
				cellCnt++;
			} 
		}
	}




//End calculate the linear ids of the grid cells
///////////////////////////////


//iterate over the grid cells with points in them 
for (int j=0; j<cellCnt; j++){
		
		int gridID=GridCellIDs[j];

			
			for (int k=index[gridID].indexmin; k<=index[gridID].indexmax; k++)
			{

				int elemid=lookupArr[k];
				double x2=database[elemid].x;
				double y2=database[elemid].y;	
				if (sqrt(((pntX-x2)*(pntX-x2))+((pntY-y2)*(pntY-y2)))<=(*epsilon))
					{
						unsigned int idx=atomicAdd(cnt,int(1));
						// results[idx].pointID=tid;
						// results[idx].pointInDist=elemid;
						
						pointIDKey[idx]=tid;
						pointInDistVal[idx]=elemid;
					}


			}

		}	



}


*/




//Kernel brute forces to generate the neighbor table for each point in the database
__global__ void kernelBruteForce(unsigned int *N, unsigned int *debug1, unsigned int *debug2, double *epsilon, unsigned int * cnt, struct point * database, int * pointIDKey, int * pointInDistVal) {

unsigned int tid=threadIdx.x+ (blockIdx.x*BLOCKSIZE); 

if (tid>=*N){
	return;
}


// if (tid==0)
// {
// 	*debug1=555;
// }

double pntX=database[tid].x;
double pntY=database[tid].y;

//original, only use global memory
// for (int i=0; i<N; i++)
// {
// 	if (sqrt(((pntX-a[i])*(pntX-a[i]))+((pntY-b[i])*(pntY-b[i])))<=*epsilon)
// 	{
		
// 		atomicAdd(cnt,int(1));
// 	}	
// }


__shared__ double x[BLOCKSIZE];
__shared__ double y[BLOCKSIZE];





//three seperate loops for the "LEFTOVERS" to avoid divergent branching
//The first two relate to those threads that make up full blocks that compare against each other
//and partial blocks, and those threads that make up a partial block that need to compare against
//the rest of the data
int iterations1=(*N/BLOCKSIZE)*BLOCKSIZE;
int iterations2=*N%BLOCKSIZE;

// if (tid==0)
// {
// *debug1=iterations1;
// *debug2=iterations2;
	
// }

__syncthreads();



//for those threads that are in full blocks
if (tid<iterations1)
	{

	//for (int i=0; i<1; i++)
	for (int i=0; i<iterations1; i+=BLOCKSIZE)
	{
		
		
		
		//manually page data into shared memory with coalescing
		int elemid=i+threadIdx.x;
		// x[threadIdx.x]=a[elemid];
		// y[threadIdx.x]=b[elemid];
		x[threadIdx.x]=database[elemid].x;
		y[threadIdx.x]=database[elemid].y;
		__syncthreads();
		
		//#pragma unroll
		for (int j=0; j<BLOCKSIZE; j++)
		{
				unsigned int idxx=atomicAdd(debug1,int(1));
				//global mem
				//if (sqrt(((pntX-a[i+j])*(pntX-a[i+j]))+((pntY-b[i+j])*(pntY-b[i+j])))<=*epsilon)
				//shared mem
				if (sqrt(((pntX-x[j])*(pntX-x[j]))+((pntY-y[j])*(pntY-y[j])))<=*epsilon)
				{	
					unsigned int idx=atomicAdd(cnt,int(1));
					//resultset[idx]=i+j; //
					//resultsetElem[idx]=tid; //
					// results[idx].pointID=tid;
					// results[idx].pointInDist=i+j;

					pointIDKey[idx]=tid;
					pointInDistVal[idx]=i+j;

				}		
			
		}
		__syncthreads();
	}


	//START LEFTOVERS (from the nice tiled calculations)

	//manually page data into shared memory with coalescing
	if (threadIdx.x<iterations2)
	{
	int elemid=iterations1+threadIdx.x;	
	// x[threadIdx.x]=a[elemid];
	// y[threadIdx.x]=b[elemid];
	x[threadIdx.x]=database[elemid].x;
	y[threadIdx.x]=database[elemid].y;
	}
	__syncthreads();

	//for (int j=0; j<iterations2; j++)
	//for (int j=iterations1; j<iterations1+iterations2; j++) //for global memory
	for (int j=0; j<iterations2; j++)
	{
			unsigned int idxx=atomicAdd(debug1,int(1));
			//access directly in global memory
			//if (sqrt(((pntX-a[j])*(pntX-a[j]))+((pntY-b[j])*(pntY-b[j])))<=*epsilon)
			
			if (sqrt(((pntX-x[j])*(pntX-x[j]))+((pntY-y[j])*(pntY-y[j])))<=*epsilon)
			{	
				unsigned int idx=atomicAdd(cnt,int(1));
				// results[idx].pointID=tid;
				// results[idx].pointInDist=iterations1+j;

				pointIDKey[idx]=tid;
				pointInDistVal[idx]=iterations1+j;
			}		

	}

	//END LEFTOVERS (for nice tiled calculations)

} //end of nice tiled calculations


//start of the "leftover" threads that don't make up a full block
// if (tid>=iterations1)
// {
// 	for (int j=0; j<N; j++)
// 	{
// 		if (sqrt(((pntX-a[j])*(pntX-a[j]))+((pntY-b[j])*(pntY-b[j])))<=*epsilon)
// 		{
// 			atomicAdd(cnt,int(1));
// 		}
// 	}
// }

//start of the "leftover" threads that don't make up a full block
//they have to compare themselves to the rest of the data

if (tid>=iterations1)
{
	
	for (int i=0; i<*N; i+=iterations2)
	{

	
		if (threadIdx.x<iterations2)
		{
		int elemid=i+threadIdx.x;	
		// x[threadIdx.x]=a[elemid];
		// y[threadIdx.x]=b[elemid];
		x[threadIdx.x]=database[elemid].x;
		y[threadIdx.x]=database[elemid].y;
		}
		__syncthreads();

		int nextIterations=min(iterations2,*N-i);

		for (int j=0; j<nextIterations; j++)
		{
			unsigned int idxx=atomicAdd(debug1,int(1));
			//if (sqrt(((pntX-a[j])*(pntX-a[j]))+((pntY-b[j])*(pntY-b[j])))<=*epsilon)
			if (sqrt(((pntX-x[j])*(pntX-x[j]))+((pntY-y[j])*(pntY-y[j])))<=*epsilon)
			{
				unsigned int idx=atomicAdd(cnt,int(1));
				// results[idx].pointID=tid;
				// results[idx].pointInDist=i+j;

				pointIDKey[idx]=tid;
				pointInDistVal[idx]=i+j;


			}
		}

		__syncthreads();

	} //end loop	

} //end of if


return;
}