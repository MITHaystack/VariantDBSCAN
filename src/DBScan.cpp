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
#include "prototypes.h"
#include "globals.h"
#include <fstream>
#include <vector>
#include <set>
#include "RTree.h"
#include "DBScan.h"
#include <omp.h>
#include <iostream>
#include <unistd.h>
#include <set>

//Not really useful but allows you to control the minimum number of points you want to reuse from a cluster
//Set to 100, but since the number of points is small it has very little effect on perfromance
const int NUMMINREUSEPTS=100;

//stores the temporary neighbour list in SEQUENTIAL DBScan. Made global, but only accessed by the DBScan class
//this is because of issues using the callback method for the R-tree inside in the DBScan class
std::vector<int> neighbourList;

//stores the temporary neighbour list in SEQUENTIAL DBScan, but that needs memory space for multiple vectors for 
//different threads. Made global, but only accessed by the DBScan class
//this is because of issues using the callback method for the R-tree inside in the DBScan class
//pass in the number of threads
//used in the function:  algDBScanParallel (although it isn't actually parallel)
std::vector<int> neighbourListParallel[NSEARCHTHREADS];


	

	//constructor for the implementation that reuses previous clustering results to short-circuit the next clustering experiment
	//input:
	//the data points
	//the query distance
	//the minimum number of points to form a cluster
	//a pointer to the R-tree index with multiple points per box
	//the lookup array for the MBBs for having multiple points per box
	//a pointer to the R-tree that's used to discern whether clusters can be reused from one experiment to the next - this only uses a single point per box
	DBScan::DBScan(std::vector<struct dataElem> *ptrData, double epsilon, int minimumPts, 
		RTree<int,double,2,float> *indexPtr, std::vector<std::vector<int> > *ptr_MPB_lookup, RTree<int,double,2,float> *highResIndex)
	{
		//set epsilon and minpts
		distance=epsilon;
		minPts=minimumPts;

		//pointer to the data
		dataPoints=ptrData;
		
		//pointer to the R-tree index (MPB)
		tree=indexPtr;

		//pointer to other R-tree index that has one point per box
		//used to determine if a cluster overlaps any other points to determine if clusters from one experiment can be reused in the next experiment
		treeHighResolution=highResIndex;

		//pointer to the MPB lookup array
		ptrMPBlookup=ptr_MPB_lookup;
		
		//initialize vector that keeps track of the points that have been visited
		initializeVisitedPoints((*ptrData).size());

		//initialize the vector that keeps track of the cluster assignment for the points
		initializeClusterIDs((*ptrData).size());

		//reserve initial space for the neighbourList:
		for (int i=0; i<NSEARCHTHREADS; i++){
		neighbourListParallel[i].reserve(10000);
		}

		//the number of clusters
		clusterCnt=0;

	}





int DBScan::getDBScanNumClusters()
{
	return clusterCnt;
}



//This method takes a series of points that form a cluster and puts an axis aligned MBB around them and returns the MBB
void DBScan::generateMBBAroundCluster(std::vector<int>*clusterPoints, double * MBB_min, double * MBB_max)
{
		int id=(*clusterPoints)[0];
		
		MBB_min[0]=(*dataPoints)[id].x;
		MBB_max[0]=(*dataPoints)[id].x;
		MBB_min[1]=(*dataPoints)[id].y;
		MBB_max[1]=(*dataPoints)[id].y;

		for (int i=0; i<clusterPoints->size(); i++){
				id=(*clusterPoints)[i];
				
				if (((*dataPoints)[id].x)<MBB_min[0]){
					MBB_min[0]=(*dataPoints)[id].x;
				}
				if (((*dataPoints)[id].x)>MBB_max[0]){
					MBB_max[0]=(*dataPoints)[id].x;
				}

				if (((*dataPoints)[id].y)<MBB_min[1]){
					MBB_min[1]=(*dataPoints)[id].y;
				}
				if (((*dataPoints)[id].y)>MBB_max[1]){
					MBB_max[1]=(*dataPoints)[id].y;
				}			
		}
	
}

//this method appends an MBB by epsilon
void DBScan::appendMBBByEpsilon(double * MBB_min, double * MBB_max, double eps)
{
	MBB_min[0]=MBB_min[0]-eps;
	MBB_max[0]=MBB_max[0]+eps;
	MBB_min[1]=MBB_min[1]-eps;
	MBB_max[1]=MBB_max[1]+eps;

}





//"stump" of DBScan, just the part that takes the epsilon neighbourhood of a point that is in a cluster and expands the epsilon neighbourhood
//setIDsInDist is accessible by all methods, so we don't need to pass it in.
void DBScan::DBScanParallelMPBSTUMP(int finishedInstanceID, bool * destroyedArr, std::vector<int>* candidatesToGrowFrom)
{	

				std::vector<int>neighborSet;	

				//assign the neighbor ids to the neighborSet, which may be expanded when searching
				//through the neighbors
				neighborSet=*candidatesToGrowFrom;
				
				//expand the cluster 	
				 while (neighborSet.size()!=0){
				 	//examine the point that's the last one in the neighbor set
				 	int pntID=neighborSet.back(); 
				 	
				 	//if this point has been visited before
				 	if (visited[pntID]==true){
				 	//remove the value from the list of neighbors to check 	
				 	neighborSet.pop_back();
					continue;
					}

					//mark the point as visited:
					visited[pntID]=true;

					//get the neighbors of the data point
					dataElem tmpDataPoint=(*dataPoints)[pntID];
					getNeighboursParallelMPB(&tmpDataPoint,distance, &setIDsInDist);

					//if the point was part of a cluster last experiment iteration, then we need to destroy this cluster
					int oldClusterID=instanceVect[finishedInstanceID]->clusterIDs[pntID];
					if (oldClusterID!=0){
						destroyedArr[oldClusterID]=true;
					}


					//if the number of neighbors is greater than the number required to form a cluster
					if (setIDsInDist.size()>=minPts){
						//assign the neighbor ids to the neighborSet
						copyVect(&neighborSet,&setIDsInDist);
					}
					//if the point has part not been assigned to a cluster yet
					if (clusterIDs[pntID]==0){
						clusterIDs[pntID]=clusterCnt; 							
					}
					

				} //end of while loop

}



//comparison function to sort vector
//of density struct
bool DBScan::comparedensityStructfn(const densityStruct &a, const densityStruct &b)
{
    return a.density > b.density;
}



void DBScan::scheduleSelector(std::vector<int>* schedule, std::vector<int> clusterArr[], int numClustersInOtherInstance)
{
	setClusterScheduleDensity(schedule, clusterArr, numClustersInOtherInstance);
}


//Sets the ordering of the clusters that are visited and attempted to be reused
//The highest density clusters are prioritized
void DBScan::setClusterScheduleDensity(std::vector<int>* schedule, std::vector<int> clusterArr[], int numClustersInOtherInstance)
{
	//order by cluster density:	
	std::vector<struct densityStruct> densityArr;

	int cnt=0;

	//Iterate over all predefined clusters and throw away the ones we don't want.
	//skip the noise cluster, we dont include it
	for (int i=1; i<numClustersInOtherInstance;i++){
		double MBB_min[2]; //MBB min
	  	double MBB_max[2]; //MBB max
		generateMBBAroundCluster(&clusterArr[i],MBB_min,MBB_max);
		appendMBBByEpsilon(MBB_min,MBB_max,distance);

	  	//If the number of points in the cluster is less than a certain cutoff, we dont use it
	  	//This basically has no effect on the overall performance in the datasets we tested
	  	if(clusterArr[i].size()<NUMMINREUSEPTS){
	  		continue;
	  	}
	  	//write to the density array if we dont throw out the cluster:
	  	else{
	  		densityStruct tmpstruct;
	  		tmpstruct.clusterID=i;
	  		tmpstruct.density=(1.0*clusterArr[i].size())/((MBB_max[0]-MBB_min[0])*(MBB_max[1]-MBB_min[1]));
	  		densityArr.push_back(tmpstruct);
	  		cnt++;	
	  	}

		

	}	

	//sort based on density:
	std::sort(densityArr.begin(), densityArr.end(),comparedensityStructfn);
	
	//set the schedule:
	for (int i=0; i<densityArr.size();i++){
		schedule->push_back(densityArr[i].clusterID);
	}

}	




void DBScan::assignPointsToPredefinedCluster(int finishedInstanceID)
{

	int numClustersInOtherInstance=instanceVect[finishedInstanceID]->getDBScanNumClusters();
	int cntreusedpnts=0;
	int cntClustersReused=0;

	//the counter for the total number of clusters
	clusterCnt=0;


	//generate array of vectors, one for each cluster that contains its point ids. 
	//Include the noise cluster to make the calculations straightforward 
	std::vector<int> clusterArr[numClustersInOtherInstance];
	//array that keeps track of clusters that have been destroyed through the proces
	bool destroyedArr[numClustersInOtherInstance];
	for (int i=0; i<numClustersInOtherInstance;i++){
		destroyedArr[i]=false;
	}

	//iterate over the finished cluster array. Store arrays of datapoint ids that belong to each cluster, including the noise cluster
	for (int i=0; i<dataPoints->size();i++){
		int clusterid=instanceVect[finishedInstanceID]->clusterIDs[i];		
		clusterArr[clusterid].push_back(i);
	}

	//For each cluster, sort the vector of data point ids, because we need to do a binary search later:
	//we skip cluster 0 which is the noise points (thats why it starts at 1 below)
	for (int i=1; i<numClustersInOtherInstance; i++){
	std::sort (clusterArr[i].begin(), clusterArr[i].end());
	}

	//We set a schedule for searching the clusters, because we want to select "good" clusters to reuse
	std::vector<int>schedule;
	
	//We use the greedy schedule only
	scheduleSelector(&schedule, clusterArr, numClustersInOtherInstance);
	
	//for each cluster, build an MBB around the cluster. Then we augment with the epsilon of the new variant and see if it intersects any other points	
	for (int i=1; i<schedule.size(); i++){
		//select the cluster based on the schedule
		int cID=schedule[i];
		//some clusters will be destroyed through the process when points from old clusters are reassigned to new clusters, so if we look at a destroyed cluster
		//we skip it
		if (destroyedArr[cID]){
			continue;
		}
		

		cntClustersReused++;

		//MBB min/max
		double MBB_min[2]; 
	  	double MBB_max[2]; 
		
	  	//generate MBB around the cluster
	  	generateMBBAroundCluster(&clusterArr[cID],MBB_min,MBB_max);
	  	//Append the MBB by epsilon
	  	appendMBBByEpsilon(MBB_min,MBB_max,distance);

	  	//Assign all of the values from the old cluster to a new cluster
	  	//and update their visited status
	  	clusterCnt++;

	  	
	  	//assign visited and the cluster id to all points copied from the last instance to this one
	  	for (int j=0; j<clusterArr[cID].size();j++){
				int indexPointinCluster=clusterArr[cID][j];	
				clusterIDs[indexPointinCluster]=clusterCnt;	
				visited[indexPointinCluster]=true;
				cntreusedpnts++;
			}

		//------------------------	
	  	//Now we query the high-resolution tree to find those points that may be candidates to be
	  	//DIRECTLY DENSITY REACHABLE from the predefined cluster
	  	//query the tree and see what results come back
	  	//send information to the callback function
	  	//------------------------
		
		char arg[500];		
		int index=omp_get_thread_num();		
		neighbourListParallel[index].clear();		
		sprintf(arg,"%d",index); 
		treeHighResolution->Search(MBB_min,MBB_max, DBSCANmySearchCallbackParallel, arg); 
		
		//------------------------
		//from these candidates we want to know which points in the predefined cluster they overlap, because we are going to "grow the cluster" from these points
		//the idea is that we don't search every individual point in the predfined cluster, because that would defeat the purpose of reusing cluster results
		//but we want to grow the cluster naturally from the points in the cluster that have directly density reachable candidate points
		//however, the list of candidates contains the cluster points themselves, so we need to exclude them
		//i.e., we want the candidates, but not the points that are already in the cluster
		//------------------------

		std::vector<int> candidatesNotInCluster;

		for (int j=0; j<neighbourListParallel[index].size();j++)
		{
			int findElem=neighbourListParallel[index][j];
			//if the value is not part of the predefined cluster, then it is a candidate to be in the cluster
			if (!(std::binary_search(clusterArr[cID].begin(), clusterArr[cID].end(), findElem)))
			{
				candidatesNotInCluster.push_back(findElem);				
			}						
		}	

		std::vector<int> candidatesToGrowFromInCluster;

		//for each candidate, we search the index tree, and find if any of the points belong to the cluster, if they do belong to the cluster
		//then these points in the cluster are where we're going to start growing the cluster from. We can think of these as seeding the neighborlist
		for (int j=0;j<candidatesNotInCluster.size();j++)
		{
			//get the neighbors of the candidate data point
			int elemIndex=candidatesNotInCluster[j];
			dataElem tmpDataPoint=(*dataPoints)[elemIndex];
			setIDsInDist.clear();
			//for each candidate, search the tree and find any of them that belong to the predefined cluster
			getNeighboursParallelMPB(&tmpDataPoint,distance, &setIDsInDist);
			for (int k=0; k<setIDsInDist.size();k++){
					
					int findElem=setIDsInDist[k];
					//if the value is in the predefined cluster, then we want to grow the cluster from this point
					if ((std::binary_search(clusterArr[cID].begin(), clusterArr[cID].end(), findElem)))
					{
						candidatesToGrowFromInCluster.push_back(findElem);
						//mark the point in the cluster as NOT visited, we already set it to visited, but this makes the alg. consistent later when we call DBScanParallelMPBSTUMP
						visited[findElem]=false;				
					}
			} //end of for loop that finds point ids within the cluster from the candidate

		}//end of for loop over all candidates not in the cluster		


		//candidatesToGrowFromInCluster contains the points inside the predefined cluster that we want to test to see if the cluster can be grown

		//grow the cluster using DBScan like normal
		DBScanParallelMPBSTUMP(finishedInstanceID, destroyedArr, &candidatesToGrowFromInCluster);


	} //end of main for loop

	//call DBSCAN on the remainder of the points
	algDBScanParallelReuseClusterResults();	
	
}









//This isn't really parallel-but allows for buffers from the r-tree and the nighbour functions etc
//Such that multiple dbscan objects can traverse the tree, get neighbours, etc.
//Use the output of one variant as input for another to see if we can reuse cluster results from smaller eps values.	
void DBScan::algDBScanParallelReuseClusterResults()
	{
		
		int tmpcnt=0;

		//neighborSet is the current set of points being searched that belongs to a cluster
		std::vector<int>neighborSet;


		for (int i=0; i<(*dataPoints).size(); i++){
			//see if the point has been visited, if so, go onto the next loop iteration
			if (visited[i]==true){
				continue;
			}

			//clear the vector of neighbors
			neighborSet.clear();

			//mark the point as visited:
			visited[i]=true;
			
			//get the neighbors of the data point
			dataElem tmpDataPoint=(*dataPoints)[i];
			
			getNeighboursParallelMPB(&tmpDataPoint,distance, &setIDsInDist);

			
			//if the number of neighbors is less than the number required for a cluster,
			//then it is noise.  The noise will be cluster 0.
			if (setIDsInDist.size()<minPts){
				clusterIDs[i]=0;
			}
			//if there's enough points to make a cluster
			else{
				tmpcnt++;
				clusterCnt++;
				//make a new cluster with the correct cluster ID 
				clusterIDs[i]=clusterCnt;	


				//assign the neighbor ids to the neighborSet, which may be expanded when searching
				//through the neighbors
				neighborSet=setIDsInDist;
				
				//expand the cluster 	
				 while (neighborSet.size()!=0){
				 	//examine the point that's the last one in the neighbor set
				 	int pntID=neighborSet.back(); 
				 	
				 	//if this point has been visited before
				 	if (visited[pntID]==true){
				 	//remove the value from the list of neighbors to check 	
				 	neighborSet.pop_back();
					continue;
					}

					//mark the point as visited:
					visited[pntID]=true;

					//get the neighbors of the data point
					dataElem tmpDataPoint=(*dataPoints)[pntID];
					getNeighboursParallelMPB(&tmpDataPoint,distance, &setIDsInDist);

					//if the number of neighbors is greater than the number required to form a cluster
					if (setIDsInDist.size()>=minPts)
					{
						//assign the neighbor ids to the neighborSet
						copyVect(&neighborSet,&setIDsInDist);		
					}
					//if the point has part not been assigned to a cluster yet
					if (clusterIDs[pntID]==0){
						clusterIDs[pntID]=clusterCnt;							
					}
					


				 } //end of while loop
			} //end of else
			

						
			

		} //end of main for loop

		//increment the total cluster count by 1 because cluster 0 is for the noise data points
		clusterCnt++;

		numClustersForStats=clusterCnt;
	}


//This isn't really parallel-but allows for buffers from the r-tree and the nighbour functions etc
//such that multiple dbscan objects can traverse the tree, get neighbours, etc.
void DBScan::algDBScanParallel()
	{
		clusterCnt=0;

		//neighborSet is the current set of points being searched that belongs to a cluster
		std::vector<int>neighborSet;

		for (int i=0; i<(*dataPoints).size(); i++){
							
			//see if the point has been visited, if so, go onto the next loop iteration
			if (visited[i]==true){
				continue;
			}
			

			//clear the vector of neighbors
			neighborSet.clear();

			//mark the point as visited:
			visited[i]=true;
			
			//get the neighbors of the data point
			dataElem tmpDataPoint=(*dataPoints)[i];
			
			getNeighboursParallelMPB(&tmpDataPoint,distance, &setIDsInDist);

			
			//if the number of neighbors is less than the number required for a cluster,
			//then it is noise.  The noise will be cluster 0.
			if (setIDsInDist.size()<minPts){
				clusterIDs[i]=0;
			}
			//if there's enough points to make a cluster
			else{
				
				clusterCnt++;
				//make a new cluster with the correct cluster ID 
				clusterIDs[i]=clusterCnt;	


				//assign the neighbor ids to the neighborSet, which may be expanded when searching
				//through the neighbors
				neighborSet=setIDsInDist;
				
				//expand the cluster 	
				 while (neighborSet.size()!=0){
				 	//examine the point that's the last one in the neighbor set
				 	int pntID=neighborSet.back(); 
				 	
				 	//if this point has been visited before
				 	if (visited[pntID]==true){
				 	//remove the value from the list of neighbors to check 	
				 	neighborSet.pop_back();
					continue;
					}

					//mark the point as visited:
					visited[pntID]=true;

					//get the neighbors of the data point
					dataElem tmpDataPoint=(*dataPoints)[pntID];
					getNeighboursParallelMPB(&tmpDataPoint,distance, &setIDsInDist);

					//if the number of neighbors is greater than the number required to form a cluster
					if (setIDsInDist.size()>=minPts){
						//assign the neighbor ids to the neighborSet
						copyVect(&neighborSet,&setIDsInDist);		
					}
					//if the point has part not been assigned to a cluster yet
					if (clusterIDs[pntID]==0){
						clusterIDs[pntID]=clusterCnt;							
					}
					


				 } //end of while loop
			} //end of else

		} //end of main for loop

		//increment the total cluster count by 1 because cluster 0 is for the noise data points
		clusterCnt++;

		//for stats
		numClustersForStats=clusterCnt;
		
	}








	//appends the elements from the source vector to the end of the dest vector
	void DBScan::copyVect(std::vector<int> * dest, std::vector<int> * source)
	{
		for (int i=0; i<(*source).size(); i++){
			(*dest).push_back((*source)[i]);	
		}
	}


	void DBScan::getNeighboursParallelMPB(struct dataElem * point, double distance, std::vector<int> * setIDsInDistPtr){
		
		
		char arg[500];
		int index=0;
		
		//when using the implementation that shares clustering results
		index=omp_get_thread_num();		
		

		//first, clear the temporary list of neighbours-- these are the ones from the R-tree
		//not the final ones that have been filtered
		//should maintain the memory allocation, so the vector doesn't need to grow again
		neighbourListParallel[index].clear();		

		//send this to the callback function
		sprintf(arg,"%d",index); 

		//construct an MBB for the point
		double MBB_min[2];
		double MBB_max[2];

		generateMBBNormal(point, distance, MBB_min, MBB_max);

		//arg is the index for the buffer for the thread in the callback function
		(*tree).Search(MBB_min,MBB_max, DBSCANmySearchCallbackParallel, arg); 
			
		//after the candidate set has been found, then we filter the points to find those that are actually 
		//within the distance		
		filterCandidatesMPB(point, &neighbourListParallel[index], distance, setIDsInDistPtr);		

		
		
	}



	//THIS IS FOR THE VERSION WITH MULTIPLE POINTS PER MBB
	int DBScan::filterCandidatesMPB(struct dataElem * point, std::vector<int> * candidateSet, double distance, std::vector<int> * setIDsInDistPtr){

		//first, clear the vector.  It should maintain its memory allocation so it can only grow.
		setIDsInDistPtr->clear();
		
		
		for (int i=0; i<(*candidateSet).size(); i++)
		{
					//DECOMPOSE THE CANDIDATE SET, WHICH CONTAINS THE RESULT OF THE OVERLAPPING MBBS
					//WITH THE MULTIPLE POINT BOXES (MULTIPLE POINTS PER MBB)
					//the ID of a candidate point
					int MBBID=(*candidateSet)[i];
													
					for (int j=0; j<(*ptrMPBlookup)[MBBID].size(); j++){
					int candID=(*ptrMPBlookup)[MBBID][j];
					
					//make a temp copy of the candidate data point
					dataElem candPoint=(*dataPoints)[candID];

					//calculate the distance between the point and the candidate point
					//if it is within the threshold distance, then add it to the vector of IDs that are within
					//the threshold distance
					if (EuclidianDistance(point,&candPoint)<=distance){
						setIDsInDistPtr->push_back(candID);
					}
					
				}//end of inner for loop
		
		} //end of outer for loop

	}


	//The 2D EuclidianDistance
	double DBScan::EuclidianDistance(struct dataElem * point1, struct dataElem * point2)
	{
		return(sqrt(((point1->x-point2->x)*(point1->x-point2->x))+((point1->y-point2->y)*(point1->y-point2->y))));
	}


	//initialize all of the points to initially not be visited
	void DBScan::initializeVisitedPoints(int size)
	{
		for (int i=0; i<size; i++){
			visited.push_back(false);
		}
	}

	void DBScan::initializeClusterIDs(int size)
	{
		for (int i=0; i<size; i++){
			clusterIDs.push_back(0);
		}
	}

	//generate a query MBB around the point to search for the values
	//this supports the other two generate MBB methods 
	void DBScan::generateMBB(struct dataElem * point, double distance, double * MBB_min, double * MBB_max)
	{
		//the MBB is made up by the same time and electron content values, but with the distance added or 
		//subtracted from the MBB
		MBB_min[0]=(point->x)-distance;
		MBB_min[1]=(point->y)-distance;
		MBB_max[0]=(point->x)+distance;
		MBB_max[1]=(point->y)+distance;

	}

	//generate a query MBB around the point to search for the values
	//returns true if it was able to generate the query using a single MBB
	//returns false if it needs two MBBs because the query wraps around the longitude of 360 degrees
	bool DBScan::generateMBBNormal(struct dataElem * point, double distance, double * MBB_min, double * MBB_max)
	{
		generateMBB(point, distance, MBB_min, MBB_max);
		return true;
	}


