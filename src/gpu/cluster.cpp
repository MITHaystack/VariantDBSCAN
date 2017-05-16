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

#include <vector>
#include <stdio.h>
#include "structs.h"
#include "cluster.h"
#include <math.h>
#include <algorithm>


//don't take in epsilon because we know what the neighbors are already!
//This one assumes perfect information (each point knows its exact neighbors)

//modified to use arrays and not vectors inside the neighbortable
void dbscanAlternate(struct neighborTableLookup * neighborTable, unsigned int numPoints, int minPts)
{	

	
	int clusterCnt=0;

	//neighborSet is the current set of points being searched that belongs to a cluster
	std::vector<int>neighborSet;

	//vector that keeps track of the points that have beeen visited
	//initialize all points to not being visited
	std::vector<bool>visited(numPoints,false);



	//vector that keeps track of the assignment of the points to a cluster
 	//cluster 0 means a noise point

	std::vector<int>clusterIDs(numPoints,0); //initialize all points to be in cluster 0
	


	for (int i=0; i<numPoints; i++){
			

		//see if the point has been visited, if so, go onto the next loop iteration
		if (visited[i]==true){
			continue;
		}

		//clear the vector of neighbors
		neighborSet.clear();

		//mark the point as visited:
		visited[i]=true;
		
		
		
		//get the neighbors of the data point
		//dataElem tmpDataPoint=(*dataPoints)[i];
		//getNeighbours(&tmpDataPoint,distance, &setIDsInDist);

		
		//if the number of neighbors is less than the number required for a cluster,
		//then it is noise.  The noise will be cluster 0.
		if (((neighborTable[i].indexmax-neighborTable[i].indexmin)+1)<minPts)
		{
			clusterIDs[i]=0;
		}
				
		
		//if there's enough points to make a cluster
		
		else
		{
				
			clusterCnt++;
			//make a new cluster with the correct cluster ID 
			clusterIDs[i]=clusterCnt;	


			//printf("\n***1size of neighbor set: %d", neighborSet.size());

			//assign the neighbor ids to the neighborSet, which may be expanded when searching
			//through the neighbors
			int sizeInsert=neighborTable[i].indexmax-neighborTable[i].indexmin+1;
			neighborSet.insert(neighborSet.end(),&neighborTable[i].dataPtr[neighborTable[i].indexmin], &neighborTable[i].dataPtr[neighborTable[i].indexmin]+(sizeInsert));

			//printf("\n***size of neighbor set: %d", neighborSet.size());
			

			
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
				//dataElem tmpDataPoint=(*dataPoints)[pntID];

				//getNeighbours(&tmpDataPoint,distance, &setIDsInDist);
				//getNeighboursBruteForce(&tmpDataPoint,distance);

				//if the number of neighbors is greater than the number required to form a cluster
				
				if (((neighborTable[pntID].indexmax-neighborTable[pntID].indexmin)+1)>=minPts)
				{
					//assign the neighbor ids to the neighborSet
					//copyVect(&neighborSet,&setIDsInDist);	
					
					//XXXXX
					//CHANGE THIS LATER TO INSERTS
					//XXXXX
					for (int j=neighborTable[pntID].indexmin; j<=neighborTable[pntID].indexmax;j++)
					{
						neighborSet.push_back(neighborTable[pntID].dataPtr[j]);
					}	
				}
				//if the point has part not been assigned to a cluster yet
				
				//AFTER FIXING THE BORDER POINTS TO BE PART OF A CLUSTER
				if (clusterIDs[pntID]==0){
					clusterIDs[pntID]=clusterCnt;							
				}
				
				


			 } //end of while loop
		} //end of else
		

					
		//testTotalNeighbors+=neighbourList.size();

		//now have the vector of ids within the distance
		//setIDsInDist
		
		

	} //end of main for loop

	//increment the total cluster count by 1 because cluster 0 is for the noise data points
	clusterCnt++;

	


	// printf("\n***printing cluster array from GPU version with neighbor table:");
	// for (int i=0; i<clusterIDs.size(); i++)
	// {

	// 	printf("\n%d, %d",i,clusterIDs[i]);
	// }

	// printf("\n***end of printing cluster array:");



	printf("\ntotal clusters, including cluster 0, which is the noisy points: %d", clusterCnt);
	


}











//don't take in epsilon because we know what the neighbors are already!
//This one assumes perfect information (each point knows its exact neighbors)
void dbscan(struct table * neighborTable, unsigned int numPoints, int minPts)
{	
	//printf("\ntotal data points: %zu", (*dataPoints).size());
	int clusterCnt=0;

	//neighborSet is the current set of points being searched that belongs to a cluster
	std::vector<int>neighborSet;

	//vector that keeps track of the points that have beeen visited
	//initialize all points to not being visited
	std::vector<bool>visited(numPoints,false);



	//vector that keeps track of the assignment of the points to a cluster
 	//cluster 0 means a noise point

	std::vector<int>clusterIDs(numPoints,0); //initialize all points to be in cluster 0
	


	for (int i=0; i<numPoints; i++){
			

		//see if the point has been visited, if so, go onto the next loop iteration
		if (visited[i]==true){
			continue;
		}

		//clear the vector of neighbors
		neighborSet.clear();

		//mark the point as visited:
		visited[i]=true;
		
		
		
		//get the neighbors of the data point
		//dataElem tmpDataPoint=(*dataPoints)[i];
		//getNeighbours(&tmpDataPoint,distance, &setIDsInDist);

		
		//if the number of neighbors is less than the number required for a cluster,
		//then it is noise.  The noise will be cluster 0.
		if (neighborTable[i].neighbors.size()<minPts)
		{
			clusterIDs[i]=0;
		}
				
		
		//if there's enough points to make a cluster
		
		else
		{
				
			clusterCnt++;
			//make a new cluster with the correct cluster ID 
			clusterIDs[i]=clusterCnt;	


			//printf("\n***1size of neighbor set: %d", neighborSet.size());

			//assign the neighbor ids to the neighborSet, which may be expanded when searching
			//through the neighbors
			neighborSet=neighborTable[i].neighbors;
			//copyVect(&neighborSet,&neighbourList);
			

			
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
				//dataElem tmpDataPoint=(*dataPoints)[pntID];

				//getNeighbours(&tmpDataPoint,distance, &setIDsInDist);
				//getNeighboursBruteForce(&tmpDataPoint,distance);

				//if the number of neighbors is greater than the number required to form a cluster
				if (neighborTable[pntID].neighbors.size()>=minPts)
				{
					//assign the neighbor ids to the neighborSet
					//copyVect(&neighborSet,&setIDsInDist);	
					for (int j=0; j<neighborTable[pntID].neighbors.size();j++)
					{
						neighborSet.push_back(neighborTable[pntID].neighbors[j]);
					}	
				}
				//if the point has part not been assigned to a cluster yet
				
				//AFTER FIXING THE BORDER POINTS TO BE PART OF A CLUSTER
				if (clusterIDs[pntID]==0){
					clusterIDs[pntID]=clusterCnt;							
				}
				
				


			 } //end of while loop
		} //end of else
		

					
		//testTotalNeighbors+=neighbourList.size();

		//now have the vector of ids within the distance
		//setIDsInDist
		
		

	} //end of main for loop

	//increment the total cluster count by 1 because cluster 0 is for the noise data points
	clusterCnt++;

	


	// printf("\n***printing cluster array from GPU version with neighbor table:");
	// for (int i=0; i<clusterIDs.size(); i++)
	// {

	// 	printf("\n%d, %d",i,clusterIDs[i]);
	// }

	// printf("\n***end of printing cluster array:");



	printf("\ntotal clusters, including cluster 0, which is the noisy points: %d", clusterCnt);
	


}



//Take in epsilon because the table has extra neighbors
//This one assumes imperfect information (each point in the neighbor table is not necessarily within epsilon)
void dbscanWithFilter(std::vector<struct dataElem> *dataPoints, struct table * neighborTable, double epsilon, int minPts)
{	
	//printf("\ntotal data points: %zu", (*dataPoints).size());
	int clusterCnt=0;

	//neighborSet is the current set of points being searched that belongs to a cluster
	std::vector<int>neighborSet;

	//vector that keeps track of the points that have beeen visited
	//initialize all points to not being visited
	std::vector<bool>visited(dataPoints->size(),false);



	//vector that keeps track of the assignment of the points to a cluster
 	//cluster 0 means a noise point

	std::vector<int> clusterIDs(dataPoints->size(),0); //initialize all points to be in cluster 0
	
	std::vector<int> setIDsInDist;


	for (int i=0; i<dataPoints->size(); i++){
			

		//see if the point has been visited, if so, go onto the next loop iteration
		if (visited[i]==true){
			continue;
		}

		//clear the vector of neighbors
		neighborSet.clear();

		//mark the point as visited:
		visited[i]=true;
		
		
		//
		//get the neighbors of the data point
		//dataElem tmpDataPoint=(*dataPoints)[i];
		
		filterCandidates(i, dataPoints, &neighborTable[i].neighbors, epsilon, &setIDsInDist);

		
		//if the number of neighbors is less than the number required for a cluster,
		//then it is noise.  The noise will be cluster 0.
		if (setIDsInDist.size()<minPts)
		{
			clusterIDs[i]=0;
		}
				
		
		//if there's enough points to make a cluster
		
		else
		{
				
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
				//dataElem tmpDataPoint=(*dataPoints)[pntID];

				//getNeighbours(&tmpDataPoint,distance, &setIDsInDist);
				//getNeighboursBruteForce(&tmpDataPoint,distance);

				filterCandidates(pntID, dataPoints, &neighborTable[pntID].neighbors, epsilon, &setIDsInDist);

				//if the number of neighbors is greater than the number required to form a cluster
				if (setIDsInDist.size()>=minPts)
				{
						//assign the neighbor ids to the neighborSet
						copyVect(&neighborSet,&setIDsInDist);		
				}
				//if the point has part not been assigned to a cluster yet
				
				//AFTER FIXING THE BORDER POINTS TO BE PART OF A CLUSTER
				if (clusterIDs[pntID]==0){
					clusterIDs[pntID]=clusterCnt;							
				}
				
				


			 } //end of while loop
		} //end of else
		

					
		//testTotalNeighbors+=neighbourList.size();

		//now have the vector of ids within the distance
		//setIDsInDist
		
		

	} //end of main for loop

	//increment the total cluster count by 1 because cluster 0 is for the noise data points
	clusterCnt++;

	


	// printf("\n***printing cluster array from GPU version with neighbor table that filters:");
	// for (int i=0; i<clusterIDs.size(); i++)
	// {

	// 	printf("\n%d, %d",i,clusterIDs[i]);
	// }

	// printf("\n***end of printing cluster array:");



	printf("\ntotal clusters, including cluster 0, which is the noisy points: %d", clusterCnt);
	


}


//appends the elements from the source vector to the end of the dest vector
void copyVect(std::vector<int> * dest, std::vector<int> * source)
{

	// printf("\ncpyvect size of dest: %d", dest->size());
	// printf("\ncpyvect size of source: %d", source->size());
	for (int i=0; i<(*source).size(); i++)
	{
		(*dest).push_back((*source)[i]);	
	}
}


//This function takes the candidate set from the table and filters them to find those that are
//actually within the threshold distance, epsilon
//takes as input the point, the candidateSet pointers, and the distance
//used when filtering the candidates from a preconstructed table has all of the neighbors for a higher epsilon value
int filterCandidates(int pointID, std::vector<struct dataElem> * dataPoints, std::vector<int> * candidateSet, double distance, std::vector<int> * setIDsInDistPtr)
{

		//first, clear the vector.  It should maintain its memory allocation so it can only grow.
		setIDsInDistPtr->clear();

		//printf("\n candidate set size: %d", (*candidateSet).size());

		for (int i=0; i<(*candidateSet).size();i++)
		{
			//the ID of a candidate point
			int candID=(*candidateSet)[i];

			//make a temp copy of the candidate data point
			//dataElem candPoint=(*dataPoints)[candID];

			//calculate the distance between the point and the candidate point
			//if it is within the threshold distance, then add it to the vector of IDs that are within
			//the threshold distance
			
			if (EuclidianDistance(&(*dataPoints)[pointID], &(*dataPoints)[candID])<=distance)
			{
				setIDsInDistPtr->push_back(candID);
			}
		}

	}

    

double EuclidianDistance(struct dataElem * point1, struct dataElem * point2)
{
		return(sqrt(((point1->x-point2->x)*(point1->x-point2->x))+((point1->y-point2->y)*(point1->y-point2->y))));
}

// void dbscan()
// {
// 	printf("\ntotal data points: %zu", (*dataPoints).size());
// 	clusterCnt=0;

// 	//neighborSet is the current set of points being searched that belongs to a cluster
// 	std::vector<int>neighborSet;

// 	for (int i=0; i<(*dataPoints).size(); i++){
			

// 		//see if the point has been visited, if so, go onto the next loop iteration
// 		if (visited[i]==true){
// 			continue;
// 		}

// 		//clear the vector of neighbors
// 		neighborSet.clear();

// 		//mark the point as visited:
// 		visited[i]=true;
		
// 		//get the neighbors of the data point
// 		dataElem tmpDataPoint=(*dataPoints)[i];
		


// 		getNeighbours(&tmpDataPoint,distance, &setIDsInDist);

		
// 		//if the number of neighbors is less than the number required for a cluster,
// 		//then it is noise.  The noise will be cluster 0.
// 		if (setIDsInDist.size()<minPts)
// 		{
// 			clusterIDs[i]=0;
// 		}
			
		
// 		//if there's enough points to make a cluster
		
// 		else
// 		{
			
// 			clusterCnt++;
// 			//make a new cluster with the correct cluster ID 
// 			clusterIDs[i]=clusterCnt;	


// 			//printf("\n***1size of neighbor set: %d", neighborSet.size());

// 			//assign the neighbor ids to the neighborSet, which may be expanded when searching
// 			//through the neighbors
// 			neighborSet=setIDsInDist;
// 			//copyVect(&neighborSet,&neighbourList);
			
			
// 			//expand the cluster
			 	
// 			 while (neighborSet.size()!=0){
// 			 	//examine the point that's the last one in the neighbor set
// 			 	int pntID=neighborSet.back(); 
			 	
// 			 	//if this point has been visited before
// 			 	if (visited[pntID]==true){
// 			 	//remove the value from the list of neighbors to check 	
// 			 	neighborSet.pop_back();
// 				continue;
// 				}

// 				//mark the point as visited:
// 				visited[pntID]=true;

// 				//get the neighbors of the data point
// 				dataElem tmpDataPoint=(*dataPoints)[pntID];

// 				getNeighbours(&tmpDataPoint,distance, &setIDsInDist);
// 				//getNeighboursBruteForce(&tmpDataPoint,distance);

// 				//if the number of neighbors is greater than the number required to form a cluster
// 				if (setIDsInDist.size()>=minPts)
// 				{
// 					//assign the neighbor ids to the neighborSet
// 					copyVect(&neighborSet,&setIDsInDist);		
// 				}
// 				//if the point has part not been assigned to a cluster yet
				
// 				//AFTER FIXING THE BORDER POINTS TO BE PART OF A CLUSTER
// 				if (clusterIDs[pntID]==0){
// 					clusterIDs[pntID]=clusterCnt;							
// 				}
				
				


// 			 } //end of while loop
// 		} //end of else
		

					
// 		//testTotalNeighbors+=neighbourList.size();

// 		//now have the vector of ids within the distance
// 		//setIDsInDist
		
		

// 	} //end of main for loop

// 	//increment the total cluster count by 1 because cluster 0 is for the noise data points
// 	clusterCnt++;

	


// 	// printf("\n***printing cluster array:");
// 	// for (int i=0; i<clusterIDs.size(); i++)
// 	// {

// 	// 	printf("\ni, ID: %d, %d",i,clusterIDs[i]);
// 	// }

// 	printf("\n***end of printing cluster array:");



// 	printf("\ntotal clusters, including cluster 0, which is the noisy points: %d", clusterCnt);
	


// }