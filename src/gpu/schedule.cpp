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


#include "schedule.h"

#include <omp.h>
#include <vector>
#include <algorithm>

Schedule::Schedule(vector<struct experiment> * experimentList){
	//pointer to list of experiments:
	expPtr=experimentList;	

	// for (int i=0; i<expPtr->size();i++)
	// {
	// 	printf("\ntest minpts: %d ",(*expPtr)[i].minpts);
	// }

	//initialize vector that keeps track of which instances have finished so the scheduler knows whats are available for reuse
	for (int i=0; i<experimentList->size();i++)
	{
		finishedList.push_back(false);
	}

	//greedy schedule
	#if SCHEDULE==0
	//initialize a queue that assigns variants to theads
	for (int i=0; i<experimentList->size();i++)
	{
		jobQueue.push(i);
	}	
	printf("\nsize of job queue: %zu",jobQueue.size());
	
	
	// while (!jobQueue.empty())
	// {
	// 	 cout<<"\njob queue: "<<jobQueue.front();
	// 	 jobQueue.pop();
	// }

	#endif

	#if SCHEDULE==1
	generatePriorityList();
	scheduleCnt=0;
	#endif

	
}


//returns false if the experiment should compute the cluster normally.
//If it can reuse a previous cluster result, it returns the experiment ID through outID
// inExperID- the id in the experiment list
// outIDReuse- the instance that has data to be reused
// outInstanceCluster- the instance in the experiment list to be clustered
bool Schedule::determineReuse(int inExperID, int * outIDReuse, int * outInstanceCluster){	
	
	//greedy schedule
	#if SCHEDULE==0
	
	//find an appropriate experiment to reuse or tell the experiment to do the full clustering algorithm
	//By definition, the very first experiment needs to be calculated because it has the smallest epsilon and highest minpts

	//Greedy:
	//Find the next useful one to reuse, if none exist, then compute yourself

	//in the greedy schedule the inExperID will be the outInstanceCluster, because the schedule clusters based on the experiment list ordering
	return schedGreedy(outIDReuse,outInstanceCluster);

	#endif

	//minpts schedule
	#if SCHEDULE==1
		
	int expID=0;
	bool exitFlag=false;
	bool exitFlagPriority=false;
		
	//only 1 thread can access the schedule
	#pragma omp critical
	{

		if (!jobQueue.empty()){
			expID=jobQueue.front();
				
			//check the id and see if it's one that needs to be clustered from scratch by the minpts schedule:
			if(clusterScratchList[expID]==true)
			{
				printf("\ncluster scratch list id: %d",expID);
				*outIDReuse=expID; 
				*outInstanceCluster=expID;
				//remove it from the queue
				jobQueue.pop();
				exitFlagPriority=true;					
			}

		}
		//no variants left to cluster
		else
		{
			exitFlag=true;
		}
	
	}//end critical section

	if(exitFlag==true)
	{
		return false;
	}

	//if a variant that needs to be clustered from scratch was found, return true, and don't call greedy schedule
	if(exitFlagPriority==true)
	{
		return true;
	}

	
		return schedGreedy(outIDReuse,outInstanceCluster);		
	

	


	// int instanceToBeClustered=0;	

	// //only allow one thread to access the schedule at a time
	// #pragma omp critical
	// {
		
	// 	//the instance to be clustered comes from the priority list
	// 	instanceToBeClustered=priorityList[scheduleCnt];
	// 	*outInstanceCluster=instanceToBeClustered;
	// 	scheduleCnt++;
	// }	
	// 	//if the instance needs to be clustered from scratch
	// 	if (clusterScratchList[instanceToBeClustered]==true)
	// 	{
	// 		return false;
	// 	}
	// 	else
	// 	{
	// 		return schedGreedy(instanceToBeClustered, outIDReuse);			
	// 	}
		
	
	
	#endif
	


}


//you want to reuse the experiment with the closest (or equal) eps, and the same or greater minpts
//experiments are ordered by epsilon, and in decreasing minpts values
//outIDReuse-the variant that should be reused
//outInstanceCluster-the variant that's being clustered
//returns false if there are no variants (jobs) left to cluster
bool Schedule::schedGreedy(int * outIDReuse, int * outInstanceCluster)
{
	int expID=0;
	bool exitFlag=false;

	//only 1 thread can access the schedule
	#pragma omp critical
	{

		if (!jobQueue.empty()){
			expID=jobQueue.front();
			int tid=omp_get_thread_num();
			printf("\ntid: %d, expID: %d",tid,expID);
			jobQueue.pop();
		}
		//no variants left to cluster
		else
		{
			exitFlag=true;
		}
	
	}//end critical section

	if(exitFlag==true)
	{
		return false;
	}


	//we cluster this variant from the Job Queue
	*outInstanceCluster=expID;

	//by default:
	//if no instances can be reused, then set the outInstanceCluster to expID, signaling that
	//no cluster exists for reuse, and the cluster needs to cluster from scratch
	*outIDReuse=expID;
	
	

	

	for (int i=expID-1; i>=0; i--)
	{
		//double eps=expPtr[i]->epsilon;
		//int minpts=expPtr[i]->minpts;

		if (finishedList[i]==true && (((*expPtr)[i].minpts)>=((*expPtr)[expID].minpts)))
		{
			printf("\nExperiment: %d is going to reuse experiment: %d",expID,i);
			*outIDReuse=i;
			return true;
		}
	}

	
	
	return true;

}

//makes a priority list of the order in which the variants are to be clustered
//don't reshuffle the schedule
void Schedule::generatePriorityList()
{

	//since the list is sorted (increasing eps, decreasing minpts), the very first value in the list needs to be clustered from scratch
	priorityList.push_back(0);

	//first make the list of experiments that need to be clustered from scratch.
	//can easily choose them because they are already sorted by epsilon and minpts
	for (int i=1; i<expPtr->size()-1; i++)
	{
		double eps1=(*expPtr)[i].epsilon;
		double eps2=(*expPtr)[i+1].epsilon;
		if (eps1!=eps2)
		{
			priorityList.push_back(i+1);
		}
	}

	for (int i=0; i<priorityList.size(); i++)
	{
		printf("\nprioritizing: %d",priorityList[i]);

	}

	//now we go and add in the same order the rest of the variants to the prioritylist
	for (int i=0; i<expPtr->size(); i++)
	{
		bool inPriorityList=false;
		for (int j=0; j<priorityList.size();j++)
		{
			if (i==priorityList[j])
			{
				inPriorityList=true;
				clusterScratchList.push_back(true); //keeps track of which variants are clustered from scratch 
			}
		}

		if (inPriorityList==false)
		{
			priorityList.push_back(i);
			clusterScratchList.push_back(false); //keeps track of which variants are clustered from scratch 
		}
	}
	
	for (int i=0; i<priorityList.size(); i++)
	{
		printf("\nnew order: %d",priorityList[i]);
	}

	for (int i=0; i<clusterScratchList.size(); i++)
	{
		int flag=0;
		if (clusterScratchList[i]==true)
		{
			flag=1;
		}
		printf("\ncluster scratch list: %d",flag);
	}


	//add the priority list to the job queue
	for (int i=0; i<priorityList.size(); i++)
	{
		jobQueue.push(priorityList[i]);
	}	
	
	// 	while (!jobQueue.empty())
	// {
	// 	 cout<<"\njob queue2: "<<jobQueue.front();
	// 	 jobQueue.pop();
	// }


} //end method

//of pointsSqAreaStruct struct
bool Schedule::compareExperiment(const experiment &a, const experiment &b)
{
    return (a.minpts > b.minpts);
}



