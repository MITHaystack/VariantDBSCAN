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

#include "schedule.h"
#include <omp.h>
#include <vector>
#include <algorithm>



Schedule::Schedule(vector<struct experiment> * experimentList){
	//pointer to list of experiments:
	expPtr=experimentList;	

	for (int i=0; i<experimentList->size();i++){
		finishedList.push_back(false);
	}

	//greedy schedule
	//initialize a queue that assigns variants to theads
	for (int i=0; i<experimentList->size();i++){
		jobQueue.push(i);
	}	
	
}


//returns false if the experiment should compute the cluster normally.
//If it can reuse a previous cluster result, it returns the experiment ID through outID
// inExperID- the id in the experiment list
// outIDReuse- the instance that has data to be reused
// outInstanceCluster- the instance in the experiment list to be clustered
bool Schedule::determineReuse(int inExperID, int * outIDReuse, int * outInstanceCluster){	
	
	//Greedy schedule:
	//Find the next useful one to reuse, if none exist, then compute yourself
	//For Greedy, the schedule clusters based on the already sorted experiment list ordering
	return schedGreedy(outIDReuse,outInstanceCluster);
}


//Reuse the experiment with the closest (or equal) eps, and the same or greater minpts
//Experiments are ordered by epsilon, and in decreasing minpts values
//outIDReuse-the variant that should be reused
//outInstanceCluster-the variant that's being clustered
//Returns false if there are no variants left to cluster
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
			jobQueue.pop();
		}
		//no variants left to cluster
		else
		{
			exitFlag=true;
		}
	
	}//end critical section

	//have to return outside of the OMP section
	if(exitFlag==true){
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
		if (finishedList[i]==true && (((*expPtr)[i].minpts)>=((*expPtr)[expID].minpts))){
			*outIDReuse=i;
			return true;
		}
	}

	
	
	return true;

}





