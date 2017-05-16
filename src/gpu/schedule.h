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


#ifndef SCHEDULE_H_
#define SCHEDULE_H_


#include "structs.h"
#include <queue> 

//The schedule determines whether a given experiment should reuse previous clustering results or 
//Do the actual clustering itself which can then be used later for other instances

using namespace std;

class Schedule{
	


	public:
	Schedule(vector<struct experiment> * experimentList);

	bool determineReuse(int inExperID, int * outIDReuse, int * outInstanceCluster);

	std::vector<bool>finishedList; //keeps track of instances that have completely finished

	//pointer to the vector of experiments
	std::vector<struct experiment> * expPtr;
	
	//list of variants that need to be clustered from scratch
	//std::vector<int>priorityList;

	//list of variants that have been assigned to a thread
	//std::vector<bool>assignedList;

	


	private:

	int scheduleCnt;	

	bool schedGreedy(int * outID, int * outInstanceCluster);	
	

	//a priority list of whether an instance should be clustered from scratch or not
	//schedInfo * priorityList;	

	//the order in which the variants should be clustered
	//stores the ids of the experiment list
	std::vector<int> priorityList;

	//corresponding list of variants that must be clustered from scratch
	std::vector<bool> clusterScratchList;
	
	//queue of the order of the priority list:
	std::queue<int>jobQueue;

	void generatePriorityList();

	static bool compareExperiment(const experiment &a, const experiment &b);

};


#endif