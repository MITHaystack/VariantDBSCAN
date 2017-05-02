#This implementation uses the Greedy Schedule (SchedGreedy), and the density cluster reuse strategy (ClusDensity)
#Described in the publication:
#Gowanlock, M., Blair, D. M. & Pankratius, V. (2016) Exploiting Variant-Based Parallelism for Data Mining of Space Weather Phenomena. 
#In Proc. of the 30th IEEE International Parallel & Distributed Processing Symposium (IPDPS 2016). pp. 760-769 DOI: 10.1109/IPDPS.2016.10

#The shared libarary function takes as input:
#The dataset
#The variant list
#The number of points per R-tree minimum bounding box for the index

#Parameters that can be modified: 

#Parameter: SEARCHTHREADS
#The number of threads that can execute in parallel for running VDBSCAN 
#Default: 4 threads
SEARCHTHREADS = -DNSEARCHTHREADS=4 

#Parameter: libpath
#The location where shared libraries should be put in the system
#Default: /usr/local/lib/
libpath = /usr/local/lib/


OBJECTS =  main.o tree_functions.o DBScan.o schedule.o
CC = gcc-5
CXX= g++-5
CXXFLAGS = -O3 $(SEARCHTHREADS)	


#For compiling the shared library:
libSharedVDBSCAN.so: $(OBJECTS)
	$(CXX) -O3 -fopenmp -c $(SEARCHTHREADS) -fPIC -Werror schedule.cpp DBScan.cpp tree_functions.cpp  
	$(CXX) -O3 -fopenmp -c $(SEARCHTHREADS) -fPIC -Werror main.cpp 
	$(CXX) -O3 -shared -fopenmp -fPIC $(OBJECTS) $(SEARCHTHREADS) -o libSharedVDBSCAN.so 
	cp libSharedVDBSCAN.so $(libpath)

#For compiling the c++ test program that uses the shared library (above)
#-ldl is to link against the loader for the shared library
c_test_prog: c_test_prog.o
	$(CC) -O3 -fopenmp -lSharedVDBSCAN c_test_prog.cpp -ldl -o c_test_prog 




#For compiling the binary for testing
# all:
# 	$(CC) -O3 -fopenmp -c $(SEARCHTHREADS) schedule.cpp DBScan.cpp tree_functions.cpp 
# 	$(CC) -O3 -fopenmp -c $(SEARCHTHREADS) main.cpp 
# 	$(CC) -O3 -fopenmp $(OBJECTS) -o main 

clean:
	rm c_test_prog $(OBJECTS) libSharedVDBSCAN.so



