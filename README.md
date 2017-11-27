# VariantDBSCAN

## Maximizing Clustering Throughput

Project lead: Mike Gowanlock

Relevant papers: 
* [1] Gowanlock, M., Blair, D. M. & Pankratius, V. (2016) Exploiting Variant-Based Parallelism for Data Mining of Space Weather Phenomena. In Proc. of the 30th IEEE International Parallel & Distributed Processing Symposium (IPDPS 2016). pp. 760-769 DOI: 10.1109/IPDPS.2016.10 
[[http://dx.doi.org/10.1109/IPDPS.2016.10 ]](http://dx.doi.org/10.1109/IPDPS.2016.10 )


* [2] Gowanlock, M., Blair, D. M., Pankratius, V. Optimizing Parallel Clustering Throughput in Shared Memory. IEEE Transactions on Parallel and Distributed Systems DOI: 10.1109/TPDS.2017.2675421 
[[http://dx.doi.org/10.1109/TPDS.2017.2675421]](http://dx.doi.org/10.1109/TPDS.2017.2675421)

![alt text](https://github.com/MITHaystack/VariantDBSCAN/blob/master/img/VDBSCAN.png)

Figure: Relative performance gains utilizing all of the optimizations over the sequential implementation on a space weather TEC dataset in [1, 2]. Values over the black line indicate a performance improvement. The red line indicates the performance gain from index optimizations only. See the papers above.    


We acknowledge support from NSF ACI-1442997 and NASA AIST14-NNX15AG84G.
