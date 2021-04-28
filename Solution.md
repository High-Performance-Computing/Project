# Solution


## Our Approach
 

### Levels of Parallelism


### Types of Parallelism within Application

The focus is data parallelism as computations on different parts of the graph will be running in parallel. For example, in PageRank, the adjacency matrix representation of the graph is evenly divided row-wise across all processors. Each processor then performs multiplication between the transition probability matrix its rows of the adjacency matrix. 

### Programming Models


### Infrastructure

##### FAS RC

We used 20 nodes with 4 GPUs per node on FAS RC.