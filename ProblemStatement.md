# Problem

## Numerical complexity

The numerical complexity of doing late-resetting and masking is O(MNt). 

- M is the number of thresholds for our masks (each mask gives us one subnetwork)
- N is the length of the trellis for late resetting  
- t is the average time to train a network (we actually use sparse subnetworks, so they train faster than the original one).

### Theoretical Speed up & expected scalability: no worker parallelization

<p align="justify"> In our case, the numerical complexity of doing late-resetting and masking is O(100t). </p>

- We take M = 20. We have 20 sparse subnetworks.
- We take N = 5. We do 5 resetting of the weights
- We estimate t at 26 h 15 min without worker parallelization. 

<p align="justify"> Thus, the expected time in order to run all the sparser substructures from the different epochs is 
2625 hours without worker parallelization. </p>


## Theoretical speed-up and scalability expected


<p align="justify"> Each worker needs to do several late resetting for the particular structure found after masking. Afterwards, there is no communication between the worker nodes. The communication time at the beginning is negligible compared to training time. The computation time per epoch is 4.5 minutes at best. We have 350 epochs, and perform 5 late resetting. Thus we achieve a run time of 131 h 15 min at best per worker node. This corresponds to a speed up of 20. </p> 

## Amdahl Law (1967)

Parallel execution Speed-up and Efficiency for a given problem size and a number of processors are given by:

![](Eqns.png)



AMHDAL'S LAW
THEORETICAL PARALLELIZATION PARTS 


GPUs: low occupation
Not using Spark elephas because then reached 100 occupation
Spark for offline processing the data. Reshaping the data as tf tensors.
Callbacks.






# Structure from last year group

## Basics of NN

For those unfamiliar with deep learning, here are some terms we will be using throughout our writeup.

- **Layer**:  
- **Weight:** 


## Challenges

Need for LTH (Storage).

## Algorithm: LTH


## Need for High Performance Computing and Big Data

### High Performance Computing

- The bottleneck for the algorithm is the need for a lot of computation. 

### Need for Big Data

- **Volume**:  
- **Velocity**: 
- **Variety**: 
