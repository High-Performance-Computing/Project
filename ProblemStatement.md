# Problem

## Numerical complexity

The numerical complexity of doing late-resetting and masking is O(MNt). 

- M is the number of thresholds for our masks 
- N is the length of the trellis for late resetting 
- t is the average time to train a network (we will actually use 
sparse subnetworks, so they will train faster than the original one)

## Theoretical Speed up & expected scalability: no worker parallelization

The numerical complexity of doing late-resetting and masking is O(MNt) = O(100t).

- M is the number of thresholds for our masks. We take M = 20. We will have 20 sparse subnetworks
- N is the length of the trellis for late resetting. We take N = 5. We do 5 resetting of the weights
- t is the average time to train a network (we will actually use sparse subnetworks, so they will train faster than the original one). We estimate this at 26 h 15 min 

Expected time in order to run all sparser substructures from the different epochs: 
2625 hours.




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