# Problem

## Basics of NN

For those unfamiliar with deep learning, here are some terms we will be using throughout our writeup.

- **Layer**:  
- **Weight:** 

## Need for Big Data

We are dealing with a Big Data problem because of the size of our dataset. We do not consider Velocity or Variety (we have all of our data available at onece, and we are only delaing with images). However, one can imagine that lottery tickets found on Imagenet coukd be used on a variety of tasks that use convolutions (computer vision, speech recognition...)

- **Volume**: 
     -  1.23M training images
- **Velocity**: Not considered in our project 
- **Variety**: Not considered in our project

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

### Amdahl Law (1967)

Parallel execution Speed-up and Efficiency for a given problem size and a number of processors are given by:

![](Eqns.png)

In our case S(

# More

AMHDAL'S LAW
THEORETICAL PARALLELIZATION PARTS 


GPUs: low occupation
Not using Spark elephas because then reached 100 occupation
Spark for offline processing the data. Reshaping the data as tf tensors.
Callbacks.

