# Problem

## Need for Big Compute

**MobileNetV2 Architecture**: 
- Total number of parameters: 3.4M 
- Number of multiply-adds (MAdds) per forward pass: 300M

**Two Nested For Loops to Find the Lottery Ticket Hypothesis**
- Outer for loop: iterate over different masks (pruning thresholds)
- Inner for loop: iterate over the range of late resetting epochs
     - Train a sparse MobileNetV2 CNN per each inner loop iteration

## Need for Big Data

<p align="justify"> We are dealing with a Big Data problem because of the size of our dataset. We do not consider Velocity or Variety (we have all of our data available at once, and we are only delaing with images). However, one can imagine that lottery tickets found on ImageNet could be used on a variety of tasks that use convolution operations (computer vision, speech recognition...). </p> 

- **Volume**: 
     -  Total number of training images: 1.23M
     -  Total number of validation images:100k
     -  ImageNet Dataset size: 157.3 GB
     -  Average image resolution (downloaded): 469x387
     -  Average image resolution (preporcessed): 227x227
- **Velocity**: Not considered in our project 
- **Variety**: Not considered in our project

## Numerical Complexity

The numerical complexity of doing late-resetting and masking is O(MNt). 

- M is the number of thresholds for our masks (each mask gives us one subnetwork)
- N is the length of the trellis for late resetting  
- t is the average time to train a network (we actually use sparse subnetworks, so we expect them to train faster than the original one).

### Theoretical Speed-up & Expected Scalability

<p align="justify"> In our case, the numerical complexity of doing late-resetting and masking is O(100t). </p>

- We take M = 20. We have 20 sparse subnetworks.
- We take N = 5. We do 5 resetting of the weights
- We estimate t at 26 h 15 min without worker parallelization. 

<p align="justify"> Thus, the expected time in order to run all the sparser substructures from the different epochs is 
2625 hours without worker parallelization. </p>

<p align="justify"> We use 20 worker nodes. Each worker needs to do several late resetting for the particular structure found after masking. Afterwards, there is no communication between the worker nodes. The communication time at the beginning is negligible compared to training time. The computation time per epoch is 4.5 minutes at best. We have 350 epochs, and perform 5 late resetting. Thus we achieve a run time of 131 h 15 min at best per worker node. This corresponds to a speed up of 20. </p> 

### Amdahl Law (1967)

Parallel execution Speed-up and Efficiency for a given problem size and a number of processors are given by:

![](Images/Eqns.png)

In our case S=20 and E=1.

