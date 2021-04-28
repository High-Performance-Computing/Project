# CS 205 Final Project: The Lottery Ticket Hypothesis (LTH) on FAS RC

## Spring 2021

## David Assaraf, Tale Lokvenec, Gael Ancel, Raphael Pellegrin

## Overview: the Lottery Ticket Hypothesis (LTH)

Our work will be based on the paper: The Lottery Ticket Hypothesis: by Jonathan Frankle and Michael Carbin. ADD LINK TO PAPER. The Lottery Ticket Hypothesis builds on the idea of Network Pruning. The idea “To reduce the extent [of a neural network] by removing superfluous or unwanted parts” ADD WHERE THE QUOTE IS FROM. Network pruning is used to reduce the storage costs and computational requirements.

The basic idea of the LTH is the following. Initially, we begin with a Neural Network where each connection has been set to a random weight. We the train the Neural Network and remove the superfluous structure. Here, we focus on pruning weights: this is called sparse pruning. We look at the magnitude of the weights and we prune the weights with the lowest magnitude. We then reset the remaining weights to their initial value - or to their value at a given epoch - and we retrain the sparse subnetwork. It’s important to reset the weights to their original value or to a value they took during training and not to random values. 

This is great because we arrive at networks that are 15% to 1% of their original size. Those sub-networks require fewer iterations to learn and they match the accuracy of the original network. 

We have two loops we parallelize: we first need to study different possible thresholds for our masks (a bigger threshold means that we throw away more weights). We also need to decide on the epoch N which we will use as our baseline when we reset the weights of our subnetwork.

## Need for Big Compute

What is the need for big compute in our project?

Firstly, we fit an overparameterized architecture, which ensures tractable non-convex optimization and robustness to corruption. The architecture we chose for the initial Neural Netork is MobileNet Volume 2 by Google, as it drastically reduces the complexity and the network size in comparison to the other state of art CNN architectures. This choice will allow us for more efficient algorithm prototyping and testing.

The MobileNet Volume 2 architecture has a total of 3.4 million parameters and 300 million multiply-add operations per single forward pass. As a comparison, another popular CNN architecture, AlexNet has 60 million parameters. Although lighter than most state of art CNN architectures, it is practically infeasible to train the MobileNet on a single CPU.

To investigate the Lottery Ticket Hypothesis, we use the MobileNetV2 architecture.

- **Number of parameters**: 3.4M 
- **Number of multiply-adds (MAdds) per forward pass**: 300M

Next, we use a pruning algorithm to find effective subnetworks with a much lower parameters count. Another - and possibly more prevalent - need for big compute are the two nested for loops present in the pruning algorithm. In the outer loop, the algorithm will be iterating over the different masks (produced by the different pruning thresholds). In the inner loop, the algorithm will be iterating over the range of possible epochs which we will use as our baseline when we reset the subnetwork weights. In order to find the Lottery Ticket Hypothesis, we will iterate over the grid of threshold values and late resetting epochs and train a sparse version of the MobileNet architecture per each inner loop iteration. In order to parallelize the nested for loops we will use the Big Compute paradigms presented in class. 



## How to Use

For a complete list of instructions on how to use the programs found in this repository, please see [How To Run]() on the project website. I THINK IT WOULD BE NICE TO DO SOMETHING LIKE THIS TO.

## Table of Contents
1. [Problem Statement](ProblemStatement.md)
2. [Solution](Solution.md)
3. [Model and Data](ModelAndData.md)
4. [How To Run](HowToRun.md)
5. [Discussion](Discussion.md)