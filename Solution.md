# Solution


## Type of application: Big Compute

**MobileNetV2 Architecture**: 
- Total number of parameters: 3.4M 
- Number of multiply-adds (MAdds) per forward pass: 300M
- **Solution**: GPU Accelerated Computing (4 GPUs single worker node)

**Two Nested For Loops to Find the Lottery Ticket Hypothesis**
- Outer for loop: iterate over different masks (pruning thresholds)
- Inner for loop: iterate over the range of latte resetting epochs
Train a sparse MobileNetV2 CNN per each inner loop iteration
Solution: MPI Distributed Computing (20 worker nodes)

## Type of application: Big Data

ImageNet Dataset: 
- Total number of training images: 1.23M  
- Total number of validation images: 100k
- Total number of test images: 50k
- ImageNet Dataset size: 157.3 GB
- Average image resolution (downloaded): 469x387
- Average image resolution (preprocessed): 227x227
- Solution: PySpark API Dataflow Parallelization (download and process ImageNet)
- Solution: Performance Optimization (caching, prefetching)
- Solution: Keras Extension Elephas Data-parallel Training

## Programming model and infrastructure

- We use FAS RC (take advantage of the SCRATCH space [300+GB] and the ease of allocating several nodes for MPI). 
-Python 3.8.5, mpi4py 3.0.3, pyspark 3.1.1
- We used Spark to download the data [working closely with FAS in order to devise the right SLURM allocations for the different workers to access the GPUs safely]
- We use MPI for communication between our nodes and Python Multiprocessing for parallelization within a node 
- Train using TensorFlow 2.0 (leveraging cuda and cudnn) and Elephas (PySpark module) in order to accelerate batch training 
- Objective: End solution comprises 20 worker nodes, each one will have 4 GPUs TESLA K80 with 11.5 GB memory and 64 CPUs 

## Profiling and training MobilenetV2


<p align="justify"> Empirically, for a batch size of 96,  we went down 20h per epoch on a single CPU, to  3h30 per epoch using one GPU, to 1h per epoch using 4 GPUs. The theoretical speed up of passing from one to four GPUs is 4, but the effective speed-up was 3.5 due to communication overhead between CPU and GPU. </p> 

<p align="justify"> But the preprocessing of our data meant the GPUs could not access the data efficiently so the GPU occupation was low. </p>

<p align="justify"> We went down to 15 min per epoch by preprocessing the data (GPU occupation: 50%). We could expect a 1X to 2X speed-up by further augmenting GPU occupation. </p>

### Summary

- Running on a single CPU: 20hrs/epoch
- Running on a single Tesla K80 GPU: 3h30/epoch
- Running on 4 Tesla K80 GPUs: 1h/epoch
- Identifying the bottleneck: slow data pipeline



## Main Overheads

### Communication: 

- Performance of GPU applications can be bottlenecked by data transfers between the CPU nodes and GPU. It limits the peak throughput that can be obtained from these memory spaces
- **Solution:** Caching & Prefetching in order to accelerate data transfers between CPU & GPU

### Data processing: 

- The CPU takes some time to feed the data to the GPU. 
- **Solution:** Parallelization of the data pipeline using 144 different workers
- **Solution:** Vectorization of the pipeline function using Batching
 
After parallelization of the data pipeline, down to 15 mins/epoch 
Next step to reach 100% GPU occupation: Offline processing of the data using Spark

Synchronization: We structured our architecture in order for different nodes to be independent

##### FAS RC

We used 20 nodes with 4 GPUs per node on FAS RC.


## Training

- We save the weights at initialization. 
- We save the weights at the final step of training
- We define a grid on the epochs for which we want to perform late resetting and save the weights during the training at every one of these epochs

Once this is done, ie we have initial weights + final weights + weights at the treillis of epochs, we can start IMP:
Define a grid of thresholds on the magnitude of the final weights
Compute the mask for every one of these thresholds [1 for loop]
For every masked network, retrain from every selected epoch [1 for loop]
These two for loops are where the parallelization occurs: this is where we will hopefully leverage MPI/OpenMP/Python multiprocessing. 







## Structure from last year group


## Our Approach
 

### Levels of Parallelism

### Types of Parallelism within Application

### Programming Models

### Infrastructure

