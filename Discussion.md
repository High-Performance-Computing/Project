# Discussion: Next steps

## Next avenues in order to speed up our training

- It would be interesing to work with more recent NVIDIA GPUs (compute capability >= 7.0). The GPUs used in the project are Nvidia Tesla K80 and can't benefit from using Tensorflow Mixed Precision policy: expected speed-up x3 (experiment to be tried on AWS).
- Use multi-node distributed training frameworks in order to speed up the training: <a href="https://github.com/horovod/horovod/blob/master/docs/keras.rst">Horovod</a>
- Modify the architecture of the Neural Networks in place. This requires deleting some weights in the architecture and not some neurons, which might be very difficult.  
- If we scale up our modela nd can't reach 100% GPU occupancy, use Elephas Model and scale up the problem using a Spark Cluster. 

## Considerations for the LTH
- Check the range of transfer learning performances of our Winning Ticket: is it working better than another simple Transfer Learning setting ? For instance, what would happen if we use it on the CelebA dataset ? 



