# Discussion: Next steps

## Next avenues in order to speed up our training

-  It would be interesting to work with more recent NVIDIA GPUs (compute capability >= 7.0). The GPUs used in the project are Nvidia Tesla K80 and can't benefit from using TensorFlow Mixed Precision policy. The xpected speed-up is x3 (those experiments are to be tried on AWS). 
- Use multi-node distributed training frameworks in order to speed up the training: <a href="https://github.com/horovod/horovod/blob/master/docs/keras.rst">Horovod</a>
- Modify the architecture of the Neural Networks in place. This requires deleting some weights in the architecture and not some neurons, which might be very difficult.  
- If we scale up our model, nd can't reach 100% GPU occupancy. The solution would be to use Elephas Model and scale up the problem using a Spark Cluster. 

## Considerations for the LTH
- It would be interesting to check the range of transfer learning performances of our winning ticket: does it work better than another simple Transfer Learning setting ? For instance, what would happen if we use it on the <a href="http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html">CelebA dataset </a> ? 



