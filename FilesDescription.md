# Description of our .py, .sh and .txt files

## Environment setup 

requirements.txt: the different dependices needed to run our code
mask.prof: result of the callback profiling
test folder: you will find some functions there in order to test your environment

## Training the Initial Model

spark_processing.py: performs an offline preprocessing step in order to efficiently resize the images to the same different shapes.

loading_imagenet.py: loads and processes the ImageNet dataset in order to feed it to the MobileNet V2 architecture. If you want to use the uniform-sized dataset, you should set the boolean Spark to True. This will speed up the training. 

train_model.py: trains the MobileNetV2 architecture and saves the weights at every epoch (for late resetting). Please use run_script.sh in order to avoid small issues. 

HP_tuning.py: this script requires you to create a wandb account an initialize a sweep https://docs.wandb.ai/guides/sweeps. Once your agent is setup, please update accordingly the bash script run_hp.sh

## IMP 

masking.py: allows to perform the IMP algorithm, loading the weights from different epochs and performing thresholding on them, and then retraining the entire model. Use run_imp.sh

transfer_learning.py: allows to check the effectiveness of our winning ticket on the CIFAR 100 dataset after only some fine-tuning. 



