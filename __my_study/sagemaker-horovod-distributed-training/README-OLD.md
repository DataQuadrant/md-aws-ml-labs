## Sagemaker Horovod Distributed Training

### Distributed training with SageMaker's script mode using Horovod distributed deep learning framework

This lab demonstrates two concepts on a simple MNIST dataset and a keras-based deep learning framework:
- SageMaker distributed training with Horovod framework
- SageMaker "script mode" which allows the keras train_mnist_hvd.py python script to be the entry point to SageMaker train() API. This approach makes it possible not to rebuild the training container with every change made to train_mnist_hvd.py

#### A. SageMaker's Horovod Distributed Training Framework 

As you know, SageMaker built-in training algorithms already provide ability to distribute workload among multiple compute nodes via 'train_instance_count' parameter. Up until now, if one were to bring his/her own algorithm, they would have to take care of providing their own disributed compute framework and encapsulating it in a container. Horovod has previously been enabled on Amazon Deep Learning AMIs 
(https://aws.amazon.com/blogs/machine-learning/aws-deep-learning-amis-now-include-horovod-for-faster-multi-gpu-tensorflow-training-on-amazon-ec2-p3-instances/). 
With the introduction of the new feature described in ths lab, Horovod distributed framework is available as part of SageMaker BringYourOwnContainer flow. Customers will soon also be able to run fully-managed Horovod jobs for high scale distributed training. 

What is Horovod? It is a framework allowing a user to distribute a deep learning workload among multiple compute nodes and take advantage of inherent parallelism of deep learning training process. It is available for both CPU and GPU AWS compute instances. Horovod follows the Message Passing Interface (MPI) model. This is a popular standard for passing messages and managing communication between nodes in a high-performance distributed computing environment. Horovod’s MPI implementation provides a more simplified programming model compared to the parameter server based distributed training model. This model enables developers to easily scale their existing single CPU/GPU training programs with minimal code changes.

For this lab, we will be instantiating CPU compute nodes for simplicity and scalability. 

#### B. SageMaker's Script Mode.
Previously (as in Lab 2-4 of this workshop), in BringYourOwnContainer situation, a user had to make his/her training Python script a part of the container. Therefore, during the debug process, every Python script change required rebuilding the container. SageMaker's "script mode" allows one to build the container once and then debug and change a python script  without rebuilding the container with every change. Instead, a user specifies script's "entry point" via 'train(script="myscript.py",....) parameter, for example:
```

train(horovod_train_script = "train_mnist_hvd.py",
      instance_count = 12,
      num_of_processes_per_host = 1)
```

#### File Structure

```buildoutcfg

bin
 |- docker-build.sh: Script to build docker file
 |- push_image.sh: Script to push socker to ECR repository.
docker
 |- Dockerfile.cpu: Docker file to build BYOC container.
 |- resources: Resource files required to build docker.
src
 |- data/: traning data
 |- horovod_launcher.py: Laucnher script to launch horovod training script.
 |- train_mnist_hvd.py: Mnist horovod training script 
notebooks
 |- Tensorflow Distributed Training - Horovod-BYOC-Example.ipynb: Sample notebook to launch the horovod distributed training example. 
```

#### Steps for launching Jupyter Notebook:
- Navigate the above file structure to the notebook in 'notebooks' directory
- If prompted, select Jupyter kernel conda_tensorflow_p36
- Launch and execute the notebook. 
Note that depending on your choice of the host machine, it may take as long as 10 min to build the container the 1st time out. 

## License

This library is licensed under the Apache 2.0 License. 
