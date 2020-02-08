
# coding: utf-8

# # Distributed training  

# ### Import packages 
# * os -  provides a portable way of using operating system dependent functionality.

# In[1]:


import os


# ### Importing standard python packages
# * utils - a collection of small Python functions and classes which make common patterns shorter and easier.
# * numpy - package for scientific computing with Python.

# In[2]:


import utils
import numpy as np


# ### Importing tensorflow packages
# * tensorflow - library for dataflow programming across a range of tasks.
# * tensorflow.contrib.learn.python.learn.datasets - contains dataset utilities and synthetic/reference datasets, for getting the mnist dataset

# In[3]:


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.learn.python.learn.datasets import mnist


# ### Importing amazon packages
# * sagemaker - Python SDK for training and deploying machine learning models on Amazon SageMaker.
# * get_execution_role - Return the role ARN whose credentials are used to call the API.
# * sagemaker.tensorflow - The Amazon SageMaker custom TensorFlow code. 

# In[4]:


import sagemaker
from sagemaker import get_execution_role
from sagemaker.tensorflow import TensorFlow


# ## Getting and preprocessing the dataset
# * Read the mnist dataset
# * split it into three : train, validation and test
# * instantiate a sagemaker session
# * upload our datasets to an S3 location.

# In[5]:


data_sets = mnist.read_data_sets('data', dtype=tf.uint8, reshape=False, validation_size=5000)


# In[6]:


utils.convert_to(data_sets.train, 'train', 'data')
utils.convert_to(data_sets.validation, 'validation', 'data')
utils.convert_to(data_sets.test, 'test', 'data')


# In[7]:


sagemaker_session = sagemaker.Session()


# In[8]:


inputs = sagemaker_session.upload_data(path='data', key_prefix='data/mnist')


# ## Training the model
# * You will need the training script predefined.
# * Define a tensorflow estimator object and pass in the python script as the entry point parameter.
# * Get the role ARN whose credentials are used to call the API.
# * To perform distributed training,the instance count is set to 2. 
# * Invoke the fit method to train the model. The fit method will create a training job in two ml.c4.xlarge instances. The logs will show the instances doing training, evaluation, and incrementing the number of training steps.
# * Invoke the deploy method to create an endpoint.
# 
# TODO: Judy once again could you put in all the methods that Sage maker uses in here and define each

# In[9]:


get_ipython().system("cat 'mnist.py'")


# In[10]:


role = get_execution_role()
role


# In[11]:


mnist_estimator = TensorFlow(entry_point='mnist.py',
                             role=role,
                             training_steps=1000, 
                             evaluation_steps=100,
                             train_instance_count=2,
                             train_instance_type='ml.c4.xlarge')


# In[ ]:


mnist_estimator.fit(inputs)


# In[20]:


mnist_predictor = mnist_estimator.deploy(initial_instance_count=1,
                                             instance_type='ml.m4.xlarge')


# ## Validating the model
# * get some data for testing
# * call the predictor to compare the labels from test data and the predicted labels

# In[23]:


mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


# In[24]:


for i in range(10):
    data = mnist.test.images[i].tolist()
    tensor_proto = tf.make_tensor_proto(values=np.asarray(data), shape=[1, len(data)], dtype=tf.float32)
    predict_response = mnist_predictor.predict(tensor_proto)
    
    print("========================================")
    label = np.argmax(mnist.test.labels[i])
    print("label is {}".format(label))
    prediction = predict_response['outputs']['classes']['int64Val'][0]
    print("prediction is {}".format(prediction))


# ## Delete the endpoint

# In[25]:


sagemaker_session.delete_endpoint(mnist_predictor.endpoint)

