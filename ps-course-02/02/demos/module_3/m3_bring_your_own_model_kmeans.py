
m3_bring_your_own_model_kmeans

# coding: utf-8

# # Bring Your Own Model (k-means)
# _**Hosting a Pre-Trained Model in Amazon SageMaker Algorithm Containers**_
# 

# ## Import required packages
# * io   -  provides the Python interfaces to stream handling.
# * os   -  provides a portable way of using operating system dependent functionality.
# * time -  provides various time-related functions.

# In[45]:


import io
import os
import time


# ### Importing some standard python packages 
# * gzip - module provides a simple interface to compress and decompress files
# * json - exposes an API for processing json data.
# * mxnet- the mxnet python package
# * numpy- package for scientific computing with Python.
# * pickle - module implements an algorithm for serializing and de-serializing a Python object structure.
# * urllib.request  - module defines functions and classes for opening URLs.
# * sklearn.cluster - the k-means clustering algorithm from scikit

# In[46]:


import gzip
import json
import pickle
import mxnet as mx
import numpy as np
import urllib.request
import sklearn.cluster


# ### Importing amazon packages
# * boto3 - The AWS SDK for Python to write software that uses Amazon services like S3 and EC2.
# * get_execution_role - Return the role ARN whose credentials are used to call the API.

# In[47]:


import boto3
from sagemaker import get_execution_role


# **This section is only included for illustration purposes. In a real use case, you'd be bringing your model from an existing process and not need to complete these steps.**

# ## Get the pickled MNIST dataset. 
# * Check if the dataset exists on the machine on which the instance runs
# * If not, download it from the url specified.

# In[48]:


DOWNLOADED_FILENAME = 'mnist.pkl.gz'


# In[49]:


if not os.path.exists(DOWNLOADED_FILENAME):
    urllib.request.urlretrieve("http://deeplearning.net/data/mnist/mnist.pkl.gz", DOWNLOADED_FILENAME)


# ## Preprocessing and splitting the dataset 
# 
# * The pickled file represents a tuple of 3 lists : **(training set, validation set, test set)**
# * Each of the three lists is a tuple: **(list of images, list of class labels)**
# * Image: Numpy 1-dimensional array of 784 (28 x 28) float values between 0 and 1
# * Labels: Numbers between 0 and 9 indicating which digit the image represents

# In[50]:


with gzip.open('mnist.pkl.gz', 'rb') as f:
    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')


# ## Preprocessing the dataset
# * Use the sklearn.cluster.KMeans method to define a kmeans model with 10 clusters and centroids.
# * Train the model locally 
# * Convert the data to a MXNet NDArray. The model format that Amazon SageMaker's k-means container expects is an MXNet NDArray with dimensions (num_clusters, feature_dim) that contains the cluster centroids.
# * tar and gzip the model array
# * Specify the S3 bucket and prefix that you want to use for training and model data.
# * Create a bucket resource using the Bucket() method
# * Create an object resource 
# * Upload the array to s3 bucket

# In[51]:


kmeans = sklearn.cluster.KMeans(n_clusters=10).fit(train_set[0])


# In[52]:


centroids = mx.ndarray.array(kmeans.cluster_centers_)


# In[53]:


mx.ndarray.save('model_algo-1', [centroids])


# In[54]:


get_ipython().system('tar czvf model.tar.gz model_algo-1')


# In[55]:


bucket = 'loonybucket1'
prefix = 'sagemaker/kmeans_byom'


# In[56]:


s3_resource = boto3.Session().resource('s3')


# In[57]:


current_bucket=s3_resource.Bucket(bucket).Object(os.path.join(prefix, 'model.tar.gz'))


# In[58]:


current_bucket.upload_file('model.tar.gz')


# ## Hosting the model
# * generate model name
# * start a sagemaker instance
# * specify the algorithm container to use
# * get the role ARN whose credentials are used to call the API to instantiate the estimator
# * create model using the create_model method
# * setup  endpoint configuration
# * initiate the endpoint and check the status to confirm deployment

# In[59]:


kmeans_model = 'kmeans-scikit-' + time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())
kmeans_model


# In[60]:


sm = boto3.client('sagemaker')


# In[61]:


containers = {'us-west-2': '174872318107.dkr.ecr.us-west-2.amazonaws.com/kmeans:latest',
              'us-east-1': '382416733822.dkr.ecr.us-east-1.amazonaws.com/kmeans:latest',
              'us-east-2': '404615174143.dkr.ecr.us-east-2.amazonaws.com/kmeans:latest',
              'eu-west-1': '438346466558.dkr.ecr.eu-west-1.amazonaws.com/kmeans:latest'}


# In[62]:


container = containers[boto3.Session().region_name]


# In[63]:


role = get_execution_role()
role


# In[64]:


create_model_response = sm.create_model(
    ModelName=kmeans_model,
    ExecutionRoleArn=role,
    PrimaryContainer={
        'Image': container,
        'ModelDataUrl': 's3://{}/{}/model.tar.gz'.format(bucket, prefix)})


# In[65]:


create_model_response


# In[66]:


print(create_model_response['ModelArn'])


# In[67]:


kmeans_endpoint_config = 'kmeans-poc-endpoint-config-' + time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())
print(kmeans_endpoint_config)


# In[68]:


create_endpoint_config_response = sm.create_endpoint_config(
    EndpointConfigName=kmeans_endpoint_config,
    ProductionVariants=[{
        'InstanceType': 'ml.m4.xlarge',
        'InitialInstanceCount': 1,
        'ModelName': kmeans_model,
        'VariantName': 'AllTraffic'}])


# In[69]:


print("Endpoint Config Arn: " + create_endpoint_config_response['EndpointConfigArn'])


# In[70]:


kmeans_endpoint = 'kmeans-poc-endpoint-' + time.strftime("%Y%m%d%H%M", time.gmtime())
print(kmeans_endpoint)


# In[71]:


create_endpoint_response = sm.create_endpoint(
    EndpointName=kmeans_endpoint,
    EndpointConfigName=kmeans_endpoint_config)


# In[72]:


print(create_endpoint_response['EndpointArn'])


# In[73]:


sm.get_waiter('endpoint_in_service').wait(EndpointName=kmeans_endpoint)
resp = sm.describe_endpoint(EndpointName=kmeans_endpoint)


# In[74]:


status = resp['EndpointStatus']
print("Arn: " + resp['EndpointArn'])
print("Status: " + status)


# ## Validate the model
# * define a method to get csv records from the training set, the model endpoint requires data in CSV format
# * take the first 100 records from our training dataset to score them using our hosted endpoint
# * instantiate a runtime session
# * score the records from training set in the endpoint
# * compare to the model labels from k-means example.

# In[75]:


def np2csv(arr):
    csv = io.BytesIO()
    np.savetxt(csv, arr, delimiter=',', fmt='%g')
    return csv.getvalue().decode().rstrip()


# In[76]:


train_set[0][0:100].shape


# In[77]:


payload = np2csv(train_set[0][0:100])


# In[78]:


payload


# In[79]:


runtime = boto3.Session().client('runtime.sagemaker')


# In[80]:


response = runtime.invoke_endpoint(EndpointName=kmeans_endpoint,
                                   ContentType='text/csv',
                                   Body=payload)


# In[81]:


result = json.loads(response['Body'].read().decode())


# In[82]:


result


# In[83]:


scored_labels = np.array([r['closest_cluster'] for r in result['predictions']])


# In[84]:


scored_labels


# In[85]:


scored_labels == kmeans.labels_[0:100]


# ## Remove endpoint to avoid stray charges

# In[86]:


sm.delete_endpoint(EndpointName=kmeans_endpoint)

