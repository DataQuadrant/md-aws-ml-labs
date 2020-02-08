
# coding: utf-8

# # Part 1: Packaging your Algorithm 
# 
# Amazon SageMaker allows you to package your own algorithms and train and host it on Sagemaker. Here we package a scikit-learn implementation of decision trees for use with SageMaker.

# ### Contents of the container
# * **Dockerfile** describes how to build your Docker container image. 
# * **build_and_push.sh** script uses the Dockerfile to build your container images and then pushes it to ECR.
# * **decision_trees** contains the files that will be installed in the container.
# * **local_test** shows how to test your new container on any computer that can run Docker.

# ### The files in the container are:
# 
# * **nginx.conf**,the configuration file for the nginx front-end.
# * **predictor.py**,program implements the Flask web server and the decision tree predictions for this app.
# * **serve** program starts when the container is started for hosting, launches the gunicorn server which runs multiple instances of the Flask app defined in predictor.py. 
# * **train**, program that is invoked when the container is run for training.
# * SageMaker will look to run an executable program named "train" for training and "serve" for hosting.
# * Or you can specify any ENTRYPOINT in your Dockerfile which has train() and serve() functions defined within.
# * **wsgi.py**,a small wrapper used to invoke the Flask app.

# In[ ]:


get_ipython().system('cat container/Dockerfile')


# ## Building and registering the container
# * Build the container image using docker build 
# * Push the container image to ECR using docker push. 
# * Get the region defined in the current configuration (default to us-west-2 if none defined)
# * Looks for an ECR repository in the current account and current default region. If the repository doesn't exist, the script will create it.
# * Get the login command from ECR and execute it directly
# * Build the docker image locally with the image name 
# * Push it to ECR with the full name.
# * On a SageMaker Notebook Instance, the docker daemon may need to be restarted in order to detect your network configuration correctly.(This is a known issue.)

# In[ ]:


get_ipython().run_cell_magic('sh', '', '\nalgorithm_name=decision-trees-sample\n\ncd container\n\nchmod +x decision_trees/train\nchmod +x decision_trees/serve\n\naccount=$(aws sts get-caller-identity --query Account --output text)\n\nregion=$(aws configure get region)\nregion=${region:-us-west-2}\n\nfullname="${account}.dkr.ecr.${region}.amazonaws.com/${algorithm_name}:latest"\n\naws ecr describe-repositories --repository-names "${algorithm_name}" > /dev/null 2>&1\n\nif [ $? -ne 0 ]\nthen\n    aws ecr create-repository --repository-name "${algorithm_name}" > /dev/null\nfi\n\n$(aws ecr get-login --region ${region} --no-include-email)\n\nif [ -d "/home/ec2-user/SageMaker" ]; then\n  sudo service docker restart\nfi\n\ndocker build  -t ${algorithm_name} .\ndocker tag ${algorithm_name} ${fullname}\n\ndocker push ${fullname}')


# # Part 2: Training and Hosting the Algorithm
# 
# Once you have your container packaged, you can use it to train and serve models. 

# ### Import packages 
# * os -  provides a portable way of using operating system dependent functionality.
# * gmtime - Convert a time expressed in seconds since the epoch to a struct_time in UTC.
# * strftime - Convert a tuple or struct_time representing a time as returned by gmtime()

# In[ ]:


import os
from time import gmtime, strftime


# ### Importing some standard python packages 
# * csv- module provides objects to read and write sequences.
# * itertools - module implements a number of iterator building blocks.
# * numpy  - package for scientific computing with Python.
# * pandas - library providing data structures and data analysis tools for Python.

# In[ ]:


import csv
import itertools
import numpy as np
import pandas as pd


# ### Importing amazon packages
# * boto3 - The AWS SDK for Python to write software that uses Amazon services like S3 and EC2.
# * psycopg2 - popular PostgreSQL database adapter for the Python
# * sagemaker - Python SDK for training and deploying machine learning models on Amazon SageMaker.
# * get_execution_role - Return the role ARN whose credentials are used to call the API.
# * csv_serializer - Defines csv as the behavior for serialization of input data.

# In[ ]:


import boto3
import psycopg2
import sagemaker as sage
from sagemaker.predictor import csv_serializer
from sagemaker import get_execution_role


# In[ ]:


con=psycopg2.connect(dbname= 'loonydb1', host='myloony-db.c680xdlyp4bo.us-east-1.redshift.amazonaws.com',
port= '5439', user= 'masteruser', password= 'Password123')


# In[ ]:


cur = con.cursor()


# In[ ]:


query="select * from public.irisdata ;"


# In[ ]:


cur.execute(query)


# In[ ]:


results = cur.fetchall()


# In[ ]:


fp = open('iris.csv','w')


# In[ ]:


c= csv.writer(fp, lineterminator='\n') 


# In[ ]:


for row in results:
    print (row)
    c.writerow(row)


# In[ ]:


fp.close()


# ## Upload the data for training
# * Set the bucket path
# * Create a sagemaker session
# * Create a bucket and upload the training data.

# In[ ]:


prefix = 'scikit-byoc'


# In[ ]:


sess = sage.Session()


# In[ ]:


data_location = sess.upload_data('iris.csv', key_prefix=prefix)


# ## Hosting the model 
# * Get the account and region information
# * Get the conatiner image
# * Get the IAM role credentials
# * Instantiate an estimator
# * Invoke the fit method to train the model
# * Deploy the model

# In[ ]:


account = sess.boto_session.client('sts').get_caller_identity()['Account']
account


# In[ ]:


region = sess.boto_session.region_name
region


# In[ ]:


image = '{}.dkr.ecr.{}.amazonaws.com/decision-trees-sample'.format(account, region)


# In[ ]:


role = get_execution_role()
role


# In[ ]:


tree = sage.estimator.Estimator(image,
                       role, 1, 'ml.c4.2xlarge',
                       output_path="s3://{}/output".format(sess.default_bucket()),
                       sagemaker_session=sess)


# In[ ]:


tree.fit(data_location)


# In[ ]:


predictor = tree.deploy(1, 'ml.m4.xlarge', serializer=csv_serializer)


# ## Validate the model
# * extract some of the data we used for training
# * pass in the data to the predictor object

# In[ ]:


shape=pd.read_csv("iris.csv", header=None)


# In[ ]:


df = shape[50:110]


# In[ ]:


names = df[0].values.T.tolist()
names


# In[ ]:


test_X =df.drop(df.columns[0], axis=1) 


# In[ ]:


test_X


# In[ ]:


results = predictor.predict(test_X.values).decode('utf-8')
results=results.split()


# In[ ]:


print(results)


# In[ ]:


print (np.array(names) == np.array(results))


# ## Delete endpoint

# In[ ]:


sess.delete_endpoint(predictor.endpoint)

