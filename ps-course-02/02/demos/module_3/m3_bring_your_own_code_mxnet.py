m3_bring_your_own_code_mxnet
# coding: utf-8

# # SageMaker Models using the Apache MXNet Module API
# 

# ### Import Ipython package
# * IPython.display - Public API for display tools in IPython.

# In[44]:


from IPython.display import HTML


# ### Importing amazon packages
# * boto3 - The AWS SDK for Python to write software that uses Amazon services like S3 and EC2.
# * sagemaker - Python SDK for training and deploying machine learning models on Amazon SageMaker.
# * sagemaker.mxnet - the Amazon sagemaker custom Apache MXNet code.
# * get_execution_role - Return the role ARN whose credentials are used to call the API.

# In[34]:


import boto3
import sagemaker
from sagemaker import get_execution_role
from sagemaker.mxnet import MXNet


# ### The training script
# 
# Your python script should implement a few methods like train, model_fn, transform_fn, input_fn etc. SagaMaker will call appropriate method when needed. https://docs.aws.amazon.com/sagemaker/latest/dg/mxnet-training-inference-code-template.html

# In[35]:


get_ipython().system('cat mnist.py')


# ### Set up for model training
# * Set the bucket location to save your custom code.
# * Set the bucket location to save trained model.
# * Get the path to train and test data.
# * Get the role ARN whose credentials are used to call the API to instantiate the estimator.
# * Get the region where the model willl be hosted

# In[36]:


region = boto3.Session().region_name
region


# In[37]:


custom_code_upload_location = 's3://loonybucket1/sagemaker/customcode/mxnet'


# In[38]:


model_artifacts_location = 's3://loonybucket1/sagemaker/artifacts'


# In[39]:


train_data_location = 's3://sagemaker-sample-data-{}/mxnet/mnist/train'.format(region)
test_data_location = 's3://sagemaker-sample-data-{}/mxnet/mnist/test'.format(region)


# In[40]:


role = get_execution_role()
role


# ### Training the model
# * Instantiate an estimator object and pass in the code as the entry point parameter.
# * Train and deploy the model

# In[41]:


mnist_estimator = MXNet(entry_point='mnist.py',
                        role=role,
                        output_path=model_artifacts_location,
                        code_location=custom_code_upload_location,
                        train_instance_count=1, 
                        train_instance_type='ml.m4.xlarge',
                        hyperparameters={'learning_rate': 0.1})


# In[42]:


mnist_estimator.fit({'train': train_data_location, 'test': test_data_location})


# In[43]:


predictor = mnist_estimator.deploy(initial_instance_count=1,
                                   instance_type='ml.m4.xlarge')


# ## Validating the model
# * Invoke the html script to read in an input. The pixel data from your drawing will be loaded into a data variable in this notebook.
# * Using the predictor object to classify the handwritten digit.
# * Raw predictions and Labelled predictions display the probabilities of the digit being each of the defined labels.
# * Most likely answer prints the label with the maximum probability. 

# In[76]:


HTML(open("input.html").read())


# In[77]:


print(data)


# In[78]:


type(data)


# In[79]:


len(data)


# In[80]:


print(len(data[0][0]))
print(len(data[0][1]))
print(len(data[0][2]))

print(len(data[0][27]))


# In[81]:


response = predictor.predict(data)
print('Raw prediction result:')
response


# In[82]:


labeled_predictions = list(zip(range(10), response[0]))
print('Labeled predictions: ')
labeled_predictions


# In[83]:


labeled_predictions.sort(key=lambda label_and_prob: 1.0 - label_and_prob[1])
print('Most likely answer: {}'.format(labeled_predictions[0]))


# ## Delete the prediction endpoint
# 

# In[84]:


sagemaker.Session().delete_endpoint(predictor.endpoint)

