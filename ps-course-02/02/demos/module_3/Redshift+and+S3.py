
# coding: utf-8
myloony-db.c680xdlyp4bo.us-east-1.redshift.amazonaws.com:5439
# In[1]:


import psycopg2
import boto3
import os
import csv


# In[2]:


con=psycopg2.connect(dbname= 'loonydb1', host='',
port= '5439', user= 'masteruser', password= 'Password123')


# In[3]:


cur = con.cursor()


# In[4]:


query1 = "create table public.irisdata( classname varchar(30),sepal_length float,sepal_width float,petal_length float,petal_width float);"


# In[5]:


cur.execute(query1)


# In[24]:


query2 = "copy public.irisdata from 's3://sample-data-bucket01/iris.csv'credentials 'aws_iam_role=arn:aws:iam::324118574079:role/NewRedshiftRole' delimiter ',' region 'us-east-1';"


# In[25]:


cur.execute(query3)


# In[ ]:


query3= "select * from public.irisdata;"


# In[33]:


con.commit()


# In[34]:


cur.close()


# In[35]:


con.close()

