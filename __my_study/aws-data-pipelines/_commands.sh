#!/bin/bash

aws datapipeline list-runs  --region us-east-1 --pipeline-id df-01168623H98EIK1EA1GZ

aws datapipeline list-runs --region us-east-1 --pipeline-id df-01168623H98EIK1EA1GZ --status finished

# retrieve a pipeline definition to be edited locally
aws datapipeline get-pipeline-definition --pipeline-id df-01168623H98EIK1EA1GZ --region us-east-1 > pipeline-definition-02.json

# update pipeline definition after editing locally
aws datapipeline put-pipeline-definition --region us-east-1 --pipeline-id df-01168623H98EIK1EA1GZ --pipeline-definition ./pipeline-definition-01.json

# activate the pipeline
aws datapipeline activate-pipeline --region us-east-1 --pipeline-id df-01168623H98EIK1EA1GZ # --start-timestamp YYYY-MM-DDTHH:MM:SSZ

# step entry for EmrActivity
# /home/hadoop/contrib/streaming/hadoop-streaming.jar,-input,s3n://elasticmapreduce/samples/wordcount/input,-output,s3://eap-aabg-s3-landingzone-02/wordcount/output/#{@scheduledStartTime},-mapper,s3n://elasticmapreduce/samples/wordcount/wordSplitter.py,-reducer,aggregate