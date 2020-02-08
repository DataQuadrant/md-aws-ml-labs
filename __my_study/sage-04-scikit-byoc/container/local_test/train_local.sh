#!/bin/sh

# image=$1
image=sagemaker-decision-trees

mkdir -p test_dir/model
mkdir -p test_dir/output

rm test_dir/model/*
rm test_dir/output/*

# docker run -v $(pwd)/test_dir:/opt/ml --rm ${image} train
docker run -v /Users/marian.dumitrascu/Dropbox/Learn/AWS-ML-Certification/md-aws-ml-labs/00-my-track/sage-04-scikit-byoc/container/local_test/test_dir:/opt/ml --rm ${image} train
