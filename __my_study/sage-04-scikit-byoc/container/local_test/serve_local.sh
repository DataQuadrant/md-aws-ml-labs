#!/bin/sh

# image=$1
image=sagemaker-decision-trees

# docker run -v $(pwd)/test_dir:/opt/ml -p 8080:8080 --rm ${image} serve
docker run -v /Users/marian.dumitrascu/Dropbox/Learn/AWS-ML-Certification/md-aws-ml-labs/00-my-track/sage-04-scikit-byoc/container/local_test/test_dir:/opt/ml -p 8080:8080 --rm ${image} serve
