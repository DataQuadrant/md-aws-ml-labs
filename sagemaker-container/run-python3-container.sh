#!/bin/bash

docker run -t \
        --name=sagemaker-notebook-container \
        --privileged \
        --restart always \
        -p 8888:8888 \
        -e AWS_PROFILE=default \
        -e JUPYTER_ENABLE_LAB=yes \
        -e GRANT_SUDO=yes \
        -v ~/.aws:/home/ec2-user/.aws:ro \
        -v ~/.ssh:/home/ec2-user/.ssh:ro \
        -v /var/run/docker.sock:/var/run/docker.sock \
        -v /Users/marian.dumitrascu/Dropbox/Learn/AWS-ML-Certification/md-aws-ml-labs:/home/ec2-user/SageMaker \
        dataquadrant/sagemaker-notebook:all-02


# Alternatively:
# docker-compose up sagemaker-notebook-container
        # -e GRANT_SUDO=yes \
        # -v /Users/marian.dumitrascu/Dropbox/Learn/AWS-ML-Certification/aws-sagem-nb-container:/home/ec2-user/SageMaker/sagemaker-notebook-container  \

