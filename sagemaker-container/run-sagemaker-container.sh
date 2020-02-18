#!/bin/bash

docker run -it \
        --name=sagemaker-nb  \
        --hostname=sagemaker-nb \
        --privileged \
        --restart always \
        -p 8888:8888 \
        -e AWS_PROFILE=default \
        -e JUPYTER_ENABLE_LAB=yes \
        -e GRANT_SUDO=yes \
        -v ~/.aws:/home/ec2-user/.aws:ro \
        -v ~/.docker:/home/ec2-user/.docker:rw \
        -v ~/.kube:/home/ec2-user/.kube:rw \
        -v /var/run/docker.sock:/var/run/docker.sock \
        -v /Users/marian.dumitrascu/Dropbox/Learn/AWS-ML-Certification/md-aws-ml-labs:/home/ec2-user/SageMaker \
        dataquadrant/sagemaker-notebook:allall
        # dataquadrant/sagemaker-notebook:all-02


# Alternatively:
# docker-compose up sagemaker-notebook-container
        # -e GRANT_SUDO=yes \
        # -v /Users/marian.dumitrascu/Dropbox/Learn/AWS-ML-Certification/aws-sagem-nb-container:/home/ec2-user/SageMaker/sagemaker-notebook-container  \

        # -v ~/.ssh:/home/ec2-user/.ssh:ro \