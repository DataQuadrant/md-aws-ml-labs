#!/bin/bash

docker run -itd \
        --name=sagemaker-nb  \
        --hostname=sagemaker-nb \
        --privileged \
        --restart always \
        -p 8888:8888 \
        -p 4040:4040 \
        -e AWS_PROFILE=default \
        -e JUPYTER_ENABLE_LAB=yes \
        -e GRANT_SUDO=yes \
        --user root \
        -v ~/.aws:/home/ec2-user/.aws:ro \
        -v ~/.docker:/home/ec2-user/.docker:rw \
        -v ~/.kube:/home/ec2-user/.kube:rw \
        -v /var/run/docker.sock:/var/run/docker.sock \
        -v /Users/marian.dumitrascu/Dropbox/Learn/AWS-ML-Certification/md-aws-ml-labs:/home/ec2-user/SageMaker \
        -v /Users/marian.dumitrascu/Dropbox/Work/current/VISTRA/weather_history:/home/ec2-user/SageMaker/weather-history \
        dataquadrant/sagemaker-notebook:allall_w_pwd start-notebook.sh --NotebookApp.password='sha1:1557010d897e:2ce4af9f0879352b4d76e179b8b703015b2a941f'
        # dataquadrant/sagemaker-notebook:all-02


# Alternatively:
# docker-compose up sagemaker-notebook-container
        # -e GRANT_SUDO=yes \
        # -v /Users/marian.dumitrascu/Dropbox/Learn/AWS-ML-Certification/aws-sagem-nb-container:/home/ec2-user/SageMaker/sagemaker-notebook-container  \

        # -v ~/.ssh:/home/ec2-user/.ssh:ro \

# "password": "sha1:1557010d897e:2ce4af9f0879352b4d76e179b8b703015b2a941f"