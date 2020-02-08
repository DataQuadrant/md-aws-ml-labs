#!/bin/bash
# Sample builds, for example:
#   ./build-env-image.sh all-python3

# export PREFIX="custom"

python3() {
    # docker build -t dataquadrant/sagemaker-notebook:${PREFIX}-python3 -f envs/docker/Dockerfile.python3 envs
    docker build -t dataquadrant/sagemaker-notebook:python3 -f envs/docker/Dockerfile.python3 envs
}

tensorflow-p36() {
    # docker build -t dataquadrant/sagemaker-notebook:${PREFIX}-tensorflow-p36 -f envs/docker/Dockerfile.tensorflow_p36 envs
    docker build -t dataquadrant/sagemaker-notebook:tensorflow-p36 -f envs/docker/Dockerfile.tensorflow_p36 envs
}

python3-tensorflow-p36() {
    # docker build -t dataquadrant/sagemaker-notebook:${PREFIX}-python3-tensorflow-p36 -f envs/docker/Dockerfile.python3_tensorflow_p36 envs
    docker build -t dataquadrant/sagemaker-notebook:python3-tensorflow-p36 -f envs/docker/Dockerfile.python3_tensorflow_p36 envs
}

all-python3() {
    # docker build -t dataquadrant/sagemaker-notebook:${PREFIX}-all-python3 -f envs/docker/Dockerfile.all_python3 envs
    docker build -t dataquadrant/sagemaker-notebook:all-python3 -f envs/docker/Dockerfile.all_python3 envs
}

$1