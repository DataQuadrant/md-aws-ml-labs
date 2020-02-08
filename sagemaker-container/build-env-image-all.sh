#!/bin/bash
docker build -t dataquadrant/sagemaker-notebook-base:latest base
docker push dataquadrant/sagemaker-notebook-base:latest

docker build -t dataquadrant/sagemaker-notebook:python3 -f envs/docker/Dockerfile.python3 envs
docker push dataquadrant/sagemaker-notebook:python3

docker build -t dataquadrant/sagemaker-notebook:all_python3 -f envs/docker/Dockerfile.all_python3 envs
docker push dataquadrant/sagemaker-notebook:all_python3

docker build -t dataquadrant/sagemaker-notebook:python2 -f envs/docker/Dockerfile.python2 envs
docker push dataquadrant/sagemaker-notebook:python2

docker build -t dataquadrant/sagemaker-notebook:all_python2 -f envs/docker/Dockerfile.all_python2 envs
docker push dataquadrant/sagemaker-notebook:all_python2

docker build -t dataquadrant/sagemaker-notebook:all -f envs/docker/Dockerfile.all envs
docker push dataquadrant/sagemaker-notebook:all





