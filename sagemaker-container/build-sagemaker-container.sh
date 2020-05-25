#!/bin/bash


# build the base
docker build -t dataquadrant/sagemaker-notebook-base:latest base

# build sagemaker container with all on it
docker build -t dataquadrant/sagemaker-notebook:allall -f envs/docker/Dockerfile envs
docker push dataquadrant/sagemaker-notebook:allall

docker build -t dataquadrant/sagemaker-notebook:allall -f envs/docker/Dockerfile envs

