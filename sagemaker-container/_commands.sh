#!/bin/bash


######################################################################################
git init
git add .
git commit -m "first commit"
git remote add origin https://github.com/mariandumitrascu/aws-sagem-nb-container.git
git push -u origin master

######################################################################################

./run-python3-container.sh