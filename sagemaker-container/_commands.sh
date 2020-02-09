#!/bin/bash


######################################################################################
git init
git add .
git commit -m "first commit"
git remote add origin https://github.com/mariandumitrascu/aws-sagem-nb-container.git
git push -u origin master

######################################################################################

./run-python3-container.sh


# commands to fix ec2-user to run docker
# reference:
# https://www.digitalocean.com/community/questions/how-to-fix-docker-got-permission-denied-while-trying-to-connect-to-the-docker-daemon-socket

USER=ec2-user
sudo groupadd docker
sudo usermod -aG docker ${USER}
# re-login
sudo chown "$USER":"$USER" /home/"$USER"/.docker -R
sudo chmod g+rwx "$HOME/.docker" -R
sudo chmod 666 /var/run/docker.sock

