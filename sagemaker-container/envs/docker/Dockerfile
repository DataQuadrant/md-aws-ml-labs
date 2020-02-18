# FROM dataquadrant/sagemaker-notebook:all_python2

FROM dataquadrant/sagemaker-notebook-base 
LABEL maintainer="Marian Dumitrascu <marian@dataquadrant.com>"

# ==================================================================
# Create environment(s)
# ------------------------------------------------------------------

USER $NB_UID

COPY . $HOME/envs/

# Fix permissions on $HOME/envs/ as root
USER root
RUN fix-permissions $HOME/envs/

USER $NB_UID

RUN	    jupyter labextension install @jupyter-widgets/jupyterlab-manager -y --no-build && \
        jupyter labextension install @lckr/jupyterlab_variableinspector -y --no-build && \
        jupyter labextension install @krassowski/jupyterlab_go_to_definition -y --no-build && \
        jupyter labextension install @jupyterlab/dataregistry-extension -y --no-build
# since we installed all extensions with --no-build option, now build:
RUN     jupyter lab build



RUN $HOME/envs/create_env_file.sh python2 && \
    $HOME/envs/create_env.sh python2 && \ 
    $HOME/envs/create_env_file.sh python3 && \
    $HOME/envs/create_env.sh python3 && \
    $HOME/envs/create_env_file.sh chainer_p36 && \
    $HOME/envs/create_env.sh chainer_p36 && \
    $HOME/envs/create_env_file.sh mxnet_p36 && \
    $HOME/envs/create_env.sh mxnet_p36 && \
    $HOME/envs/create_env_file.sh pytorch_p36 && \
    $HOME/envs/create_env.sh pytorch_p36 && \
    $HOME/envs/create_env_file.sh tensorflow_p36 && \
    $HOME/envs/create_env.sh tensorflow_p36

RUN $HOME/envs/create_env_file.sh python3 && \
    $HOME/envs/create_env.sh python3
    
USER $NB_UID



