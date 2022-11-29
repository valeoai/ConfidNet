FROM ubuntu:18.04

RUN apt-get update && apt-get install -y wget bzip2 gcc g++
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py37_4.12.0-Linux-x86_64.sh -O miniconda.sh
RUN bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh
ENV PATH="/opt/conda/bin:${PATH}"
RUN conda config --set always_yes yes

RUN conda install pytorch==1.1.0 torchvision==0.3 cudatoolkit=10.0 -c pytorch
RUN pip install tensorflow==1.13.1 torchsummary pyyaml verboselogs coloredlogs click scikit-learn pillow==6.0.0 protobuf==3.20.0

COPY ./ ./ConfidNet
RUN pip install -e ./ConfidNet

# checks.
RUN python -c "import confidnet"
RUN python -c "import torch; from confidnet import structured_map_ranking_loss"

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all
WORKDIR ./ConfidNet
