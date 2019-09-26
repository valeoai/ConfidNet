FROM ubuntu:18.04

RUN apt-get update && apt-get install -y wget bzip2 gcc g++
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
RUN bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh
ENV PATH="/opt/conda/bin:${PATH}"
RUN conda config --set always_yes yes

RUN conda install pytorch=1.0.1 torchvision cudatoolkit=10.0 -c pytorch
RUN pip install tensorflow torchsummary pyyaml verboselogs coloredlogs future

COPY ./ ./ConfidNet
RUN pip install -e ./ConfidNet

WORKDIR ./ConfidNet

RUN python -c "import confidnet"
RUN python -c "from confidnet import structured_map_ranking_loss"
