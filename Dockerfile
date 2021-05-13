FROM ubuntu:latest

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update
RUN apt-get install -y \
    g++ \
    cmake \
    vim \
    gdb \
    valgrind \
    git \
    libeigen3-dev \
    libboost-all-dev \
    nlohmann-json3-dev \
    python3.8 \
    python3-pip

RUN pip3 install jupyter \
                 numpy \
                 pandas \
                 matplotlib \
                 torch \
                 torchvision 

WORKDIR /code
CMD /bin/bash
