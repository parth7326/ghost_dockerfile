FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa -y
RUN apt-get update && apt-get install -y \
    wget \
    python3.8 \
    python3.8-distutils \
    ffmpeg \
    libsm6 \
    libxext6

RUN wget https://bootstrap.pypa.io/get-pip.py

RUN python3.8 get-pip.py

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt
