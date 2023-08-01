FROM gooseai/torch-base:6cfdc11

RUN apt-get update && apt-get -y upgrade \
  && apt-get install -y --no-install-recommends \
    git \
    wget \
    g++ \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*
 

RUN apt-get update && apt-get install -y \
        cuda-nvcc-11-3 \
        cuda-nvml-dev-11-3 \
        libcurand-dev-11-3 \
        libcublas-dev-11-3 \
        libcusparse-dev-11-3 \
        libcusolver-dev-11-3 \
        cuda-nvprof-11-3 \
        ninja-build && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/
RUN apt-get update && apt-get install -y software-properties-common && \
add-apt-repository ppa:deadsnakes/ppa -y && \
 apt-get update && apt-get install -y \
    wget \
    vim \
    python3.8 \
    python3.8-distutils \
    ffmpeg \
    libsm6 \
    libxext6 && \
apt-get clean && \
rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN wget https://bootstrap.pypa.io/get-pip.py

RUN python3.8 get-pip.py

COPY ./ /workspace/

RUN pip install -r requirements.txt

EXPOSE 80
ENV PORT 80

ENTRYPOINT []
CMD ["python3.8", "/workspace/server.py"]
