FROM nvidia/cuda:11.0.3-base-ubuntu20.04

ENV TZ=Europe/Prague
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

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

RUN pip install -r sber-swap/requirements.txt

EXPOSE 80
ENV PORT 80

ENTRYPOINT []
CMD ["python3.8", "/workspace/server.py"]
