ARG PYTORCH_VERSION=1.13.0
ARG PYTORCH_CUDA_VERSION=11.6
ARG PYTORCH_CUDNN_VERSION=8

FROM pytorch/pytorch:${PYTORCH_VERSION}-cuda${PYTORCH_CUDA_VERSION}-cudnn${PYTORCH_CUDNN_VERSION}-devel

ENV DEBIAN_FRONTEND=noniteractive
RUN apt update -y && \
    apt install -y \
    libopencv-dev \
    git

ARG UID
RUN useradd -l -m -u ${UID} dockeruser
USER ${UID}
ENV PATH=$PATH:/home/dockeruser/.local/bin

COPY docker/torch/requirements.txt requirements.txt
RUN pip --default-timeout=100 install --user -r requirements.txt
