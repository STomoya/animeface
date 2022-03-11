FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel

ENV DEBIAN_FRONTEND=noniteractive
RUN apt update -y && \
    apt install -y \
    libopencv-dev

ARG UID
RUN useradd -l -m -u ${UID} dockeruser
USER ${UID}
ENV PATH=$PATH:/home/dockeruser/.local/bin

COPY docker/preprocess/requirements.txt requirements.txt
RUN pip --default-timeout=100 install -r requirements.txt
