FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel
ENV DEBIAN_FRONTEND=noniteractive
RUN apt update -y && \
    apt install -y \
    libopencv-dev
COPY docker/torch/requirements.txt requirements.txt
RUN pip --default-timeout=100 install -r requirements.txt