FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel
COPY docker/torch/requirements.txt requirements.txt
RUN pip --default-timeout=100 install -r requirements.txt