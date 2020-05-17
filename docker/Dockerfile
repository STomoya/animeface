FROM python:3.7
WORKDIR /usr/src
COPY docker/requirements.txt requirements.txt
RUN pip --default-timeout=100 install -r requirements.txt