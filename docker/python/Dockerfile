FROM python:3.7
WORKDIR /usr/src
COPY docker/python/requirements.txt requirements.txt
RUN pip --default-timeout=100 install -r requirements.txt