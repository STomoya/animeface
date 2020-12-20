FROM ubuntu:18.04

ARG WAVE_VERSION

RUN apt update -y \
    && apt upgrade -y \
    && apt install -y \
    curl

RUN curl -L https://github.com/h2oai/wave/releases/download/v${WAVE_VERSION}/wave-${WAVE_VERSION}-linux-amd64.tar.gz -o wave-${WAVE_VERSION}-linux-amd64.tar.gz \
    && tar -xzf wave-${WAVE_VERSION}-linux-amd64.tar.gz \
    && mv wave-${WAVE_VERSION}-linux-amd64 /usr/wave