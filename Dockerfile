FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list

RUN apt-get update && \
    apt-get install -y vim python3 python3-pip && \
    apt-get clean

RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple
RUN pip install uv

WORKDIR /app

COPY pyproject.toml ./
COPY model/bge-large-zh-v1.5 ./model/bge-large-zh-v1.5

RUN uv sync

VOLUME ["/app/src", "/app/logs", "/app/data", "/app/settings.prod.yml", "/app/settings.dev.yml"]

CMD ["/bin/bash"]
