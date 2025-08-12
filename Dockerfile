FROM nvcr.io/nvidia/pytorch:24.06-py3

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y vim && \
    apt-get clean

RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple
RUN pip install uv

WORKDIR /app

COPY pyproject.toml ./
COPY model/bge-large-zh-v1.5 ./model/bge-large-zh-v1.5

RUN uv sync

VOLUME ["/app/src", "/app/logs", "/app/data", "/app/settings.prod.yml", "/app/settings.dev.yml"]

CMD ["/bin/bash"]
