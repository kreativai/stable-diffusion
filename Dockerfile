FROM nvcr.io/nvidia/pytorch:20.12-py3

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

ENV TZ=Europe/Berlin
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update \
    && apt-get clean \
    && apt-get update -qqq \
    && pip install --upgrade pip \
    && apt-get install -y git \
    && apt-get install -y wget \
    && apt-get install -y zip \
    && apt-get install -y ffmpeg libsm6 libxext6

RUN pip install diffusers==0.2.4 transformers scipy ftfy
RUN python3 -m pip install huggingface_hub

RUN mkdir Stable

COPY token.txt Stable/
COPY hugging_run.py Stable/

WORKDIR Stable

#ENTRYPOINT ["python3", "hugging_run.py"]
