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

RUN pip3 install -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers
RUN pip3 install git+https://github.com/openai/CLIP.git
RUN pip3 install git+https://github.com/crowsonkb/k-diffusion/

RUN mkdir Stable
COPY ./ Stable/

WORKDIR Stable
RUN pip3 install -r requirements.txt

ENTRYPOINT ["python3", "scripts/txt2img_k.py", "--prompt", "\"beautiful clean oil painting portrait of jesus by rafael albuquerque, wayne barlowe, rembrandt, complex, stunning, realistic\"", "--ckpt", "./sd-v1-3-full-ema.ckpt", "--n_samples", "3", "--n_iter", "1", "--ddim_steps", "50", "--H", "512", "--W", "512", "--seed", "2815820077"]

