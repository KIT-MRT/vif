FROM nvcr.io/nvidia/pytorch:23.06-py3

RUN apt-get update
RUN apt-get -y install ffmpeg

WORKDIR /code
COPY ./requirements.txt /code
RUN python -m pip install -r ./requirements.txt