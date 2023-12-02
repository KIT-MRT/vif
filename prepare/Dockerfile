FROM ubuntu:22.04

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6
RUN apt-get update && apt-get install -y python3 python3-pip
RUN python3 -m pip install kornia imagesize waymo-open-dataset-tf-2-11-0==1.5.2 tqdm opencv-python==4.7.0.72 sqlalchemy shapely
RUN python3 -m pip install numpy --upgrade

COPY . /code
WORKDIR /code
