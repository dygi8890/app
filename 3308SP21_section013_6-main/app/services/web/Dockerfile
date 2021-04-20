FROM python:3.8.1-slim-buster
LABEL mantainer="33008_13_06:TeamNatives"

# set work directory
WORKDIR /usr/src/app

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

## Before doing anything, upgrade pip
RUN pip install --upgrade pip

## install OpenCV and its dependencies
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install opencv-python

# install other, more normal, dependencies
COPY ./requirements.txt /usr/src/app/requirements.txt
RUN pip install -r requirements.txt

# copy project
COPY . /usr/src/app/