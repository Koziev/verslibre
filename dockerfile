FROM python:3.7-slim

SHELL ["/bin/bash", "-c"]

# Default to UTF-8 file.encoding
ENV LANG C.UTF-8

RUN apt-get update
RUN apt-get install -y python python-pip
RUN apt-get -y install git-core
RUN apt-get install -y liblzma-dev

RUN pip install --upgrade pip
RUN pip install --upgrade setuptools

RUN pip install torch
RUN pip install tensorflow
RUN pip install transformers
RUN pip install pyconll
RUN pip install ufal.udpipe

RUN pip install git+https://github.com/Koziev/rutokenizer

RUN apt-get clean

WORKDIR /text_generator/py/generative_poetry
COPY ./py/generative_poetry/*.py ./

WORKDIR /text_generator/models/
COPY ./models/udpipe_syntagrus.model ./

WORKDIR /text_generator/scripts/
COPY ./scripts/*.sh ./

WORKDIR /text_generator/models/rugpt_caption_generator
COPY ./models/rugpt_caption_generator/* ./

WORKDIR /text_generator/models/rugpt_poem_generator
COPY ./models/rugpt_poem_generator/* ./

WORKDIR /text_generator/py/generative_poetry
CMD "/text_generator/scripts/console.sh"
