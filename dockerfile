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
RUN pip install fasttext
RUN pip install nltk
RUN pip install sentencepiece


WORKDIR /home
ADD ruword2tags.tar.gz /home
WORKDIR /home/ruword2tags
RUN pip install .

RUN pip install git+https://github.com/Koziev/rutokenizer
RUN pip install git+https://github.com/Koziev/rusyllab

RUN apt-get clean

WORKDIR /text_generator/py/generative_poetry
COPY ./py/generative_poetry/*.py ./

WORKDIR /text_generator/py/poetry
COPY ./py/poetry/*.py ./

WORKDIR /text_generator/py/transcriptor_models
COPY ./py/transcriptor_models/*.py ./

WORKDIR /text_generator/py/transcriptor_models/stress_model
COPY ./py/transcriptor_models/stress_model/*.py ./

WORKDIR /text_generator/models/
COPY ./models/udpipe_syntagrus.model ./

WORKDIR /text_generator/tmp
COPY ./tmp/rselector.pkl ./

WORKDIR /text_generator/tmp
COPY ./tmp/accents.pkl ./

WORKDIR /text_generator/tmp
COPY ./tmp/rut5_for_poem_completion.pt ./


WORKDIR /text_generator/tmp/stress_model
COPY ./tmp/stress_model/*.* ./

WORKDIR /text_generator/tmp/stress_model/nn_stress.model/variables
COPY ./tmp/stress_model/nn_stress.model/variables/* ./

WORKDIR /text_generator/tmp/stress_model/nn_stress.model
COPY ./tmp/stress_model/nn_stress.model/*.pb ./



WORKDIR /text_generator/scripts/
COPY ./scripts/*.sh ./

WORKDIR /text_generator/models/rugpt_caption_generator
COPY ./models/rugpt_caption_generator/* ./

WORKDIR /text_generator/models/rugpt_poem_generator
COPY ./models/rugpt_poem_generator/* ./

WORKDIR /text_generator/py/generative_poetry
CMD "/text_generator/scripts/console.sh"

WORKDIR /text_generator