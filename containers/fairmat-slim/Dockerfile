# NOTE, the Docker build context should be set to the repo root
# not this folder! This means you should build using
# 
# docker build -f containers/fairmat-slim/Dockerfile .
# 
# or a similar invocation.

FROM continuumio/miniconda3 AS build

COPY containers/fairmat-slim/environment-fairmat-slim.yml . 
RUN conda env create -f environment-fairmat-slim.yml

RUN conda install -c conda-forge conda-pack

RUN conda-pack -n arpes -o /tmp/env.tar && \
  mkdir /venv && cd /venv && tar xf /tmp/env.tar && \
  rm /tmp/env.tar

RUN /venv/bin/conda-unpack

FROM debian:buster AS runtime

COPY --from=build /venv /venv

SHELL ["bin/bash", "-c"]

# just verify there are no major surprises
ENTRYPOINT source /venv/bin/activate && python -c "from arpes.all import *"