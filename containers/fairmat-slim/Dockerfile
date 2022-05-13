# NOTE, the Docker build context should be set to the repo root
# not this folder! This means you should build using
# 
# # (from project root)
# yarn build-container 
#
# # (or manually)
# docker build -f containers/fairmat-slim/Dockerfile .
# 
# or a similar invocation.

FROM condaforge/mambaforge AS build
LABEL build_version="0.1.0"
LABEL description="Provides a minimal disk space containerized installation of PyARPES."

# Multi stage environment build which serves two purposes:
# Stage 0 installs PyQt5 with pip instead of with conda. This needs to happen
# first so that we get version 5.15 which is not available on conda. If we install
# the correct version with pip after the conda environment builds, then
# pip clobbers the PyQt5 installation conda chose and `conda-pack` refuses to run.
COPY containers/fairmat-slim/environment-fairmat-slim-stage0.yml . 
RUN mamba env create -f environment-fairmat-slim-stage0.yml

# Stage 1 installs the rest of the dependencies. We specifically request
# matplotlib-base to prevent Qt installation.
COPY containers/fairmat-slim/environment-fairmat-slim-stage1.yml . 
RUN mamba env update --prefix /opt/conda/envs/arpes -f environment-fairmat-slim-stage1.yml

# Stage 2 installs PyARPES with all dependencies already installed.
# This can probably be pushed after the `conda-pack` step,
# which would make rebuilds very fast. We should consider this.
COPY containers/fairmat-slim/environment-fairmat-slim-stage2.yml . 
COPY . ./
RUN mamba env update --prefix /opt/conda/envs/arpes -f environment-fairmat-slim-stage2.yml

RUN mamba install -c conda-forge conda-pack

RUN conda-pack -n arpes -o /tmp/env.tar && \
  mkdir /venv && cd /venv && tar xf /tmp/env.tar && \
  rm /tmp/env.tar

RUN /venv/bin/conda-unpack

FROM debian:buster AS runtime

COPY --from=build /venv /venv

SHELL ["bin/bash", "-c"]

# just verify there are no major surprises
ENTRYPOINT source /venv/bin/activate && python -c "from arpes.all import *"