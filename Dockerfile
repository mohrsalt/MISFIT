# #Dockerfile
# FROM --platform=linux/amd64 pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime
# LABEL authors="Mohor_Banerjee"

# RUN apt-get update -y

# COPY misfit_env.yml .

# RUN conda env create -f misfit_env.yml
# SHELL ["conda", "run", "-n", "misfit", "/bin/bash", "-c"]
# # Other necessary instructions 
# COPY tools tools/
# COPY checkpoints checkpoints/
# COPY main.py .

# CMD ["python", "main.py"]

FROM --platform=linux/amd64 mambaorg/micromamba:1.5.8

LABEL authors="Mohor_Banerjee"

# Set environment name
ARG ENV_NAME=misfit
ENV MAMBA_ENV=${ENV_NAME}
ENV PATH=/opt/conda/envs/${ENV_NAME}/bin:$PATH

# Copy the environment file first to leverage caching
COPY misfit_env.yml /tmp/env.yml

# Create the environment
RUN micromamba create -y -n ${ENV_NAME} -f /tmp/env.yml && \
    micromamba clean --all --yes


ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV CUDA_VISIBLE_DEVICES=all

COPY tools tools/
COPY checkpoints checkpoints/
COPY main.py .

ENTRYPOINT []
SHELL ["micromamba", "run", "-n", "misfit", "/bin/bash", "-c"]


CMD ["python", "main.py"]
