# Use the specified base image
FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-devel

# Set environment variable to prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Update the package list
RUN apt-get update -y

# Install git
RUN apt-get install -y git

# Install r-base and tzdata
RUN apt-get install -y r-base tzdata

# Install Python packages using pip
RUN pip install packaging
RUN pip install scgpt "flash-attn<1.0.5"
RUN pip install markupsafe==2.0.1
RUN pip install wandb