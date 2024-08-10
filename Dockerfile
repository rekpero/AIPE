# Use an official Python runtime as a parent image
FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    libgl1-mesa-glx \
    ffmpeg \
    wget \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh \
    && bash miniconda.sh -b -p /opt/conda \
    && rm miniconda.sh
ENV PATH="/opt/conda/bin:${PATH}"

# Create a conda environment
RUN conda create -n pipeline python=3.9 -y
SHELL ["conda", "run", "-n", "pipeline", "/bin/bash", "-c"]

# Copy the requirements file into the container
COPY requirements.txt /app/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install additional development tools
RUN pip install --no-cache-dir \
    pytest \
    pylint \
    black \
    ipython \
    jupyter

# Install Ollama
RUN curl https://ollama.ai/install.sh | sh

# Copy the current directory contents into the container at /app
COPY . /app

# Make port 8888 available for Jupyter Notebook
EXPOSE 8888

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTEST_ADDOPTS="--color=yes"

# Create a non-root user and switch to it
RUN useradd -m jupyteruser
USER jupyteruserFROM python:3.9-slim-buster

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    curl \
    espeak \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file
COPY app/requirements.txt .

# Install Python dependencies with verbose output
RUN pip install --no-cache-dir -v -r requirements.txt

# Install Ollama
RUN curl https://ollama.ai/install.sh | sh

# Copy the application code
COPY app /app
COPY config /app/config

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Create a volume for Ollama's model storage
VOLUME /root/.ollama

# Make port 11434 available to the world outside this container
EXPOSE 11434

# Copy and set the entrypoint script
USER root
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]

# Set the working directory to the jupyteruser home
WORKDIR /home/jupyteruser

# Create a symlink to the /app directory in the jupyteruser's home
RUN ln -s /app /home/jupyteruser/app

# Run Jupyter Notebook when the container launches
CMD ["conda", "run", "--no-capture-output", "-n", "pipeline", "jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]