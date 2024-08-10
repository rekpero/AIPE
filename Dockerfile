FROM python:3.9-slim-buster

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