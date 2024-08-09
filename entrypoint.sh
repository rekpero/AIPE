#!/bin/bash

mkdir -p /app/output

# Start Ollama server in the background
ollama serve &

# Wait for Ollama server to start
echo "Waiting for Ollama server to start..."
while ! curl -s http://localhost:11434/api/tags > /dev/null; do
    sleep 1
done
echo "Ollama server is up and running!"

# Check if llama2 model is already pulled
if ollama list | grep -q "llama2"; then
    echo "llama2 model is already available."
else
    echo "Pulling llama2 model..."
    ollama pull llama2 > /dev/null 2>&1
fi

echo "Warming up the agent..."
# Run the main application
if [ -f /app/config/config.yaml ]; then
    python /app/main.py /app/config/config.yaml
else
    echo "Error: config.yaml not found in /app/config"
    exit 1
fi