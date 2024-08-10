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

# Extract model_name from the nested structure using awk
model_name=$(awk '/context:/,/steps:/ {if ($1 == "model_name:") print $2}' /app/config/config.yaml | tr -d '"' | tr -d "'")

# Debug: Print the extracted model_name
echo "Extracted model_name: '$model_name'"

# Check if model_name is empty
if [ -z "$model_name" ]; then
    echo "Error: Unable to read model_name from config.yaml"
    exit 1
fi

# Check if the specified model is already pulled
if ollama list | grep -q "$model_name"; then
    echo "$model_name model is already available."
else
    echo "Pulling $model_name model..."
    ollama pull "$model_name" > /dev/null 2>&1
fi

echo "Warming up the agent..."
# Run the main application
if [ -f /app/config/config.yaml ]; then
    echo "Running main.py with config: /app/config/config.yaml"
    python /app/main.py /app/config/config.yaml
else
    echo "Error: config.yaml not found in /app/config"
    exit 1
fi