#!/bin/bash
set -e

# Usage:
# ./run_docker.sh [--no-cache] [--help]
#
# Options:
#   --no-cache   Build the Docker image without using the cache.
#   --help       Show this help message and exit.

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Docker is not installed. Please install Docker and try again."
    exit 1
fi

IMAGE_NAME="embeddings_server"
CONTAINER_NAME="embeddings_server_container"

# Determine the Dockerfile path based on the current directory
if [ -f "./ypl/embeddings/Dockerfile" ]; then
    DOCKERFILE_PATH="./ypl/embeddings/Dockerfile"
elif [ -f "./Dockerfile" ]; then
    DOCKERFILE_PATH="./Dockerfile"
else
    echo "Dockerfile not found. Please run the script from the project root or embeddings directory."
    exit 1
fi
echo "Using Dockerfile at: $DOCKERFILE_PATH"

# Function to check if NVIDIA runtime is available
function check_nvidia_runtime {
    if docker info | grep -q "Runtimes:.*nvidia"; then
        echo "nvidia"
    else
        echo ""
    fi
}

# Parse arguments
NO_CACHE=""
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --no-cache) NO_CACHE="--no-cache"; shift ;;
        --help)
            echo "Usage: $0 [--no-cache] [--help]"
            echo ""
            echo "Options:"
            echo "  --no-cache   Build the Docker image without using the cache."
            echo "  --help       Show this help message and exit."
            exit 0
            ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
done

# Build the Docker image
echo "Building Docker image..."
start_time=$(date +%s)
docker build $NO_CACHE -t $IMAGE_NAME -f $DOCKERFILE_PATH .
end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
echo "Docker image built in $elapsed_time seconds."

# Get the size of the Docker image
image_size=$(docker image inspect $IMAGE_NAME --format='{{.Size}}')
image_size_mb=$(echo "scale=2; $image_size / (1024 * 1024)" | bc)
echo "Docker image size: $image_size_mb MB"

# Check for NVIDIA runtime
RUNTIME=$(check_nvidia_runtime)

# Run the Docker container
echo "Running Docker container..."
if [ -n "$RUNTIME" ]; then
    echo "NVIDIA runtime detected, allocating GPU..."
    docker run --rm --gpus all --name $CONTAINER_NAME -p 8000:8000 $IMAGE_NAME &
else
    echo "No NVIDIA runtime detected. For GPU support, please install the NVIDIA Container Toolkit following the instructions at https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html"
    echo "Running without GPU support..."
    docker run --rm --name $CONTAINER_NAME -p 8000:8000 $IMAGE_NAME &
fi
