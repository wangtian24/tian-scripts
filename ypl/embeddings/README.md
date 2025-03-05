# Embeddings Server

This is a FastAPI-based server that provides an endpoint to embed a list of strings into vector embeddings.

## Requirements

- Python 3.11+
- Poetry
- Mamba

## Installation

1. Activate your mamba environment:
    ```bash
    mamba activate ys-dev
    ```

2. Install the required packages using Poetry:
    ```bash
    poetry install
    ```

## Local Run

```bash
GCP_PROJECT=yupp-llms API_KEY=foobar ypl/embeddings/entrypoint.sh service --port 8080
```

Note, the entrypoint.sh script binds to port 80 by default. You might need
to adjust it to 8080 or some other port for your local machine.

## Deploy to Google Compute Engine

This guide outlines the steps to deploy and run the embedding server.

1. Creating a Deep Learning VM Instance:
  
   Begin by creating a Google Compute Engine (GCE) T4 GPU instance
   and select the "Deep Learning VM for PyTorch 2.4 with CUDA 12.4 M127"
   image when creating your instance. Ensure that "Allow HTTP traffic"
   and "Allow HTTPS traffic are both checked". Also, ensure that our
   static IP, 34.16.22.222, is assigned to the instance. You can find
   this setting under Networking/Networking Interfaces/External IPv4 Address.

2. Installing NVIDIA Drivers:

   Upon the first boot of your newly created instance, you will be prompted
   to install the NVIDIA drivers.

   ```
   This VM requires Nvidia drivers to function correctly.   Installation takes ~1 minute.
   Would you like to install the Nvidia driver? [y/n] y
   Installing Nvidia driver.
   ```

3. Connecting to Your Instance:

   Once the instance is running and the drivers are installed, establish a
   secure connection from your local machine using the gcloud command-line tool. 
   Execute the following command:

   Bash

   ```shell
   gcloud compute ssh <user>@embedding-omni
   ```
   This command uses OS Login for secure authentication.

4. Deploying the Embedding Server:

   With a connection established to your instance, you can now deploy the
   embedding server.  The server is distributed as a Docker image hosted
   on Google Container Registry (GCR).  To pull and run the latest version
   of the image, execute the following command:

   ```bash
   sudo docker run \
       -e API_KEY=$(gcloud secrets versions access latest --secret="embedding_service_api_key") \
       -e GCP_PROJECT=yupp-llms \
       -v "$HOME/.config/gcloud:/root/.config/gcloud" \
       --gpus all -d -p 80:80 \
       gcr.io/yupp-llms/embeddings-server:latest
   ```

5. Updating the Embedding Server:

   Pull the latest version of the image to ensure you're using the most up-to-date one:

   ```bash
   docker pull gcr.io/yupp-llms/embeddings-server:latest
   ```

   Check if the container is running:

   ```bash
   docker ps -a --filter "ancestor=gcr.io/yupp-llms/embeddings-server:latest"
   ```

   If a container is running, it will show up in the output.

   Stop and remove the existing container (if it's running):

   ```bash
   docker stop <container_id>
   docker rm <container_id>
   ```

   You can retrieve the container ID by running:

   ```bash
   docker ps -aq --filter "ancestor=gcr.io/yupp-llms/embeddings-server:latest"
   ```

   Start a new container with the latest image. See "Deploying the Embedding Server".

6. Test the Server.

   Since embed.yupp.ai is bound to our static IP address, 34.16.22.222, the following
   command will test whether the server is up correctly.

   ```bash
   API_KEY=$(gcloud secrets versions access latest --secret="embedding_service_api_key")

   curl -X POST "http://embed.yupp.ai/embed" \
        -H "Content-Type: application/json" \
        -H "x-api-key: ${API_KEY}" \
        -d '{
               "texts": ["Hello, world!", "Yupp with Yapps is great!"]
            }'
   ```

### Resources

 - [.github/workflows/build_and_push_embeddings_docker.yml](https://github.com/yupp-ai/yupp-mind/blob/main/.github/workflows/build_and_push_embeddings_docker.yml) script will create a
Docker image in the GCP Artifact Registry at [gcr.io/yupp-llms](https://console.cloud.google.com/artifacts/docker/yupp-llms/us/gcr.io?project=yupp-llms)
under the name [embeddings-server](https://console.cloud.google.com/artifacts/docker/yupp-llms/us/gcr.io/embeddings-server?project=yupp-llms).  

## Running the Server

If you prefer to run the server using Docker, which is the recommended approach,
use the following command:

```bash
./ypl/embeddings/run_docker.sh
```

Otherwise you can also start the server using the entrypoint script. Run the
following command from `yupp-mind/`:

```bash
./ypl/embeddings/entrypoint.sh service
```

## Security

The server reads the OS environment variable API_KEY on startup, and uses it to secure access
to all API endpoints. In production, the key is first generated and stored in the Google Cloud
secret named `embedding_service_api_key`:

```shell
LC_ALL=C tr -dc '[:alnum:]' < /dev/random | head -c 48 | gcloud secrets create embedding_service_api_key --data-file=-
```

## API

### POST /embed

**Headers:**
- `x-api-key`: Your API key for authentication.

**Request:**
- `texts`: List of strings to embed.
- `model_name`: The name of the model to use for embedding.

**Response:**
- `embeddings`: List of vector embeddings (each embedding is a list of floats).

**Example:**

Request:
```bash
curl -X POST "http://0.0.0.0/embed" -H "Content-Type: application/json" -H "x-api-key: YOUR_API_KEY" -d '{
    "texts": ["Hello, world!", "Yupp with Yapps is great!"]
}'
```

Response:
```json
{
    "embeddings": [
        [0.123, 0.456, ...],
        [0.789, 0.012, ...]
    ]
}
```

### GET /status

Check if the model is loaded and the service is ready.

**Example:**

**Headers:**
- `x-api-key`: Your API key for authentication.

Request:
```bash
curl -X GET "http://0.0.0.0:8000/status"
```

Response:
```json
{
    "status": "OK"
}
```
