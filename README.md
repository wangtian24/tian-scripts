# Yupp LLMs


## Purpose

The backend for providing services over multiple LLMs.

## Setup

### Conda

1. Install miniconda by following the [operating system-specific instructions](https://docs.conda.io/projects/miniconda/en/latest/).

    <details>
    <summary>Sample for macOS with zsh (click to expand)</summary>

    ```sh
    mkdir -p ~/miniconda3
    curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh -o ~/miniconda3/miniconda.sh
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
    rm -rf ~/miniconda3/miniconda.sh
    ~/miniconda3/bin/conda init zsh
    ```

    Warning: copied instructions can get outdated. Please update if you find there is a new or better way.
    </details>

1. Insta mamba through the [recommended Miniforge distribution](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html).

    <details>
    <summary>Sample macOS (click to expand)</summary>
    
    ```sh
    brew install miniforge
    ```

    Warning: copied instructions can get outdated. Please update if you find there is a new or better way.
    </details>

1. In the yupp-llms project directory, run the following command to create the run environment. This will create a minimal virtual environment.

```sh
mamba env create -n ys-dev --file envs/dev.yml
mamba activate ys-dev
```

### Poetry

Once the virtual environment is setup and activated, dependencies are managed by [poetry](https://python-poetry.org/). In order to install the backend dependencies, run (once the conda development environment is active):

```sh
poetry install --no-root
```

If you get an error about `psycopg2`, installing postgresql fixes it. `brew install postgresql` and then rerun poetry.

Once all the dependencies are installed, make a copy of the sample `.env.copy` environment and fill in the values:

```sh
cp .env.copy .env
```

To add new dependencies or update the versions of existings ones, modify the `[poetry]` section in [pyproject.toml] and run `poetry update`. This will modify the locked package versions in [poetry.lock]. Do not modify that file directly.

## Running the service

From the root of the repo, start the backend service with:

```sh
uvicorn backend.server:app --reload
```

See [localhost:8000/api/docs](http://localhost:8000/api/docs) for the available API routes. This only works if your `ENVIRONMENT` variable in the `.env` file is set to `local`.

## Testing

```sh
pytest
```

## Credentials

Important credentials (ie LLM API keys) can be found in 1Password.


## For Developers

### Project config

`pyproject.toml` is a configuration file to specify your project's metadata and to set the behavior of other tools such as linters, type checkers etc. You can learn more [here](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/).

### Linting and type checking

This repo uses `ruff` for Python linting and `mypy` to make sure Python code is typed.
Github actions in `.github/workflows` are set up to run these linters on push and pull requests.

## Deployment

This repo uses `Docker` to build the backend image. For faster builds, a base image contains most of the expensive dependencies; when these are modified significantly, the base image should be rebuilt.

The `main` branch is auto-deployed nightly to llms-staging.yupp.ai.

### Rebuilding the base image (slow, infrequent, when many dependencies change):

Note: if you get an authentication error on `docker push`, run `gcloud auth login` first.

```sh
docker build \
  --platform linux/amd64 \
  -t gcr.io/yupp-llms/backend-base:latest \
  -f Dockerfile.base .

docker push gcr.io/yupp-llms/backend-base:latest
```

### Building and deploying the actual backend image, which depends on the base:

Use the [Build and Deploy](https://github.com/yupp-ai/yupp-llms/actions/workflows/deploy.yml) workflow to build and deploy the backend image to staging or production; production will be pushed to llms.yupp.ai.