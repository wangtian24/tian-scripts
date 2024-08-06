# Yupp LLMs


## Purpose

The backend for providing services over multiple LLMs.

## Setup

### Conda

1. Install micoconda by following the [operating system-specific instructions](https://docs.conda.io/projects/miniconda/en/latest/). 
1. Insta mamba through the [recommended Miniforge distribution](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html/).
1. Run the following command to create the run environment. This will createa a minimal virtual environment.

```sh
mamba env create -n ys-dev --file envs/dev.yml
mamba activate ys-dev
```

### Poetry

Once the virtual environment is setup and activated, dependencies are managed by [poetry](https://python-poetry.org/). In order to install the backend dependencies, run (once the conda development environment is active):

```sh
poetry install --no-root
```

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

This repo uses `Docker` to build the backend image.

To manually build version `v1`:

```sh
export YUPP_VERSION=v1

docker build \
  --platform linux/amd64 \
  -t gcr.io/yupp-llms/backend:$YUPP_VERSION .

docker push gcr.io/yupp-llms/backend:$YUPP_VERSION

gcloud run deploy backend \
  --image gcr.io/yupp-llms/backend:$YUPP_VERSION \
  --platform managed \
  --region us-east4 \
  --cpu 2 \
  --memory 4Gi
```
