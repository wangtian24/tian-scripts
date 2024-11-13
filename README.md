# Yupp LLMs

Multi Intelligence Neural Distribution Service

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

1. In the yupp-mind project directory, run the following command to create the run environment `ys-dev`. This will create a minimal virtual environment.

```sh
mamba env create -n ys-dev --file envs/dev.yml
mamba init # needed the first time
mamba activate ys-dev
```
Confirm the `ys-dev` environment is activated by running `mamba info --envs` and seeing `ys-dev` in the list with a `*` next to it.

### Poetry

Once the virtual environment is setup and activated, dependencies are managed by [poetry](https://python-poetry.org/). In order to install the backend dependencies, run (once the conda development environment is active):

```sh
poetry install --no-root
```

If you get an error about `psycopg2`, installing postgresql fixes it. `brew install postgresql` and then rerun poetry.

Once all the dependencies are installed, make a copy of the sample `.env.copy` environment and fill in the values. You can get the values from 1Password (yupp-mind .env file):

```sh
cp .env.copy .env
```

To add new dependencies or update the versions of existings ones, modify the `[poetry]` section in [pyproject.toml] and run `poetry update`. This will modify the locked package versions in [poetry.lock]. Do not modify that file directly.

Finally, install `ypl` as a package in the `ys-dev` environment:

```sh
poetry build
pip install -e .  # editable mode for dev
```

## Google Cloud Logging

Google Cloud Logging is disabled by default. To enable it, set `USE_GOOGLE_CLOUD_LOGGING=True` in the `.env` file. In order to do GCP logging from local environment, you need to download the GCP service account key file from 1Password and set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable to the path of the downloaded key file.

In order to redact sensitive data, the `redact_sensitive_data` function is used. If you need to add new sensitive data patterns to redact, you can do so by adding them to the function.

## Running the service

Start the backend service in any folder containing the `.env` file with:

```sh
uvicorn ypl.backend.server:app --reload
```

See [localhost:8000/api/docs](http://localhost:8000/api/docs) for the available API routes. This only works if your `ENVIRONMENT` variable in the `.env` file is set to `local`.

## Accessing the APIs

The APIs are protected by API key. The key is stored in the Github Secrets (and Vercel Environment Variables), which will be injected as part of Github Actions Workflow. If you want to access the API, you need to set the `X-API-KEY` header with the right key value.
At the moment, `local` environment is exempt from authentication and is enabled only for other (`staging` and `production`) environments.

If you need to access the APIs from FastAPI Docs, you can add the API key by clicking on `Authorize` button (top right).

<img height="400" src="./assets/auth_api_key.png" alt="Authorize API Key">

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

This repo uses `ruff` for Python linting and `mypy` to make sure Python code is typed. Please download the IDE extensions for `ruff` and `mypy` from [here](https://code.visualstudio.com/docs/python/linting) to help with linting and type checking.

When using `mypy` in VSCode, you may need to enable the "Run using active interpreter" setting to make sure `mypy` uses the daemon in the virtual environment. You can find it by going to `Settings`, searching for `mypy.runUsingActiveInterpreter`, and checking the box.


Github actions in `.github/workflows` are set up to run these linters on push and pull requests.

## Deployment

This repo uses `Docker` to build the backend image. For faster builds, a base image contains most of the expensive dependencies; when these are modified significantly, the base image should be rebuilt.

The `main` branch is auto-deployed nightly to llms-staging.yupp.ai.

### Rebuilding the base image (slow, infrequent, when many dependencies change):

Note: if you get an authentication error on `docker push`, run `gcloud auth login` followed by `gcloud auth configure-docker` first.

```sh
docker build \
  --platform linux/amd64 \
  -t gcr.io/yupp-llms/backend-base:latest \
  -f Dockerfile.base .

docker push gcr.io/yupp-llms/backend-base:latest
```

### Building and deploying the actual backend image, which depends on the base:

Use the "Run Workflow" button on the [Build and Deploy](https://github.com/yupp-ai/yupp-mind/actions/workflows/deploy.yml) workflow to build and deploy the backend image to staging or production; production will be pushed to llms.yupp.ai:

<img height="400" src="./assets/deploy.png" alt="Build and Deploy">


### Rollbacks

If you want to rollback to the previous revision, you can just use the [Rollback to previous revision](https://github.com/yupp-ai/yupp-mind/actions/workflows/rollback.yml) Github workflow.

To roll back to a different version, first list the revisions of the service (use `backend-staging` instead of `backend` to do the same for the staging service):

```sh
gcloud run revisions list --service=backend --region=us-east4 --platform=managed
```

This should result in a list of previous deploys:

```sh
   REVISION                   ACTIVE  SERVICE          DEPLOYED                 DEPLOYED BY
✔  backend-00004-qpm  yes     backend  2024-08-23 05:34:06 UTC  github-deploy@yupp-llms.iam.gserviceaccount.com
✔  backend-00003-nz8          backend  2024-08-23 05:24:52 UTC  github-deploy@yupp-llms.iam.gserviceaccount.com
✔  backend-00002-qjv          backend  2024-08-23 05:03:39 UTC  github-deploy@yupp-llms.iam.gserviceaccount.com
```

Choose the one to roll back to, and run:

```sh
gcloud run services update-traffic backend \
  --to-revisions=backend-00002-qjv=100 \
  --region=us-east4 \
  --platform=managed
```

## Periodic tasks

Periodic tasks are run using GitHub actions, based on [`.github/workflows/cronjob.yml`](.github/workflows/cronjob.yml)); any command and arguments to [cli.py](ypl/cli.py) can be run as a periodic task.

To add a new periodic task:
1. Add a command to [cli.py](ypl/cli.py) (ex: `update_ranking()`).
1. Create a GitHub action for that command with the desired schedule and any arguments to supply (ex: [`.github/workflows/update_ranking.yml`](.github/workflows/update_ranking.yml)).

Periodic tasks will run automatically from the `main` branch, but can also be manually triggered for any branch from [the action page](https://github.com/yupp-ai/yupp-mind/actions/workflows/cronjob.yml).