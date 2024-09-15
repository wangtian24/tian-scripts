FROM gcr.io/yupp-llms/backend-base:latest

# set work directory
WORKDIR /app/

ENV PYTHONPATH=/app

# Install Poetry
RUN pip install poetry==1.8.2 && \
    poetry config virtualenvs.create false

COPY ./pyproject.toml ./poetry.lock* /app/
COPY ./ypl/backend /app/ypl/backend
COPY ./data/nltk_data /home/nltk_data
COPY ./ypl/db /app/ypl/db
COPY ./.env /app/.env
COPY ./ypl/cli.py /app/ypl/cli.py
COPY ./README.md /app/README.md

RUN bash -c "poetry install --no-root"
RUN bash -c "poetry build"
RUN bash -c "pip install ."

EXPOSE 8080

RUN chmod +x /app/ypl/backend/entrypoint.sh

ENTRYPOINT ["/app/ypl/backend/entrypoint.sh"]