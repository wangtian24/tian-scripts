FROM gcr.io/yupp-llms/backend-base:latest

# set work directory
WORKDIR /app/

ENV PYTHONPATH=/app

# Install Poetry
RUN pip install poetry==1.8.2 && \
    poetry config virtualenvs.create false

COPY ./pyproject.toml ./poetry.lock* /app/
COPY ./ypl/ /app/ypl
COPY ./data/nltk_data /home/nltk_data
COPY ./data/ /app/data
COPY ./.env /app/.env
COPY ./README.md /app/README.md

RUN bash -c "poetry install --no-root"
RUN bash -c "poetry build"
RUN bash -c "pip install ."

EXPOSE 8080

RUN chmod +x /app/ypl/backend/entrypoint.sh /app/ypl/partner_payments/server/entrypoint.sh

# Default script that can be overridden
ENV SCRIPT_TO_RUN=/app/ypl/backend/entrypoint.sh

# Use shell form to allow for environment variable expansion
ENTRYPOINT exec $SCRIPT_TO_RUN
