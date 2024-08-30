FROM gcr.io/yupp-llms/backend-base:latest

# set work directory
WORKDIR /app/

ENV PYTHONPATH=/app

# Install Poetry
RUN pip install poetry==1.8.2 && \
    poetry config virtualenvs.create false

COPY ./pyproject.toml ./poetry.lock* /app/
COPY ./backend /app/backend
COPY ./data/nltk_data /home/nltk_data
COPY ./db /app/db
COPY ./.env /app/.env

RUN bash -c "poetry install --no-root"

EXPOSE 8080

CMD ["uvicorn", "backend.server:app", "--host", "0.0.0.0", "--port", "8080"]
