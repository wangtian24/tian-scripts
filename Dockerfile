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

# Need this as the Stripe package is private
ARG STRIPE_GITHUB_TOKEN
RUN if [ -n "$STRIPE_GITHUB_TOKEN" ]; then \
    poetry config http-basic.github $STRIPE_GITHUB_TOKEN x-oauth-basic && \
    git config --global url."https://${STRIPE_GITHUB_TOKEN}@github.com/".insteadOf "https://github.com/" && \
    echo "GitHub authentication configured successfully"; \
    else \
    echo "Error: STRIPE_GITHUB_TOKEN is required but was not provided" && \
    exit 1; \
    fi

RUN echo "Installing dependencies..." && \
    STRIPE_GITHUB_TOKEN=$STRIPE_GITHUB_TOKEN poetry install --no-root || (echo "Poetry install failed. Check the logs above for details." && exit 1)
RUN bash -c "poetry build"
RUN bash -c "pip install ."

EXPOSE 8080

RUN chmod +x /app/ypl/backend/entrypoint.sh /app/ypl/partner_payments/server/entrypoint.sh

# Default script that can be overridden
ENV SCRIPT_TO_RUN=/app/ypl/backend/entrypoint.sh

# Use shell form to allow for environment variable expansion
ENTRYPOINT exec $SCRIPT_TO_RUN
