#!/usr/bin/env python3

import sys
from typing import Literal, cast, get_args

Environment = Literal["staging", "production"]
Service = Literal["backend", "partner-payments-server", "backend-pytorch-service", "webhooks"]


def generate_deploy_secrets_string(environment: Environment, service_name: Service) -> str:
    """
    Generate a string of secrets for the deploy action.

    Given a KEY=VALUE pair, the deploy action will set the secret in the container on the following format:
    --set-secrets='KEY1=secret-1:version-1,KEY2=secret-2:latest,...'

    KEY will be the name of the environment variable in the container.
    VALUE has to be `secret-name:version-number` where `secret-name` is the name of the secret in GCP Secret Manager,
    and `version-number` is the version of the secret to use. Use `latest` to use the latest version of the secret.

    Refer https://cloud.google.com/run/docs/configuring/services/secrets?hl=en for more details.
    """
    backend_secrets = [
        f"CHECKOUT_COM_API_URL=checkout-com-api-url-{environment}:latest",
        f"CHECKOUT_COM_ENTITY_ID=checkout-com-entity-id-{environment}:latest",
        f"CHECKOUT_COM_PROCESSING_CHANNEL=checkout-com-processing-channel-{environment}:latest",
        "EMBED_X_API_KEY=embedding_service_api_key:latest",
        f"GUEST_MANAGEMENT_SLACK_WEBHOOK_URL=guest-mgmt-slack-webhook-url-{environment}:latest",
        f"HYPERWALLET_API_URL=hyperwallet-api-url-{environment}:latest",
        f"HYPERWALLET_PASSWORD=hyperwallet-password-{environment}:latest",
        f"HYPERWALLET_PROGRAM_TOKEN=hyperwallet-program-token-{environment}:latest",
        f"HYPERWALLET_USERNAME=hyperwallet-username-{environment}:latest",
        "IPINFO_API_KEY=ipinfo-api-key:latest",
        f"PARTNER_PAYMENTS_API_URL=partner-payments-api-url-{environment}:latest",
        f"PAYPAL_WEBHOOK_ID=paypal-webhook-id-{environment}:latest",
        f"VALIDATE_DESTINATION_IDENTIFIER_SECRET_KEY=validate-destination-identifier-secret-key-{environment}:latest",
        "VPNAPI_API_KEY=vpnapi-api-key:latest",
    ]

    # Add production-only secrets
    if environment == "production":
        backend_secrets.extend(
            [
                f"RESEND_API_KEY=resend-api-key-{environment}:latest",
            ]
        )

    # Ensure that the deploy script is also modified if you're adding a secret here.
    webhooks_secrets: list[str] = []
    partner_payments_server_secrets: list[str] = []

    if service_name == "backend" or service_name == "backend-pytorch-service":
        secrets = backend_secrets
    elif service_name == "webhooks":
        secrets = webhooks_secrets
    elif service_name == "partner-payments-server":
        secrets = partner_payments_server_secrets

    for secret in secrets:
        key, value = secret.split("=")
        assert (
            key.replace("_", "").isalnum() and not key[0].isdigit()
        ), f"KEY {key} in {secret} must be a valid environment variable name"
        assert key.isupper(), f"KEY {key} in {secret} must be uppercase"
        assert ":" in value, f"SECRET {secret} must be in format KEY=name:version"
        _, version = value.split(":")
        assert version == "latest" or version.isdigit(), f"VERSION {version} in {secret} must be 'latest' or a number"
    return ",".join(secrets)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: generate_gcloud_deploy_secrets_string.py <environment> <service-name>")
        sys.exit(1)

    environment = sys.argv[1]
    service_name = sys.argv[2]
    assert environment in get_args(Environment), f"<environment> {environment} must be one of {get_args(Environment)}"
    assert service_name in get_args(Service), f"<service_name> {service_name} must be one of {get_args(Service)}"

    secrets_string = generate_deploy_secrets_string(
        environment=cast(Environment, environment), service_name=cast(Service, service_name)
    )
    # GitHub Actions specific output format
    print(f"secrets_string={secrets_string}")
