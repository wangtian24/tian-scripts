import logging
import os

import plaid
from plaid.api import plaid_api
from ypl.backend.utils.json import json_dumps

# Available environments are
# 'Production'
# 'Sandbox'
host = plaid.Environment.Production if os.environ.get("ENVIRONMENT") == "production" else plaid.Environment.Sandbox

_plaid_client_instance: plaid_api.PlaidApi | None = None


def get_plaid_client() -> plaid_api.PlaidApi:
    """Get the singleton instance of Plaid client."""
    global _plaid_client_instance
    if _plaid_client_instance is None:
        init_plaid_client()
    return _plaid_client_instance


def init_plaid_client() -> None:
    """Initialize Plaid client singleton instance."""
    global _plaid_client_instance

    try:
        # Initial client with environment variables
        configuration = plaid.Configuration(
            host=host,
            api_key={
                "clientId": os.getenv("PLAID_CLIENT_ID"),
                "secret": os.getenv("PLAID_SECRET"),
            },
        )

        api_client = plaid.ApiClient(configuration)
        _plaid_client_instance = plaid_api.PlaidApi(api_client)

        log_dict = {"message": "Plaid client initialized", "environment": host}
        logging.info(json_dumps(log_dict))
    except Exception as e:
        # incase of exception do not raise the error so that this does not block server instantiation
        log_dict = {"message": "Failed to initialize Plaid client", "error": str(e)}
        logging.error(json_dumps(log_dict))


def cleanup_plaid_client() -> None:
    """Cleanup Plaid client during application shutdown."""
    global _plaid_client_instance
    if _plaid_client_instance is not None:
        # Add any cleanup code here if needed
        _plaid_client_instance = None
        log_dict = {"message": "Plaid client cleaned up"}
        logging.info(json_dumps(log_dict))
