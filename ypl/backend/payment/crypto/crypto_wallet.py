import json
import logging
import os

from cdp import Cdp, Wallet
from dotenv import load_dotenv
from ypl.backend.config import settings
from ypl.backend.utils.files import download_gcs_to_local_temp, read_file, write_file
from ypl.backend.utils.json import json_dumps

# File Paths
SEED_FILE_NAME = "encrypted_seed.json"
WALLET_FILE_NAME = "wallet.json"
CRYPTO_WALLET_PATH = settings.CRYPTO_WALLET_PATH
SEED_FILE_PATH = os.path.join(CRYPTO_WALLET_PATH, SEED_FILE_NAME)
WALLET_FILE_PATH = os.path.join(CRYPTO_WALLET_PATH, WALLET_FILE_NAME)


def init_cdp() -> None:
    """Initialize CDP configuration"""
    load_dotenv()
    api_key_name = settings.CDP_API_KEY_NAME
    api_key_private_key = settings.CDP_API_KEY_PRIVATE_KEY

    if not api_key_name or not api_key_private_key:
        raise ValueError("CDP configuration is missing")

    # Configure the CDP SDK
    private_key = api_key_private_key.replace("\\n", "\n")
    Cdp.configure(api_key_name, private_key)


def create_wallet() -> str | None:
    """
    Create a new CDP wallet and save its credentials.

    Returns:
        Optional[str]: The wallet address if successful, None otherwise
    """
    try:
        # Initialize CDP first
        init_cdp()

        # Create new wallet
        sending_wallet = Wallet.create(network_id="base-mainnet")

        # Save wallet credentials
        wallet_id_string = json.dumps(sending_wallet.id)
        write_file(WALLET_FILE_PATH, wallet_id_string)
        sending_wallet.save_seed(SEED_FILE_PATH, encrypt=True)

        # Get and return the wallet address
        sending_address = sending_wallet.default_address
        wallet_address = sending_address.address_id if sending_address else None

        log_dict = {
            "message": "Created wallet with address",
            "address": wallet_address,
        }
        logging.info(json_dumps(log_dict))

        return wallet_address

    except Exception as e:
        log_dict = {"message": "Failed to create wallet", "error": str(e)}
        logging.error(json_dumps(log_dict))
        return None


def import_existing_wallet() -> Wallet:
    """
    Import an existing wallet from saved credentials.

    Returns:
        Wallet: The imported wallet instance

    Raises:
        FileNotFoundError: If wallet files don't exist
        ValueError: If wallet import fails
    """
    try:
        wallet_data = read_file(WALLET_FILE_PATH)
        wallet_id = json.loads(wallet_data)

        with download_gcs_to_local_temp(SEED_FILE_PATH) as local_seed_path:
            wallet = Wallet.fetch(wallet_id)
            wallet.load_seed(local_seed_path)
            return wallet

    except Exception as e:
        log_dict = {"message": "Failed to import wallet", "error": str(e)}
        logging.error(json_dumps(log_dict))
        raise ValueError("Failed to import wallet") from e


def get_wallet_balance() -> None:
    """
    Get the balance of a specific asset in the wallet.

    Args:
        asset_id (str): The ID of the asset to check

    Returns:
        Decimal: The balance amount

    Raises:
        ValueError: If wallet import fails
    """
    try:
        init_cdp()
        wallet = import_existing_wallet()
        eth_balance = wallet.balance("eth")
        usdc_balance = wallet.balance("usdc")
        cbbtc_balance = wallet.balance("cbbtc")

        log_dict = {
            "message": "Retrieved wallet balance",
            "wallet_address": wallet.default_address.address_id if wallet.default_address else None,
            "eth_balance": str(eth_balance),
            "usdc_balance": str(usdc_balance),
            "cbbtc_balance": str(cbbtc_balance),
        }
        logging.info(json_dumps(log_dict))

    except Exception as e:
        log_dict = {"message": "Failed to get wallet balance", "error": str(e)}
        logging.error(json_dumps(log_dict))
        raise ValueError("Failed to get wallet balance") from e
