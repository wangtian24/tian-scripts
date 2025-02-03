import json
import logging
import os
import secrets
from base64 import b64decode, b64encode
from typing import Any

from cdp import Cdp, Wallet
from cdp.wallet import WalletData
from cryptography.fernet import Fernet
from sqlalchemy import select
from ypl.backend.config import settings
from ypl.backend.db import get_async_session
from ypl.backend.utils.files import download_gcs_to_local_temp
from ypl.backend.utils.json import json_dumps
from ypl.db.payments import PaymentInstrument

SEED_FILE_PATH = os.environ.get("SEED_FILE_PATH", "")  # this is the path to the seed file
SLACK_WEBHOOK_CASHOUT = settings.SLACK_WEBHOOK_CASHOUT
WALLET_ENCRYPTION_KEY = settings.WALLET_ENCRYPTION_KEY


def init_cdp() -> None:
    """Initialize CDP configuration"""
    api_key_name = settings.CDP_API_KEY_NAME
    api_key_private_key = settings.CDP_API_KEY_PRIVATE_KEY

    if not api_key_name or not api_key_private_key:
        raise ValueError("CDP configuration is missing")

    # Configure the CDP SDK
    private_key = api_key_private_key.replace("\\n", "\n")
    Cdp.configure(api_key_name, private_key)


async def create_wallet() -> str | None:
    """
    Create a new CDP wallet and save its credentials.

    Returns:
        Optional[str]: The wallet address if successful, None otherwise
    """
    # This was needed during initial wallet creation. Please DO NOT USE this to
    #  create a new wallet as it will override the existing wallet in the db
    try:
        # Initialize CDP first
        # init_cdp()

        # # Create new wallet
        # sending_wallet = Wallet.create(network_id="base-mainnet")

        # # uncomment the below line to update db with this new wallet if you want to really update the db
        # # await store_wallet_data(sending_wallet, bytes.fromhex(settings.WALLET_ENCRYPTION_KEY))

        # # Get and return the wallet address
        # sending_address = sending_wallet.default_address
        # wallet_address = sending_address.address_id if sending_address else None

        # log_dict = {
        #     "message": "Created wallet with address",
        #     "address": wallet_address,
        # }
        # logging.info(json_dumps(log_dict))

        # return wallet_address
        return None
    except Exception as e:
        log_dict = {"message": "Failed to create wallet", "error": str(e)}
        logging.error(json_dumps(log_dict))
        return None


async def import_existing_wallet() -> Wallet:
    """
    Import an existing wallet from saved credentials.

    Returns:
        Wallet: The imported wallet instance

    Raises:
        ValueError: If wallet import fails
    """
    try:
        init_cdp()
        wallet_data = await load_wallet_data(bytes.fromhex(settings.WALLET_ENCRYPTION_KEY))
        wallet = Wallet.import_data(wallet_data)
        return wallet
    except Exception as e:
        log_dict = {"message": "Failed to import wallet", "error": str(e)}
        logging.error(json_dumps(log_dict))
        raise ValueError("Failed to import wallet") from e


async def get_wallet_balance() -> dict:
    """
    Get the balance of assets in the wallet.

    Returns:
        dict: A dictionary containing wallet address and balances for different assets
            Format:
            {
                'wallet_address': str,
                'balances': [
                    {'currency': str, 'balance': Decimal},
                    ...
                ]
            }

    Raises:
        ValueError: If wallet import fails
    """
    try:
        init_cdp()
        wallet = await import_existing_wallet()
        eth_balance = wallet.balance("eth")
        usdc_balance = wallet.balance("usdc")
        cbbtc_balance = wallet.balance("cbbtc")

        wallet_data = {
            "wallet_address": wallet.default_address.address_id if wallet.default_address else None,
            "balances": [
                {"currency": "ETH", "balance": eth_balance},
                {"currency": "USDC", "balance": usdc_balance},
                {"currency": "CBBTC", "balance": cbbtc_balance},
            ],
        }

        log_dict = {
            "message": "Retrieved wallet balance",
            "wallet_address": wallet_data["wallet_address"],
            "eth_balance": str(eth_balance),
            "usdc_balance": str(usdc_balance),
            "cbbtc_balance": str(cbbtc_balance),
        }
        logging.info(json_dumps(log_dict))

        return wallet_data
    except Exception as e:
        log_dict = {"message": "Failed to get wallet balance", "error": str(e)}
        logging.error(json_dumps(log_dict))
        raise ValueError("Failed to get wallet balance") from e


def generate_encryption_key() -> bytes:
    """
    Generate a secure encryption key using secrets module.

    Returns:
        bytes: A 32-byte (256-bit) encryption key
    """
    try:
        encryption_key = secrets.token_bytes(32)
        log_dict = {
            "message": "Generated new encryption key",
            "key_length": len(encryption_key),
            "key": encryption_key.hex(),
        }
        logging.info(json_dumps(log_dict))
        return encryption_key
    except Exception as e:
        log_dict = {"message": "Failed to generate encryption key", "error": str(e)}
        logging.error(json_dumps(log_dict))
        raise ValueError("Failed to generate encryption key") from e


async def store_wallet_data(wallet: Wallet, encryption_key: bytes) -> None:
    """
    Securely store wallet data in the database using encryption.

    Args:
        wallet: The wallet instance to store
        encryption_key: The encryption key to use for securing the data
    """
    #  This was a one time script to migrate wallet credentials from old to new
    #  and won't be needed in the future. This is just captured here for incident reference
    #  Only if in future we need to create a new wallet, we will need to update the db
    #  by calling this function with the new wallet instance and encryption key
    try:
        # Export wallet data
        data = wallet.export_data()
        data_dict = data.to_dict()

        log_dict = {
            "message": "Exported wallet data",
            "data_dict": len(data_dict),
        }
        logging.info(json_dumps(log_dict))

        # Encrypt the data
        fernet = Fernet(b64encode(encryption_key))
        encrypted_data = fernet.encrypt(json.dumps(data_dict).encode())

        # Convert encrypted bytes to base64 string for JSON storage
        base64_encrypted_data = b64encode(encrypted_data).decode("utf-8")

        async with get_async_session() as session:
            # Find the existing system wallet record
            query = select(PaymentInstrument).where(
                PaymentInstrument.user_id == "SYSTEM",  # type: ignore
                PaymentInstrument.facilitator == "ON_CHAIN",  # type: ignore
                PaymentInstrument.identifier_type == "CRYPTO_ADDRESS",  # type: ignore
                PaymentInstrument.deleted_at.is_(None),  # type: ignore
            )
            result = await session.execute(query)
            payment_instrument = result.scalar_one_or_none()

            if payment_instrument:
                # Update existing record
                payment_instrument.instrument_metadata = {"encrypted_wallet_data": base64_encrypted_data}
                await session.commit()

                log_dict = {
                    "message": "Successfully stored encrypted wallet data",
                    "wallet_address": wallet.default_address.address_id if wallet.default_address else None,
                }
                logging.info(json_dumps(log_dict))
            else:
                raise ValueError("No wallet data found in database")

    except Exception as e:
        log_dict = {"message": "Failed to store wallet data", "error": str(e)}
        logging.error(json_dumps(log_dict))
        raise ValueError("Failed to store wallet data") from e


async def load_wallet_data(encryption_key: bytes) -> WalletData:
    """
    Load encrypted wallet data from the database and decrypt it.

    Args:
        encryption_key: The encryption key used to secure the data

    Returns:
        WalletData: The decrypted wallet data

    Raises:
        ValueError: If wallet data cannot be loaded or decrypted
    """
    from ypl.db.payments import PaymentInstrument

    try:
        async with get_async_session() as session:
            query = select(PaymentInstrument).where(
                PaymentInstrument.user_id == "SYSTEM",  # type: ignore
                PaymentInstrument.facilitator == "ON_CHAIN",  # type: ignore
                PaymentInstrument.identifier_type == "CRYPTO_ADDRESS",  # type: ignore
                PaymentInstrument.deleted_at.is_(None),  # type: ignore
            )
            result = await session.execute(query)
            payment_instrument = result.scalar_one_or_none()

            if not payment_instrument or "encrypted_wallet_data" not in payment_instrument.instrument_metadata:
                raise ValueError("No wallet data found in database")

            # Convert base64 string back to bytes for decryption
            base64_encrypted_data = payment_instrument.instrument_metadata["encrypted_wallet_data"]
            encrypted_data = b64decode(base64_encrypted_data.encode("utf-8"))

            # Decrypt the data
            fernet = Fernet(b64encode(encryption_key))
            decrypted_data = json.loads(fernet.decrypt(encrypted_data).decode())

            log_dict = {"message": "Successfully loaded wallet data"}
            logging.info(json_dumps(log_dict))

            return WalletData(**decrypted_data)

    except Exception as e:
        log_dict = {"message": "Failed to load wallet data", "error": str(e)}
        logging.error(json_dumps(log_dict))
        raise ValueError("Failed to load wallet data") from e


async def migrate_wallet_credentials_for_wallet_id(
    old_api_key_name: str,
    old_api_private_key: str,
    new_api_key_name: str,
    new_api_private_key: str,
    wallet_id: str,
) -> None:
    """
    Migrate wallet credentials using new API keys while loading data with old API keys.
    Stores the wallet data in an encrypted format in the database.

    Args:
        old_api_key_name: Previous CDP API key name
        old_api_private_key: Previous CDP API private key
        new_api_key_name: New CDP API key name
        new_api_private_key: New CDP API private key
        wallet_id: The ID of the wallet to recover

    Raises:
        ValueError: If migration fails
    """
    # This is a one time script to migrate wallet credentials from old to new
    # and won't be needed in the future. This is just captured here for incident reference
    try:
        # Configure CDP with new API keys to fetch wallet
        new_private_key = new_api_private_key.replace("\\n", "\n")
        Cdp.configure(new_api_key_name, new_private_key)
        log_dict: dict[str, Any] = {
            "message": "Configured CDP with new API keys",
            "api_key_name": new_api_key_name[-7:] if new_api_key_name else "",
            "api_private_key": new_private_key[-7:] if new_private_key else "",
        }
        logging.info(json_dumps(log_dict))
        wallet = Wallet.fetch(wallet_id)
        log_dict = {
            "message": "Fetched wallet",
            "wallet_id": wallet_id,
            "wallet_address": str(wallet.default_address.address_id) if wallet.default_address else "",
        }
        logging.info(json_dumps(log_dict))

        # Switch to old API keys to load the seed
        old_private_key = old_api_private_key.replace("\\n", "\n")
        Cdp.configure(old_api_key_name, old_private_key)
        log_dict = {
            "message": "Configured CDP with old API keys",
            "api_key_name": old_api_key_name[-7:] if old_api_key_name else "",
            "api_private_key": old_private_key[-7:] if old_private_key else "",
        }
        logging.info(json_dumps(log_dict))

        # Load the seed using existing file
        with download_gcs_to_local_temp(SEED_FILE_PATH) as local_seed_path:
            wallet.load_seed(local_seed_path)

        log_dict = {
            "message": "Loaded wallet seed",
            "wallet_id": wallet_id,
            "wallet_address": str(wallet.default_address.address_id) if wallet.default_address else "",
        }
        logging.info(json_dumps(log_dict))

        # Generate new encryption key
        encryption_key = generate_encryption_key()
        # Store the wallet data with encryption
        await store_wallet_data(wallet, encryption_key)

        # Switch back to new API keys for future operations
        Cdp.configure(new_api_key_name, new_private_key)
        log_dict = {
            "message": "Configured CDP with new API keys",
            "api_key_name": new_api_key_name[-7:] if new_api_key_name else "",
            "api_private_key": new_private_key[-7:] if new_private_key else "",
        }
        logging.info(json_dumps(log_dict))
        wallet_data = await load_wallet_data(encryption_key)
        log_dict = {
            "message": "Loaded wallet data",
        }
        logging.info(json_dumps(log_dict))
        imported_wallet = Wallet.import_data(wallet_data)
        log_dict = {
            "message": "Imported wallet data",
            "wallet_id": wallet_id,
            "wallet_address": str(imported_wallet.default_address.address_id)
            if imported_wallet.default_address
            else "",
        }
        logging.info(json_dumps(log_dict))

        eth_balance = imported_wallet.balance("eth")
        usdc_balance = imported_wallet.balance("usdc")
        cbbtc_balance = imported_wallet.balance("cbbtc")

        log_dict = {
            "message": "Successfully migrated wallet credentials",
            "wallet_id": wallet_id,
            "wallet_address": str(imported_wallet.default_address.address_id)
            if imported_wallet.default_address
            else "",
            "eth_balance": str(eth_balance),
            "usdc_balance": str(usdc_balance),
            "cbbtc_balance": str(cbbtc_balance),
        }
        logging.info(json_dumps(log_dict))
        return None

    except Exception as e:
        log_dict = {"message": "Failed to migrate wallet credentials", "wallet_id": wallet_id, "error": str(e)}
        logging.error(json_dumps(log_dict))
        raise ValueError("Failed to migrate wallet credentials") from e
