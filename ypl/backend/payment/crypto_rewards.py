import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from decimal import Decimal

from cdp import Cdp, Transfer, Wallet
from dotenv import load_dotenv
from ypl.backend.config import settings
from ypl.backend.utils.files import (
    download_gcs_to_local_temp,
    file_exists,
    read_file,
)
from ypl.backend.utils.json import json_dumps

POLL_INTERVAL_SECONDS = 5
MAX_WAIT_TIME_SECONDS = 60

SEED_FILE_NAME = "encrypted_seed.json"
WALLET_FILE_NAME = "wallet.json"
CRYPTO_WALLET_PATH = settings.CRYPTO_WALLET_PATH
SEED_FILE_PATH = os.path.join(CRYPTO_WALLET_PATH, SEED_FILE_NAME)
WALLET_FILE_PATH = os.path.join(CRYPTO_WALLET_PATH, WALLET_FILE_NAME)


@dataclass
class CryptoReward:
    user_id: str
    wallet_address: str
    asset_id: str
    amount: Decimal


class CryptoWalletError(Exception):
    """Base exception for crypto wallet operations."""

    pass


class WalletNotFoundError(CryptoWalletError):
    """Raised when wallet files are not found."""

    pass


class CryptoRewardProcessor:
    """Handles cryptocurrency reward processing and wallet management."""

    def __init__(self) -> None:
        """Initialize the processor"""
        self.wallet: Wallet | None = None

    async def _init_cdp(self) -> None:
        """Initialize CDP configuration"""
        load_dotenv()
        api_key_name = settings.CDP_API_KEY_NAME
        api_key_private_key = settings.CDP_API_KEY_PRIVATE_KEY

        if not api_key_name or not api_key_private_key:
            log_dict = {
                "message": "Wallet configuration is missing",
                "error": "Missing CDP API key name or private key",
            }
            logging.exception(json_dumps(log_dict))
            return

        private_key = api_key_private_key.replace("\\n", "\n")
        Cdp.configure(api_key_name, private_key)

    async def _init_wallet(self) -> None:
        """Initialize sending wallet"""

        if not file_exists(SEED_FILE_PATH) or not file_exists(WALLET_FILE_PATH):
            log_dict = {
                "message": "Required wallet files not found",
                "seed_file_path": SEED_FILE_PATH,
                "wallet_file_path": WALLET_FILE_PATH,
            }
            logging.exception(json_dumps(log_dict))
            return

        self.wallet = self._import_existing_wallet()

    def _import_existing_wallet(self) -> Wallet:
        """Import an existing wallet"""

        wallet_data = read_file(WALLET_FILE_PATH)
        wallet_id = json.loads(wallet_data)

        with download_gcs_to_local_temp(SEED_FILE_PATH) as local_seed_path:
            wallet = Wallet.fetch(wallet_id)
            wallet.load_seed(local_seed_path)
            return wallet

    async def ensure_wallet_funded(self, total_required: Decimal, asset_id: str) -> bool:
        """
        Ensure wallet has sufficient funds for pending transfers.

        Args:
            total_required: Required amount in asset
            asset_id: The asset ID to check the balance of

        Returns:
            bool: True if wallet has sufficient funds, False otherwise

        Raises:
            ValueError: If wallet funding fails
        """
        start_time = time.time()
        attempts = 0

        while time.time() - start_time < MAX_WAIT_TIME_SECONDS:
            attempts += 1
            asset_balance = self.wallet.balance(asset_id) if self.wallet else Decimal("0")

            if asset_balance >= total_required:
                return True

            if asset_balance < total_required:
                log_dict = {
                    "message": "Not enough funds.",
                    "asset_id": asset_id,
                    "total_required": total_required,
                    "current_balance": asset_balance,
                    "attempt": attempts,
                }
                logging.info(json_dumps(log_dict))

                if settings.ENVIRONMENT != "production":
                    try:
                        self.wallet.faucet(asset_id) if self.wallet else None
                    except Exception as e:
                        log_dict = {
                            "message": "Failed to request funds from faucet",
                            "error": str(e),
                            "attempt": attempts,
                        }
                        logging.error(json_dumps(log_dict))
                        raise CryptoWalletError("Failed to request funds from faucet") from e
                else:
                    raise CryptoWalletError("Not enough funds in the wallet")

            # Wait before checking balance again
            await asyncio.sleep(POLL_INTERVAL_SECONDS)

        log_dict = {
            "message": "Timed out waiting for wallet funding",
            "timeout_seconds": MAX_WAIT_TIME_SECONDS,
            "total_attempts": attempts,
            "final_balance": self.wallet.balance(asset_id) if self.wallet else Decimal("0"),
            "required_balance": str(total_required),
            "deficit": str(total_required - (self.wallet.balance(asset_id) if self.wallet else Decimal("0"))),
        }
        logging.error(json_dumps(log_dict))
        return False

    async def process_reward(self, reward: CryptoReward) -> tuple[str, Transfer]:
        """Process a single crypto reward

        Returns:
            tuple[str, Transfer]: A tuple containing the transaction hash and transfer object
        """
        try:
            start_time = time.time()
            if not self.wallet:
                new_processor = await get_processor()
                if new_processor.wallet:
                    self.wallet = new_processor.wallet
                else:
                    raise CryptoWalletError("Wallet not initialized")

            await self.ensure_wallet_funded(reward.amount, reward.asset_id)
            transfer = self.wallet.transfer(
                amount=reward.amount, asset_id=reward.asset_id, destination=reward.wallet_address
            )

            if not transfer:
                raise CryptoWalletError("Transfer creation failed")

            end_time = time.time()
            duration = end_time - start_time
            log_dict = {
                "message": "Created transaction in the blockchain",
                "user_id": reward.user_id,
                "asset_id": reward.asset_id,
                "wallet_address": reward.wallet_address,
                "amount": str(reward.amount),
                "transaction_hash": transfer.transaction_hash,
                "transfer_status": transfer.status,
                "duration": str(duration),
            }
            logging.info(json_dumps(log_dict))
            return str(transfer.transaction_hash), transfer

        except Exception as e:
            log_dict = {
                "message": "Failed to process crypto reward",
                "user_id": reward.user_id,
                "asset_id": reward.asset_id,
                "wallet_address": reward.wallet_address,
                "amount": str(reward.amount),
                "error_message": str(e),
            }
            logging.error(json_dumps(log_dict))
            raise CryptoWalletError("Failed to process reward") from e


async def process_pending_crypto_rewards() -> None:
    """Process all pending crypto rewards"""
    processor = await get_processor()

    # Mock rewards array for testing
    pending_rewards = [
        CryptoReward(
            user_id="test_user_123",
            asset_id=os.getenv("ASSET_ID") or "",
            wallet_address=os.getenv("CHAOS_RECEIVER_ADDRESS") or "",
            amount=Decimal(os.getenv("DEFAULT_TRANSFER_AMOUNT") or "0"),
        )
    ]

    # Process each pending reward
    for reward in pending_rewards:
        await processor.process_reward(reward)


async def process_single_crypto_reward(reward: CryptoReward) -> tuple[str, Transfer]:
    """Process a single cryptocurrency reward payment.

    Args:
        reward: The CryptoReward object containing payment details

    Returns:
        tuple[str, Transfer]: A tuple containing the transaction hash and transfer object
    """
    processor = await get_processor()
    return await processor.process_reward(reward)


async def cleanup_crypto_processor() -> None:
    """Cleanup the processor instance during application shutdown"""
    global _processor_instance
    if _processor_instance is not None:
        # Add any cleanup code here if needed
        _processor_instance = None


_processor_instance: CryptoRewardProcessor | None = None
_initialization_lock: asyncio.Lock | None = None


async def get_processor() -> CryptoRewardProcessor:
    """Get or create a cached CryptoRewardProcessor instance"""
    global _processor_instance, _initialization_lock

    # Initialize the lock if needed
    if _initialization_lock is None:
        _initialization_lock = asyncio.Lock()

    # Fast path: return existing instance if available
    if _processor_instance is not None:
        return _processor_instance

    # Slow path: initialize with lock to prevent multiple simultaneous initializations
    async with _initialization_lock:
        # Check again in case another task initialized while we were waiting
        if _processor_instance is not None:
            return _processor_instance

        processor = CryptoRewardProcessor()
        await processor._init_cdp()
        await processor._init_wallet()
        _processor_instance = processor
        log_dict = {
            "message": "Crypto processor initialized",
        }
        logging.info(json_dumps(log_dict))
        return _processor_instance
