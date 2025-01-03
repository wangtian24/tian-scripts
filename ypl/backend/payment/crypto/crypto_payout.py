import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Final

import httpx
from cdp import Cdp, Transfer, Wallet
from dotenv import load_dotenv
from ypl.backend.config import settings
from ypl.backend.llm.utils import post_to_slack
from ypl.backend.utils.files import (
    download_gcs_to_local_temp,
    file_exists,
    read_file,
)
from ypl.backend.utils.json import json_dumps
from ypl.db.payments import CurrencyEnum, PaymentTransactionStatusEnum

POLL_INTERVAL_SECONDS = 5
MAX_WAIT_TIME_SECONDS = 60

SEED_FILE_NAME = "encrypted_seed.json"
WALLET_FILE_NAME = "wallet.json"
CRYPTO_WALLET_PATH = settings.CRYPTO_WALLET_PATH
SEED_FILE_PATH = os.path.join(CRYPTO_WALLET_PATH, SEED_FILE_NAME)
WALLET_FILE_PATH = os.path.join(CRYPTO_WALLET_PATH, WALLET_FILE_NAME)

MIN_BALANCES: dict[CurrencyEnum, Decimal] = {
    CurrencyEnum.ETH: Decimal(0.25),
    CurrencyEnum.USDC: Decimal(200),
}

# Rate limiter for Basescan API - max 4 calls per second
BASESCAN_RATE_LIMIT: Final[int] = 4
_basescan_semaphore = asyncio.Semaphore(BASESCAN_RATE_LIMIT)
_last_call_time = 0.0


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

    async def get_asset_balance(self, asset_id: str) -> Decimal:
        if not self.wallet:
            await self._init_wallet()

        return self.wallet.balance(asset_id) if self.wallet else Decimal("0")

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

        if not self.wallet:
            await self._init_wallet()

        while time.time() - start_time < MAX_WAIT_TIME_SECONDS:
            attempts += 1
            asset_balance = self.wallet.balance(asset_id) if self.wallet else Decimal("0")

            # Check for low balance condition
            min_balance = MIN_BALANCES.get(CurrencyEnum(asset_id.upper()), Decimal("0"))
            if asset_balance < min_balance + total_required and settings.ENVIRONMENT == "production":
                message = (
                    f"🔴 *Low Balance Alert*\n"
                    f"Asset: {asset_id.upper()}\n"
                    f"Current Balance: {asset_balance}\n"
                    f"Current Transaction Required: {total_required}\n"
                    f"Minimum Required: {min_balance}"
                )
                asyncio.create_task(post_to_slack(message))

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

    async def process_reward(self, reward: CryptoReward) -> tuple[str | None, Transfer]:
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
            if (
                reward.asset_id.lower() == CurrencyEnum.USDC.value.lower()
                or reward.asset_id.lower() == CurrencyEnum.CBBTC.value.lower()
            ):
                transfer = self.wallet.transfer(
                    amount=reward.amount, asset_id=reward.asset_id, destination=reward.wallet_address, gasless=True
                )
            else:
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
            return str(transfer.transaction_hash) if transfer.transaction_hash else None, transfer

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


async def process_single_crypto_reward(reward: CryptoReward) -> tuple[str | None, Transfer]:
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


async def get_crypto_balance(asset_id: CurrencyEnum) -> Decimal:
    """Get the balance of crypto currency in the system wallet."""
    processor = await get_processor()
    return await processor.get_asset_balance(asset_id.value.lower())


async def get_transaction_status(transaction_hash: str) -> PaymentTransactionStatusEnum:
    """Get the status of a transaction in the blockchain using Basescan API.

    Args:
        transaction_hash: The transaction hash to check

    Returns:
        PaymentTransactionStatusEnum: Transaction status ('success', 'fail', or 'pending')
    """
    if not settings.BASESCAN_API_KEY or not settings.BASESCAN_API_URL:
        raise CryptoWalletError("Basescan API key not configured")

    url = settings.BASESCAN_API_URL
    params = {
        "module": "transaction",
        "action": "gettxreceiptstatus",
        "txhash": transaction_hash,
        "apikey": settings.BASESCAN_API_KEY,
    }

    try:
        # Acquire semaphore to enforce rate limit
        async with _basescan_semaphore:
            # Limit max of 4 calls per second
            global _last_call_time
            current_time = asyncio.get_event_loop().time()
            time_since_last_call = current_time - _last_call_time
            if time_since_last_call < 0.25:
                await asyncio.sleep(0.25 - time_since_last_call)
            _last_call_time = asyncio.get_event_loop().time()

            async with httpx.AsyncClient() as client:
                response = await client.get(url, params=params)
                if response.status_code != 200:
                    log_dict = {
                        "message": "Basescan API request failed",
                        "status_code": response.status_code,
                        "transaction_hash": transaction_hash,
                    }
                    logging.error(json_dumps(log_dict))

                data = response.json()
                if data.get("status") == "1":
                    # status "1" means success, "0" means fail
                    tx_status = data.get("result", {}).get("status", "")
                    if tx_status == "1":
                        return PaymentTransactionStatusEnum.SUCCESS
                    elif tx_status == "0":
                        return PaymentTransactionStatusEnum.FAILED

                return PaymentTransactionStatusEnum.PENDING

    except Exception as e:
        log_dict = {
            "message": "Failed to get transaction status",
            "transaction_hash": transaction_hash,
            "error": str(e),
        }
        logging.error(json_dumps(log_dict))
        raise CryptoWalletError(f"Failed to get transaction status: {str(e)}") from e
