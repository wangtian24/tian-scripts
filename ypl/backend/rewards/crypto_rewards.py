import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Any
from uuid import UUID

from cdp import Cdp, Wallet
from dotenv import load_dotenv
from ypl.backend.config import settings
from ypl.backend.utils.files import (
    download_gcs_to_local_temp,
    file_exists,
    read_file,
    write_file,
)
from ypl.backend.utils.json import json_dumps

# Constants
POLL_INTERVAL_SECONDS = 5
MAX_WAIT_TIME_SECONDS = 120

# File Paths
SEED_FILE_NAME = "encrypted_seed.json"
WALLET_FILE_NAME = "wallet.json"
CRYPTO_WALLET_PATH = settings.CRYPTO_WALLET_PATH
SEED_FILE_PATH = os.path.join(CRYPTO_WALLET_PATH, SEED_FILE_NAME)
WALLET_FILE_PATH = os.path.join(CRYPTO_WALLET_PATH, WALLET_FILE_NAME)


class CryptoRewardStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class CryptoReward:
    reward_id: UUID
    user_id: str
    wallet_address: str
    asset_id: str
    amount: Decimal
    status: CryptoRewardStatus
    transaction_hash: str | None = None
    error_message: str | None = None


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

    async def __aenter__(self) -> "CryptoRewardProcessor":
        await self._init_cdp()
        await self._init_wallet()
        return self

    async def __aexit__(self, exc_type: type | None, exc_val: Exception | None, exc_tb: Any | None) -> None:
        # Cleanup code here
        pass

    async def _init_cdp(self) -> None:
        """Initialize CDP configuration"""
        load_dotenv()
        api_key_name = settings.CDP_API_KEY_NAME
        api_key_private_key = settings.CDP_API_KEY_PRIVATE_KEY

        if not api_key_name or not api_key_private_key:
            raise ValueError("Wallet configuration is missing")

        # Configure the CDP SDK
        private_key = api_key_private_key.replace("\\n", "\n")
        Cdp.configure(api_key_name, private_key)

    async def _init_wallet(self) -> None:
        """Initialize sending wallet"""
        log_dict = {
            "message": "Initializing wallet",
            "seed_file_exists": file_exists(SEED_FILE_PATH),
            "wallet_file_exists": file_exists(WALLET_FILE_PATH),
        }
        logging.info(json_dumps(log_dict))

        if not file_exists(SEED_FILE_PATH) or not file_exists(WALLET_FILE_PATH):
            log_dict = {
                "message": "Required wallet files not found",
                "seed_file": SEED_FILE_PATH,
                "wallet_file": WALLET_FILE_PATH,
            }
            logging.exception(json_dumps(log_dict))
            raise WalletNotFoundError(
                f"Required wallet files not found. Seed: {SEED_FILE_PATH}, Wallet: {WALLET_FILE_PATH}"
            )

        self.wallet = self._import_existing_wallet()

    def _create_sending_wallet(self) -> Wallet:
        """Create a new sending wallet"""

        sending_wallet = Wallet.create()
        # Persist the wallet locally
        wallet_id_string = json.dumps(sending_wallet.id)
        write_file(WALLET_FILE_PATH, wallet_id_string)
        sending_wallet.save_seed(SEED_FILE_PATH)

        sending_address = sending_wallet.default_address
        log_dict = {
            "message": "Created wallet with address",
            "address": sending_address.address_id if sending_address else "",
        }
        logging.info(json_dumps(log_dict))
        return sending_wallet

    def _import_existing_wallet(self) -> Wallet:
        """Import an existing wallet"""
        log_dict = {"message": "Importing existing CDP wallet..."}
        logging.info(json_dumps(log_dict))

        # Get the wallet ID
        wallet_data = read_file(WALLET_FILE_PATH)
        wallet_id = json.loads(wallet_data)

        # Get seed file path - download if GCS, use directly if local
        with download_gcs_to_local_temp(SEED_FILE_PATH) as local_seed_path:
            # Get the wallet
            wallet = Wallet.fetch(wallet_id)
            wallet.load_seed(local_seed_path)
            return wallet

    async def ensure_wallet_funded(self, total_required: Decimal, asset_id: str) -> bool:
        """
        Ensure wallet has sufficient funds for pending transfers.

        Args:
            total_required: Required amount in asset

        Returns:
            bool: True if wallet has sufficient funds, False otherwise

        Raises:
            ValueError: If wallet funding fails
        """
        MAX_WAIT_TIME = 60  # 1 minute timeout
        POLL_INTERVAL = 5  # Check balance every 5 seconds

        start_time = time.time()
        attempts = 0

        while time.time() - start_time < MAX_WAIT_TIME:
            attempts += 1
            asset_balance = self.wallet.balance(asset_id) if self.wallet else Decimal("0")
            log_dict = {
                "message": f"Checking current {asset_id} balance",
                "balance": asset_balance,
                "required": str(total_required),
                "attempt": attempts,
                "elapsed_seconds": int(time.time() - start_time),
            }
            logging.info(json_dumps(log_dict))

            if asset_balance >= total_required:
                log_dict = {
                    "message": "Wallet has sufficient funds",
                    "balance": asset_balance,
                    "required": str(total_required),
                    "attempts_needed": attempts,
                }
                logging.info(json_dumps(log_dict))
                return True

            if asset_balance < total_required:
                log_dict = {
                    "message": f"Need {total_required} {asset_id}; attempting to fund wallet with faucet...",
                    "current_balance": asset_balance,
                    "attempt": attempts,
                }
                logging.info(json_dumps(log_dict))

                try:
                    faucet_transaction = self.wallet.faucet() if self.wallet else None
                    log_dict = {
                        "message": "Faucet transaction completed",
                        "transaction": faucet_transaction,
                        "attempt": attempts,
                    }
                    logging.info(json_dumps(log_dict))

                    new_asset_balance = self.wallet.balance(asset_id) if self.wallet else Decimal("0")
                    log_dict = {
                        "message": "New ETH balance after faucet",
                        "previous_balance": asset_balance,
                        "new_balance": new_asset_balance,
                        "required": str(total_required),
                        "attempt": attempts,
                    }
                    logging.info(json_dumps(log_dict))

                except Exception as e:
                    log_dict = {"message": "Failed to request funds from faucet", "error": str(e), "attempt": attempts}
                    logging.error(json_dumps(log_dict))

            # Wait before checking balance again
            await asyncio.sleep(POLL_INTERVAL)

        log_dict = {
            "message": "Timed out waiting for wallet funding",
            "timeout_seconds": MAX_WAIT_TIME,
            "total_attempts": attempts,
            "final_balance": self.wallet.balance(asset_id) if self.wallet else Decimal("0"),
            "required_balance": str(total_required),
            "deficit": str(total_required - (self.wallet.balance(asset_id) if self.wallet else Decimal("0"))),
        }
        logging.error(json_dumps(log_dict))
        return False

    async def process_reward(self, reward: CryptoReward) -> bool:
        """Process a single crypto reward"""
        try:
            log_dict = {
                "message": "Processing crypto reward",
                "reward_id": str(reward.reward_id),
                "user_id": reward.user_id,
                "wallet_address": reward.wallet_address,
                "amount": str(reward.amount),
            }
            logging.info(json_dumps(log_dict))

            # Attempt transfer
            transfer = (
                self.wallet.transfer(amount=reward.amount, asset_id=reward.asset_id, destination=reward.wallet_address)
                if self.wallet
                else None
            )

            # Wait for transfer with timeout
            transfer.wait()

            # Update reward with transaction details
            reward.status = CryptoRewardStatus.COMPLETED
            reward.transaction_hash = transfer.transaction_hash

            log_dict = {
                "reward_id": str(reward.reward_id),
                "user_id": reward.user_id,
                "transaction_hash": transfer.transaction_hash,
                "transaction_link": transfer.transaction_link,
                "status": reward.status,
            }
            logging.info(json_dumps(log_dict))
            return True

        except Exception as e:
            reward.status = CryptoRewardStatus.FAILED
            reward.error_message = str(e)
            log_dict = {
                "message": "Failed to process crypto reward",
                "reward_id": str(reward.reward_id),
                "user_id": reward.user_id,
                "error_message": str(e),
            }
            logging.error(json_dumps(log_dict))
            return False


async def process_pending_crypto_rewards() -> None:
    """Process all pending crypto rewards"""
    async with CryptoRewardProcessor() as processor:
        # TODO: Uncomment this once we have a database
        # engine = get_engine()
        # with Session(engine) as session:
        #     query = select(CryptoReward).where(
        #         CryptoReward.status == CryptoRewardStatus.PENDING
        #     )
        #     pending_rewards = session.exec(query).all()

        # Mock rewards array for testing
        pending_rewards = [
            CryptoReward(
                reward_id=UUID("12345678-1234-5678-1234-567812345678"),
                user_id="test_user_123",
                asset_id=os.getenv("ASSET_ID") or "",
                wallet_address=os.getenv("CHAOS_RECEIVER_ADDRESS") or "",
                amount=Decimal(os.getenv("DEFAULT_TRANSFER_AMOUNT") or Decimal("0")),
                status=CryptoRewardStatus.PENDING,
            )
        ]

        # Process each pending reward
        for reward in pending_rewards:
            await processor.process_reward(reward)
