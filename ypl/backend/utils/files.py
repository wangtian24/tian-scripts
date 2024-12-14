import logging
import os
import tempfile
from collections.abc import Generator
from contextlib import contextmanager

from google.cloud import storage


class FileStorage:
    """Handles file operations for both local and GCS storage."""

    @staticmethod
    def is_gcs_path(path: str) -> bool:
        return path.startswith("gs://")

    @staticmethod
    def read_file(path: str) -> str:
        try:
            if FileStorage.is_gcs_path(path):
                return FileStorage._read_gcs_file(path)
            return FileStorage._read_local_file(path)
        except Exception as e:
            logging.error(f"Error reading file {path}: {e}")
            return ""

    @staticmethod
    def gcs_write_file(gcs_path: str, content: str, content_type: str = "text/plain") -> None:
        """Write file to GCS

        Args:
            gcs_path: Path to GCS file (gs://bucket/path)
            content: Content to write
            content_type: Content type of file (default: text/plain)
        """
        try:
            bucket_name = gcs_path.split("//")[1].split("/")[0]
            blob_name = "/".join(gcs_path.split("//")[1].split("/")[1:])

            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_name)

            blob.content_type = content_type
            blob.upload_from_string(content, content_type=content_type, encoding="utf-8")
        except Exception as e:
            logging.error(f"Error writing GCS file: {e}")

    @staticmethod
    def _read_gcs_file(path: str) -> str:
        return str(gcs_read_file(path))

    @staticmethod
    def _read_local_file(path: str) -> str:
        with open(path, encoding="utf-8") as f:
            return str(f.read())


def file_exists(path: str) -> bool:
    """Check if file exists in GCS or local filesystem"""
    if FileStorage.is_gcs_path(path):
        return gcs_file_exists(path)
    return bool(os.path.exists(path))


def read_file(path: str) -> str:
    """Read file from GCS or local filesystem"""
    try:
        if FileStorage.is_gcs_path(path):
            return gcs_read_file(path)
        with open(path, encoding="utf-8") as f:
            return str(f.read())
    except Exception as e:
        logging.error(f"Error reading file {path}: {e}")
        return ""


def write_file(path: str, content: str) -> None:
    """Write file to GCS or local filesystem"""
    try:
        if FileStorage.is_gcs_path(path):
            FileStorage.gcs_write_file(path, content)
        else:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
    except Exception as e:
        logging.error(f"Error writing file {path}: {e}")


def gcs_file_exists(gcs_path: str) -> bool:
    """Check if file exists in GCS"""
    try:
        bucket_name = gcs_path.split("//")[1].split("/")[0]
        blob_name = "/".join(gcs_path.split("//")[1].split("/")[1:])

        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        return bool(blob.exists())
    except Exception as e:
        logging.error(f"Error checking GCS file: {e}")
        return False


def gcs_read_file(gcs_path: str) -> str:
    """Read text file from GCS"""
    try:
        bucket_name = gcs_path.split("//")[1].split("/")[0]
        blob_name = "/".join(gcs_path.split("//")[1].split("/")[1:])

        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        return str(blob.download_as_text(encoding="utf-8"))
    except Exception as e:
        logging.error(f"Error reading GCS file: {e}")
        return ""


@contextmanager
def download_gcs_to_local_temp(gcs_path: str) -> Generator[str, None, None]:
    """Download GCS file to temp directory and yield path in a context manager"""
    if not FileStorage.is_gcs_path(gcs_path):
        yield gcs_path
        return

    with tempfile.NamedTemporaryFile(mode="w", delete=True) as temp_file:
        try:
            content = gcs_read_file(gcs_path)
            temp_file.write(content)
            temp_file.flush()  # Ensure content is written
            yield temp_file.name
        except Exception as e:
            logging.error(f"Error downloading GCS file: {e}")
            raise
