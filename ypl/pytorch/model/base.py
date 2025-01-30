import os
import zipfile
from typing import Self

import google.cloud.storage as storage
import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

from ypl.cache import get_cache_dir
from ypl.pytorch.data.base import StrTensorDict
from ypl.pytorch.torch_utils import DeviceMixin


# For now, assume that all models can fit on a single device.
class YuppModel(PyTorchModelHubMixin, DeviceMixin, nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @property
    def device(self) -> torch.device:
        self._device = next(self.parameters()).device
        return self._device

    @device.setter
    def device(self, device: torch.device) -> None:
        self.to(device)
        self._device = device

    @classmethod
    def from_gcp_zip(cls, gcp_path: str, download_always: bool = False) -> Self:
        """
        Downloads the zip from GCP, extracts it, and loads the model if it doesn't exist locally.

        Args:
            gcp_path: The path to the zip file on GCP, e.g., `gs://yupp-models/categorizer.zip`.
            download_always: If True, the zip file will be downloaded again, even if it already exists locally.

        Returns:
            The model loaded from the zip file.
        """
        assert gcp_path.startswith("gs://"), "GCP path must start with gs://"

        base_path, created = get_cache_dir().get(gcp_path, create_folder_if_not_exists=True)
        client = storage.Client()
        bucket_name, blob_name = gcp_path.split("gs://")[1].split("/", 1)
        blob = client.bucket(bucket_name).blob(blob_name)
        model_file_name = os.path.basename(blob_name)

        if not blob.exists():
            raise FileNotFoundError(f"File not found: {gcp_path}")

        model_path = base_path / model_file_name
        gcp_is_newer_version = False
        blob.reload()

        if model_path.exists() and model_path.stat().st_mtime < blob.updated.timestamp():
            gcp_is_newer_version = True

        if created or not model_path.exists() or download_always or gcp_is_newer_version:
            get_cache_dir().delete(gcp_path)  # clears the cache
            base_path, created = get_cache_dir().get(gcp_path, create_folder_if_not_exists=True)

            with (base_path / model_file_name).open("wb") as f:
                client.download_blob_to_file(gcp_path, f)

            with zipfile.ZipFile(base_path / model_file_name, "r") as zip_ref:
                zip_ref.extractall(base_path)

        return cls.from_pretrained(base_path)

    def push_to_gcp_zip(self, gcp_path: str) -> None:
        """Zips and pushes the model to GCP."""
        assert gcp_path.startswith("gs://"), "GCP path must start with gs://"

        bucket_name, blob_name = gcp_path.split("gs://")[1].split("/", 1)
        model_file_name = os.path.basename(blob_name)
        base_path, _ = get_cache_dir().get(gcp_path, create_folder_if_not_exists=True)
        self.save_pretrained(base_path)

        with zipfile.ZipFile(base_path / model_file_name, "w") as zip_ref:
            for file in base_path.iterdir():
                if file.name != model_file_name:
                    zip_ref.write(file, file.relative_to(base_path))

        storage.Client().bucket(bucket_name).blob(blob_name).upload_from_filename(base_path / model_file_name)

    def forward(self, batch: StrTensorDict) -> StrTensorDict:
        raise NotImplementedError


class YuppClassificationModel(YuppModel):
    def __init__(self, model_name: str, label_map: dict[str, int], multilabel: bool = False) -> None:
        super().__init__()
        self.model_name = model_name
        self.label_map = label_map
        self.multilabel = multilabel


class YuppBucketingModel(YuppModel):
    def __init__(self, model_name: str, buckets: list[int]) -> None:
        super().__init__()
        self.model_name = model_name
        self.buckets = buckets

    def bucket_index(self, response_length: int) -> int:
        for i, bucket in enumerate(self.buckets):
            if response_length <= bucket:
                return i

        return len(self.buckets) - 1
