import importlib.metadata

try:
    __version__ = importlib.metadata.version("yupp-llms")
except importlib.metadata.PackageNotFoundError:
    __version__ = "Not installed. Please install yupp-llms to enable single-source versioning."
