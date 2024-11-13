import importlib.metadata

try:
    __version__ = importlib.metadata.version("ypl")
except importlib.metadata.PackageNotFoundError:
    __version__ = "Not installed. Please install yupp-mind to enable single-source versioning."
