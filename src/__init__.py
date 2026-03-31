"""Pipeline package entrypoint."""
from .pipeline import process_batch
from .cli import main

__all__ = ["process_batch", "main"]
