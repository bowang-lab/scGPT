__version__ = "0.1.0"
import logging
import sys

logger = logging.getLogger("scGPT")
# check if logger has been initialized
if not logger.hasHandlers():
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(name)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

from . import model, tokenizer, scbank, utils
from .data_collator import DataCollator
from .data_sampler import SubsetsBatchSampler
