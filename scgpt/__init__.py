__version__ = "0.1.2"
import logging
import sys

logger = logging.getLogger("scGPT")
# check if logger has been initialized
if not logger.hasHandlers() or len(logger.handlers) == 0:
    logger.propagate = False
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(name)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

from . import model, tokenizer, scbank, utils, tasks
from .data_collator import DataCollator
from .data_sampler import SubsetsBatchSampler
from .trainer import (
    prepare_data,
    prepare_dataloader,
    train,
    define_wandb_metrcis,
    evaluate,
    eval_testdata,
    test,
)
