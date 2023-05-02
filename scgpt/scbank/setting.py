from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union


@dataclass
class Setting:
    """
    The configuration for scBank :class:`DataBank`.
    """

    remove_zero_rows: bool = field(
        default=True,
        metadata={
            "help": "When load data from numpy or sparse matrix, "
            "whether to remove rows with zero values."
        },
    )
    max_tokenize_batch_size: int = field(
        default=1e6,
        metadata={
            "help": "Maximum number of cells to tokenize in a batch. "
            "May be useful for processing numpy arrays, currently not used."
        },
    )
    immediate_save: bool = field(
        default=False,
        metadata={
            "help": "Whether to save DataBank whenever it is initiated or updated."
        },
    )
