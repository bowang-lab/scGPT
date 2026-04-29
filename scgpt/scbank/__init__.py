# scBank is the single cell data bank toolbox for building up large-scale single
# cell dataset to allow flexible cell data access and manipulation across studies,
# and to support large-scale computing.
import logging
import sys

logger = logging.getLogger("scBank")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(name)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
)
handler.setFormatter(formatter)
logger.addHandler(handler)

from .setting import Setting


def __getattr__(name):
    if name == "DataBank":
        from .databank import DataBank

        return DataBank
    if name == "DataTable":
        from .data import DataTable

        return DataTable
    if name == "MetaInfo":
        from .data import MetaInfo

        return MetaInfo
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
