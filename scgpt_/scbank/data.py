import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Union
from typing_extensions import Self, Literal

from datasets import Dataset


@dataclass
class DataTable:
    """
    The data structure for a single-cell data table.
    """

    name: str
    data: Optional[Dataset] = None

    @property
    def is_loaded(self) -> bool:
        return self.data is not None and isinstance(self.data, Dataset)

    def save(
        self,
        path: Union[Path, str],
        format: Literal["json", "parquet"] = "json",
    ) -> None:
        if not self.is_loaded:
            raise ValueError("DataTable is not loaded.")

        if isinstance(path, str):
            path = Path(path)

        if format == "json":
            self.data.to_json(path)
        elif format == "parquet":
            self.data.to_parquet(path)
        else:
            raise ValueError(f"Unknown format: {format}")


@dataclass
class MetaInfo:
    """
    The data structure for meta info of a scBank data directory.
    """

    on_disk_path: Union[Path, str, None] = None
    on_disk_format: Literal["json", "parquet"] = "json"
    main_table_key: Optional[str] = None
    # TODO: use md5 to check the vocab file name on disk
    gene_vocab_md5: Optional[str] = None
    study_ids: Optional[List[int]] = field(
        default=None,
        metadata={"help": "List of study IDs"},
    )
    cell_ids: Optional[List[int]] = field(
        default=None,
        metadata={"help": "List of cell IDs"},
    )
    # md5: Optional[str] = field(
    #     default=None,
    #     metadata={"help": "MD5 hash of the gene vocabulary"},
    # )

    def __post_init__(self):
        if self.on_disk_path is not None:
            self.on_disk_path: Path = Path(self.on_disk_path)

    def save(self, path: Union[Path, str, None] = None) -> None:
        """
        Save meta info to path. If path is None, will save to the same path at
        :attr:`on_disk_path`.
        """
        if path is None:
            path = self.on_disk_path

        if isinstance(path, str):
            path = Path(path)

        manifests = {
            "on_disk_format": self.on_disk_format,
            "main_data": self.main_table_key,
            "gene_vocab_md5": self.gene_vocab_md5,
        }
        with open(path / "manifest.json", "w") as f:
            json.dump(manifests, f, indent=2)

        # TODO: currently only save study table, add saving other tables
        with open(path / "studytable.json", "w") as f:
            json.dump({"study_ids": self.study_ids}, f, indent=2)

    def load(self, path: Union[Path, str, None] = None) -> None:
        """
        Load meta info from path. If path is None, will load from the same path
        at :attr:`on_disk_path`.
        """
        if path is None:
            path = self.on_disk_path

        if isinstance(path, str):
            path = Path(path)

        with open(path / "manifest.json") as f:
            manifests = json.load(f)
        self.on_disk_format = manifests["on_disk_format"]
        self.main_table_key = manifests["main_data"]
        self.gene_vocab_md5 = manifests["gene_vocab_md5"]

        if (path / "studytable.json").exists():
            with open(path / "studytable.json") as f:
                study_ids = json.load(f)
            self.study_ids = study_ids["study_ids"]

    @classmethod
    def from_path(cls, path: Union[Path, str]) -> Self:
        """
        Create a MetaInfo object from a path.
        """
        if isinstance(path, str):
            path = Path(path)

        if not path.exists():
            raise ValueError(f"Path {path} does not exist.")

        if not path.is_dir():
            raise ValueError(f"Path {path} is not a directory.")

        if not (path / "manifest.json").exists():
            raise ValueError(f"Path {path} does not contain manifest.json.")

        meta_info = cls()
        meta_info.on_disk_path = path
        meta_info.load(path)
        return meta_info
