import json
from pathlib import Path
from dataclasses import InitVar, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union
from typing_extensions import Self, Literal

import numpy as np
from scipy.sparse import spmatrix, csr_matrix
from anndata import AnnData
from datasets import Dataset, load_dataset
from scvi import settings

from scgpt.tokenizer import GeneVocab
from .data import DataTable, MetaInfo
from .setting import Setting
from . import logger


@dataclass
class DataBank:
    """
    The data structure for large-scale single cell data containing multiple studies.
    See https://github.com/subercui/scGPT-release#the-data-structure-for-large-scale-computing.
    """

    meta_info: MetaInfo = None
    data_tables: Dict[str, DataTable] = field(
        default_factory=dict,
        metadata={"help": "Data tables in the DataBank."},
    )
    gene_vocab: InitVar[GeneVocab] = field(
        default=None,
        # Because of the issue https://github.com/python/cpython/issues/94067, the
        # attribute can not be correctly set with default value, if using a getter
        # and setter, i.e. @property. So just explicityly set it in the __post_init__.
        metadata={"help": "Gene vocabulary mapping gene tokens to integer ids."},
    )
    settings: Setting = field(
        default_factory=Setting,
        metadata={"help": "The settings for scBank, use default if not set."},
    )

    def __post_init__(self, gene_vocab) -> None:
        """
        Initialize a DataBank. If initialize a non-empty DataBank, will check the
        meta info and import the data tables. The main data table idicator will be
        required, if at least one data table is provided.
        """
        # primarily like the DataSets here, should support slicing, indexing, map, etc.

        # also, because DataBank is desined to work with large-scale data. The class
        # instance itself works like a scheme framework that stores the metadata of
        # complete data to easily show or print the info. In the meantime, the actual
        # data chunks are kept synchronized in disk, and the DataBank will efficiently
        # load and return the partion in memory when queried, in the memory and
        # computational effective format of Datasets or Apache Arrow.

        # DataBank always tracks the current accessed and queried table/partition,
        # and the changes made. When trying to close the current table or switching
        # to another one, DataBank will explicitly ask whether to save the changes
        # to disk, and show the info of the updated table fields.

        if isinstance(gene_vocab, property):
            # walkaround for the issue https://github.com/python/cpython/issues/94067
            self._gene_vocab = None
        else:
            self.gene_vocab = gene_vocab

        # empty initialization:
        if self.meta_info is None:
            if len(self.data_tables) > 0:
                raise ValueError("Need to provide meta info if non-empty data tables.")
            if self.gene_vocab is not None:
                raise ValueError("Need to provide meta info if non-empty gene vocab.")
            logger.debug("Initialize an empty DataBank.")
            return

        # only-meta-info initializtion:
        if len(self.data_tables) == 0 and self.gene_vocab is None:
            logger.debug("DataBank initialized with meta info only.")
            self.sync() if self.settings.immediate_save else self.track()
            return

        # full initialization:
        if self.gene_vocab is None:
            raise ValueError("Need to provide gene vocab if non-empty data tables.")
        # validate the meta info, and the consistency between meta and data tables
        # we assume the input meta_info is complete and correct. Usually this should
        # be handled by factory constructors.
        self._validate_data()
        self.sync() if self.settings.immediate_save else self.track()

    @property
    def gene_vocab(self) -> Optional[GeneVocab]:
        """
        The gene vocabulary mapping gene tokens to integer ids.
        """
        # if self._gene_vocab is None:
        #     self._gene_vocab = GeneVocab.load(self.meta_info.gene_vocab_path)
        return self._gene_vocab

    @gene_vocab.setter
    def gene_vocab(self, gene_vocab: Union[GeneVocab, Path, str]) -> None:
        """
        Set the gene vocabulary from an :obj:`GeneVocab` object or a file path.
        """
        if isinstance(gene_vocab, (Path, str)):
            gene_vocab = GeneVocab.from_file(gene_vocab)
        elif not isinstance(gene_vocab, GeneVocab):
            raise ValueError("gene_vocab must be a GeneVocab object or a path.")

        self._validate_vocab(gene_vocab)  # e.g. check md5 every time calling setter
        self._gene_vocab = gene_vocab
        self.sync(  # sync to disk, update md5 in meta info
            attr_keys=["gene_vocab"]
        ) if self.settings.immediate_save else self.track(["gene_vocab"])

    @property
    def main_table_key(self) -> Optional[str]:
        """
        The main data table key.
        """
        if self.meta_info is None:
            return None
        return self.meta_info.main_table_key

    @main_table_key.setter
    def main_table_key(self, table_key: str) -> None:
        """Set the main data table key."""
        if self.meta_info is None:
            raise ValueError("Need to have self.meta_info if setting main table key.")
        self.meta_info.main_table_key = table_key
        self.sync(["meta_info"]) if self.settings.immediate_save else self.track(
            ["meta_info"]
        )

    @property
    def main_data(self) -> DataTable:
        """The main data table."""
        return self.data_tables[self.main_table_key]

    @classmethod
    def from_anndata(
        cls,
        adata: Union[AnnData, Path, str],
        vocab: Union[GeneVocab, Mapping[str, int]],
        to: Union[Path, str],
        main_table_key: str = "X",
        token_col: str = "gene name",
        immediate_save: bool = True,
    ) -> Self:
        """
        Create a DataBank from an AnnData object.

        Args:
            adata (AnnData): Annotated data or path to anndata file.
            vocab (GeneVocab or Mapping[str, int]): Gene vocabulary maps gene
                token to index.
            to (Path or str): Data directory.
            main_table_key (str): This layer/obsm in anndata will be used as the
                main data table.
            token_col (str): Column name of the gene token.
            immediate_save (bool): Whether to save the data immediately after creation.

        Returns:
            DataBank: DataBank instance.
        """

        if isinstance(adata, str) or isinstance(adata, Path):
            import scanpy as sc

            adata = sc.read(adata, cache=True)
        elif not isinstance(adata, AnnData):
            raise ValueError("adata must points to an AnnData object.")

        if isinstance(vocab, (Path, str)):
            vocab = GeneVocab.from_file(vocab)
        elif isinstance(vocab, Mapping) and not isinstance(vocab, GeneVocab):
            vocab = GeneVocab.from_dict(vocab)
        elif not isinstance(vocab, GeneVocab):
            raise ValueError("vocab must be a GeneVocab object or a mapping.")

        if isinstance(to, str):
            to = Path(to)
        to.mkdir(parents=True, exist_ok=True)
        db = cls(
            meta_info=MetaInfo(on_disk_path=to),
            gene_vocab=vocab,
            settings=Setting(immediate_save=immediate_save),
        )

        # TODO: Add other data tables, currently only read the main data
        data_table = db.load_anndata(
            adata,
            data_keys=[main_table_key],
            token_col=token_col,
        )[0]
        # update and immediate save
        db.main_table_key = main_table_key
        db.update_datatables(new_tables=[data_table], immediate_save=immediate_save)

        return db

    @classmethod
    def batch_from_anndata(cls, adata: List[AnnData], to: Union[Path, str]) -> Self:
        raise NotImplementedError

    @classmethod
    def from_path(cls, path: Union[Path, str]) -> Self:
        """
        Create a DataBank from a directory containing scBank data. **NOTE**: this
        method will automatically check whether md5sum record in the :file:`manifest.json`
        matches the md5sum of the loaded gene vocabulary.

        Args:
            path (Path or str): Directory path.

        Returns:
            DataBank: DataBank instance.
        """
        if isinstance(path, str):
            path = Path(path)
        if not path.exists():
            raise ValueError(f"Path {path} does not exist.")
        if not path.is_dir():
            raise ValueError(f"Path {path} is not a directory.")

        if not path.joinpath("gene_vocab.json").exists():
            logger.warning(f"DataBank at {path} does not contain gene_vocab.json.")
        data_table_files = [f for f in path.glob("*.datatable.*") if f.is_file()]
        if len(data_table_files) == 0:
            logger.warning(f"Loading empty DataBank at {path} without datatables.")

        db = cls(meta_info=MetaInfo.from_path(path))
        db.gene_vocab = GeneVocab.from_file(path / "gene_vocab.json")
        data_format = db.meta_info.on_disk_format
        for data_table_file in data_table_files:
            logger.info(f"Loading datatable {data_table_file}.")
            data_table = DataTable(
                name=data_table_file.name.split(".")[0],
                data=load_dataset(
                    data_format,
                    data_files=str(data_table_file),
                    cache_dir=str(path),
                    split="train",
                ),
            )
            db.update_datatables(new_tables=[data_table])
        return db

    def __len__(self) -> int:
        """Return the number of cells in DataBank."""
        raise NotImplementedError

    def _load_anndata_layer(
        self,
        adata: AnnData,
        data_key: Optional[str] = "X",
        index_map: Optional[Mapping[int, int]] = None,
    ) -> Optional[Dataset]:
        """
        Load anndata layer as a :class:Dataset object.

        Args:
            adata (:class:`AnnData`): Annotated data object to load.
            data_key (:obj:`str`, optional): Data key to load, default to "X". The data
                key must be in :attr:`adata.X`, :attr:`adata.layers` or :attr:`adata.obsm`.
            index_map (:obj:`Mapping[int, int]`, optional): A mapping from old index
                to new index. If None, meaning the index is unchanged and the converted
                data rows do not have explicit keys.

        Returns:
            :class:`Dataset`: Dataset object loaded.
        """
        if not isinstance(adata, AnnData):
            raise ValueError("adata must be an AnnData object.")

        if index_map is None:
            # TODO: implement loading data w/ index unchanged, i.e. w/o explicit keys
            raise NotImplementedError

        if data_key == "X":
            data = adata.X
        elif data_key in adata.layers:
            data = adata.layers[data_key]
        elif data_key in adata.obsm:
            data = adata.obsm[data_key]
        else:
            logger.warning(f"Data key {data_key} not found, skip loading.")
            return None

        tokenized_data = self._tokenize(data, index_map)

        return Dataset.from_dict(tokenized_data)

    def _tokenize(
        self,
        data: Union[np.ndarray, csr_matrix],
        ind2ind: Mapping[str, int],
        new_indices: Optional[List[int]] = None,
    ) -> Dict[str, List]:
        """
        Tokenize the data with the given vocabulary.

        TODO: currently by default uses the key-value datatable scheme. Add
        other support.

        TODO: currently just counting from zero for the cell id.

        Args:
            data (np.ndarray or spmatrix): Data to be tokenized.
            tokens (List[str]): List of gene tokens.
            ind2ind (Mapping[str, int]): Old index to new index mapping.
            new_indices (List[int]): New indices to be used, will ignore ind2ind.
                **NOTE**: Usually this should be None. Should only be used in the
                spawned recursive calls if any.

        Returns:
            Dict[str, List]: Tokenized data.
        """
        if not isinstance(data, (np.ndarray, csr_matrix)):
            raise ValueError("data must be a numpy array or sparse matrix.")

        if isinstance(data, np.ndarray):
            zero_ratio = np.sum(data == 0) / data.size
            if zero_ratio > 0.85:
                logger.debug(
                    f"{zero_ratio*100:.0f}% of the data is zero, "
                    "auto convert to sparse matrix before tokenizing."
                )
                data = csr_matrix(data)  # process sparse matrix actually faster

        # remove zero rows
        if self.settings.remove_zero_rows:
            if isinstance(data, np.ndarray) and data.size > 1e9:
                logger.warning(
                    "Going to remove zero rows from a large ndarray data. This "
                    "may take long time. If you want to disable this, set "
                    f"`remove_zero_rows` to False in {self.__name__}.settings."
                )
            if isinstance(data, csr_matrix):
                data = data[data.getnnz(axis=1) > 0]
            else:
                data = data[~np.all(data == 0, axis=1)]

        n_rows, n_cols = data.shape  # usually means n_cells, n_genes
        if new_indices is None:
            new_indices = np.array(
                [ind2ind.get(i, -100) for i in range(n_cols)], int
            )  # TODO: can be accelerated
        else:
            assert len(new_indices) == n_cols

        if isinstance(data, csr_matrix):
            indptr = data.indptr
            indices = data.indices
            non_zero_data = data.data

            tokenized_data = {"id": [], "genes": [], "expressions": []}
            tokenized_data["id"] = list(range(n_rows))
            for i in range(n_rows):  # ~2s/100k cells
                row_indices = indices[indptr[i] : indptr[i + 1]]
                row_new_indices = new_indices[row_indices]
                row_non_zero_data = non_zero_data[indptr[i] : indptr[i + 1]]

                match_mask = row_new_indices != -100
                row_new_indices = row_new_indices[match_mask]
                row_non_zero_data = row_non_zero_data[match_mask]

                tokenized_data["genes"].append(row_new_indices)
                tokenized_data["expressions"].append(row_non_zero_data)
        else:
            # remove -100 cols in data and new_indices, i.e. genes not in vocab
            mask = new_indices != -100
            new_indices = new_indices[mask]
            data = data[:, mask]
            tokenized_data = _nparray2mapped_values(
                data, new_indices, "numba"
            )  # ~7s/100k cells

        return tokenized_data

    def _validate_vocab(self, vocab: Optional[GeneVocab] = None) -> None:
        """
        Validate the vocabulary. Check :attr:`self.vocab` if no vocab is given.
        """
        if vocab is None:
            vocab = self.gene_vocab
        # TODO: implement validation

    def _validate_data(self) -> None:
        """
        Validate the current DataBank, including checking md5sum, table names, etc.
        """
        # first validate vocabulary
        self._validate_vocab()

        if len(self.data_tables) == 0 and self.main_table_key is not None:
            raise ValueError(
                "No data tables found, but main table key is set. "
                "Please set main table key to None or add data tables."
            )

        if len(self.data_tables) > 0:
            if self.main_table_key is None:
                raise ValueError(
                    "Main table key can not be empty if non-empty data tables."
                )
            if self.main_table_key not in self.data_tables:
                raise ValueError(
                    "Main table key {self.main_table_key} not found in data tables."
                )

    def update_datatables(
        self,
        new_tables: List[DataTable],
        use_names: List[str] = None,
        overwrite: bool = False,
        immediate_save: Optional[bool] = None,
    ) -> None:
        """
        Update the data tables in the DataBank with new data tables.

        Args:
            new_tables (list of :class:`DataTable`): New data tables to update.
            use_names (list of :obj:`str`): Names of the new data tables to use.
                If not provided, will use the names of the new data tables.
            overwrite (:obj:`bool`): Whether to overwrite the existing data tables.
            immediate_save (:obj:`bool`): Whether to save the data immediately after
                updating. Will save to :attr:`self.meta_info.on_disk_path`. If not
                provided, will follow :attr:`self.settings.immediate_save` instead.
                Default to None.
        """
        if not isinstance(new_tables, list) or not all(
            isinstance(t, DataTable) for t in new_tables
        ):
            raise ValueError("new_tables must be a list of DataTable.")

        if use_names is None:
            use_names = [t.name for t in new_tables]
        else:
            if len(use_names) != len(new_tables):
                raise ValueError("use_names must have the same length as new_tables.")

        if not overwrite:
            overlaps = set(use_names) & set(self.data_tables.keys())
            if len(overlaps) > 0:
                raise ValueError(
                    f"Data table names {overlaps} already exist in the DataBank. "
                    "Please set overwrite=True if replacing the existing data table."
                )

        if immediate_save is None:
            immediate_save = self.settings.immediate_save

        for dt, name in zip(new_tables, use_names):
            self.data_tables[name] = dt

        self._validate_data()
        self.sync("data_tables") if immediate_save else self.track("data_tables")

    def load_anndata(
        self,
        adata: AnnData,
        data_keys: Optional[List[str]] = None,
        token_col: str = "gene name",
    ) -> List[DataTable]:
        """
        Load anndata into datatables.

        Args:
            adata (:class:`AnnData`): Annotated data object to load.
            data_keys (list of :obj:`str`): List of data keys to load. If None,
                all data keys in :attr:`adata.X`, :attr:`adata.layers` and
                :attr:`adata.obsm` will be loaded.
            token_col (:obj:`str`): Column name of the gene token. Tokens will be
                converted to indices by :attr:`self.gene_vocab`.

        Returns:
            list of :class:`DataTable`: List of data tables loaded.
        """
        if not isinstance(adata, AnnData):
            raise ValueError("adata must be an AnnData object.")

        if data_keys is None:
            data_keys = ["X"] + list(adata.layers.keys()) + list(adata.obsm.keys())

        if token_col not in adata.var:
            raise ValueError(f"token_col {token_col} not found in adata.var.")
        if not isinstance(adata.var[token_col][0], str):
            raise ValueError(f"token_col {token_col} must be of type str.")

        # validate matching between tokens and vocab
        tokens = adata.var[token_col].tolist()
        match_ratio = sum([1 for t in tokens if t in self.gene_vocab]) / len(tokens)
        if match_ratio < 0.9:
            raise ValueError(
                f"{match_ratio*100:.0f}% of the tokens in adata.var[{token_col}] are not in "
                "vocab. Please check if using the correct vocab and token_col."
            )

        # build mapping to scBank datatable keys
        _ind2ind = _map_ind(tokens, self.gene_vocab)  # old index to new index

        data_tables = []
        for key in data_keys:
            data = self._load_anndata_layer(adata, key, _ind2ind)
            data_table = DataTable(
                name=key,
                data=data,
            )
            data_tables.append(data_table)

        return data_tables

    def append_study(
        self,
        study_id: int,
        study_data: Union[AnnData, "DataBank"],
    ) -> None:
        """
        Append a study to the current DataBank.

        Args:
            study_id (str): Study ID.
            study_data (AnnData or DataBank): Study data.
        """
        raise NotImplementedError

    def delete_study(self, study_id: int) -> None:
        """
        Delete a study from the current DataBank.
        """
        raise NotImplementedError

    def filter(
        self,
        study_ids: Optional[List[int]] = None,
        cell_ids: Optional[List[int]] = None,
        inplace: bool = True,
    ) -> Self:
        """
        Filter the current DataBank by study ID and cell ID.

        Args:
            study_ids (list): Study IDs to filter.
            cell_ids (list): Cell IDs to filter.
            inplace (bool): Whether to also filter inplace.

        Returns:
            DataBank: Filtered DataBank.
        """
        raise NotImplementedError

    def custom_filter(
        self,
        field: str,
        filter_func: callable,
        inplace: bool = True,
    ) -> Self:
        """
        Filter the current DataBank by applying a custom filter function to a field.

        Args:
            field (str): Field to filter.
            filter_func (callable): Filter function.
            inplace (bool): Whether to also filter inplace.

        Returns:
            DataBank: Filtered DataBank.
        """
        raise NotImplementedError

    def link(self, data_path: Union[Path, str]) -> None:
        """
        Link to a scBank data directory. This will only load the meta info
        and perform validation check, but not load the data tables. Usually, can
        use the `.load_table` method to load a data table later.
        """
        if isinstance(path, str):
            path = Path(path)
        assert path.is_dir(), f"Data path {path} should be a directory."
        raise NotImplementedError

    def load_table(self, table_name: str) -> Dataset:
        """
        Load a data table from the current DataBank.
        """
        raise NotImplementedError

    def load(self, path: Union[Path, str]) -> Dataset:
        """
        Load scBank data from a data directory. Since DataBank is designed to work
        with large-scale data, this only loads the main data table to memory by
        default. This does as well load the meta info and perform validation check.
        """
        if isinstance(path, str):
            path = Path(path)
        assert path.is_dir(), f"Data path {path} should be a directory."
        raise NotImplementedError

    def load_all(self, path: Union[Path, str]) -> Dict[str, Dataset]:
        """
        Load scBank data from a data directory. This will load all the data tables
        to memory.
        """
        if isinstance(path, str):
            path = Path(path)
        assert path.is_dir(), f"Data path {path} should be a directory."
        raise NotImplementedError

    def track(self, attr_keys: Union[List[str], str, None] = None) -> List:
        """
        Track all the changes made to the current DataBank and that have not been
        synced to disk. This will return a list of changes.

        Args:
            attr_keys (list of :obj:`str`): List of attribute keys to look for
                changes. If None, all attributes will be checked.
        """
        if attr_keys is None:
            attr_keys = [
                "meta_info",
                "data_tables",
                "gene_vocab",
            ]
        elif isinstance(attr_keys, str):
            attr_keys = [attr_keys]

        changed_attrs = []
        # TODO: implement the checking of the changes, currently just return all
        changed_attrs.extend(attr_keys)
        return changed_attrs

    def sync(self, attr_keys: Union[List[str], str, None] = None) -> None:
        """
        Sync the current DataBank to a data directory, including, save the updated
        data/vocab to files, update the meta info and save to files.
        **NOTE**: This will overwrite the existing data directory.

        Args:
            attr_keys (list of :obj:`str`): List of attribute keys to sync. If None, will
                sync all the attributes with tracked changes.
        """
        if attr_keys is None:
            attr_keys = self.track()
        elif isinstance(attr_keys, str):
            attr_keys = [attr_keys]

        # TODO: implement. Remeber particularly to update md5 in metainfo when
        # updating the gene vocabulary.

        on_disk_path = self.meta_info.on_disk_path
        data_format = self.meta_info.on_disk_format
        if "meta_info" in attr_keys:
            self.meta_info.save(on_disk_path)
        if "data_tables" in attr_keys:
            for data_table in self.data_tables.values():
                save_to = on_disk_path / f"{data_table.name}.datatable.{data_format}"
                logger.info(f"Saving data table {data_table.name} to {save_to}.")
                data_table.save(
                    path=save_to,
                    format=data_format,
                )
        if "gene_vocab" in attr_keys:
            if self.gene_vocab is not None:
                self._gene_vocab.save_json(on_disk_path / "gene_vocab.json")

    def save(self, path: Union[Path, str, None], replace: bool = False) -> None:
        """
        Save scBank data to a data directory.

        Args:
            path (Path): Path to save scBank data. If None, will save to the
                directory at :attr:`self.meta_info.on_disk_path`.
            replace (bool): Whether to replace existing data in the directory.
        """
        if path is None:
            path = self.meta_info.on_disk_path
        elif isinstance(path, str):
            path = Path(path)

        path.mkdir(parents=True, exist_ok=True)
        raise NotImplementedError


def _map_ind(tokens: List[str], vocab: Mapping[str, int]) -> Mapping[int, int]:
    """
    Create a mapping from old index to new index, for a list of tokens.

    Args:
        tokens (list): List of tokens, in the order of old indeces.
        vocab (Mapping[str, int]): Vocabulary mapping from token to new index.
    """
    ind2ind = {}
    unmatched_tokens = []
    for i, t in enumerate(tokens):
        if t in vocab:
            ind2ind[i] = vocab[t]
        else:
            unmatched_tokens.append(t)
    if len(unmatched_tokens) > 0:
        logger.warning(
            f"{len(unmatched_tokens)}/{len(tokens)} tokens/genes unmatched "
            "during vocabulary conversion."
        )

    return ind2ind


def _nparray2mapped_values(
    data: np.ndarray,
    new_indices: np.ndarray,
    mode: Literal["plain", "numba"] = "plain",
) -> Dict[str, List]:
    """
    Convert a numpy array to mapped values. Only include the non-zero values.

    Args:
        data (np.ndarray): Data matrix.
        new_indices (np.ndarray): New indices.

    Returns:
        Dict[str, List]: Mapping from column name to list of values.
    """
    if mode == "plain":
        convert_func = _nparray2indexed_values
    elif mode == "numba":
        convert_func = _nparray2indexed_values_numba
    else:
        raise ValueError(f"Unknown mode {mode}.")
    tokenized_data = {}
    row_ids, col_inds, values = convert_func(data, new_indices)

    tokenized_data["id"] = row_ids
    tokenized_data["genes"] = col_inds
    tokenized_data["expressions"] = values
    return tokenized_data


def _nparray2indexed_values(
    data: np.ndarray,
    new_indices: np.ndarray,
) -> Tuple[List, List, List]:
    """
    Convert a numpy array to indexed values. Only include the non-zero values.

    Args:
        data (np.ndarray): Data matrix.
        new_indices (np.ndarray): New indices.

    Returns:
        Tuple[List, List, List]: Row IDs, column indices, and values.
    """
    row_ids, col_inds, values = [], [], []
    for i in range(len(data)):  # TODO: accelerate with numba? joblib?
        row = data[i]
        idx = np.nonzero(row)[0]
        expressions = row[idx]
        genes = new_indices[idx]

        row_ids.append(i)
        col_inds.append(genes)
        values.append(expressions)

    return row_ids, col_inds, values


from numba import jit, njit, prange


@njit(parallel=True)
def _nparray2indexed_values_numba(
    data: np.ndarray,
    new_indices: np.ndarray,
) -> Tuple[List, List, List]:
    """
    Convert a numpy array to indexed values. Only include the non-zero values.
    Using numba to accelerate.

    Args:
        data (np.ndarray): Data matrix.
        new_indices (np.ndarray): New indices.

    Returns:
        Tuple[List, List, List]: Row IDs, column indices, and values.
    """
    row_ids, col_inds, values = (
        [1] * len(data),
        [np.empty(0, dtype=np.int64)] * len(data),
        [np.empty(0, dtype=data.dtype)] * len(data),
    )  # placeholders
    for i in prange(len(data)):
        row = data[i]
        idx = np.nonzero(row)[0]
        expressions = row[idx]
        genes = new_indices[idx]

        row_ids[i] = i
        col_inds[i] = genes
        values[i] = expressions

    return row_ids, col_inds, values
