from pathlib import Path
import tempfile

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from datasets import Dataset
from anndata import AnnData

from scgpt.tokenizer import GeneVocab
from scgpt.scbank import DataBank, DataTable, MetaInfo, Setting

tmp_dir = tempfile.gettempdir()
save_path = Path(tmp_dir) / "test_scGPT"
save_path.mkdir(parents=True, exist_ok=True)


def clear_files(directory: Path):
    """helper function to clear files in a dir"""
    for f in directory.iterdir():
        f.unlink()


def test_empty_databank():
    db = DataBank()
    assert db.data_tables == {}
    assert db.settings == Setting()
    assert db.gene_vocab == None

    db = DataBank(meta_info=MetaInfo())
    assert db.data_tables == {}
    assert db.gene_vocab == None

    db = DataBank(
        meta_info=MetaInfo(on_disk_path=save_path),
        settings=Setting(immediate_save=True),
    )
    assert (save_path / "studytable.json").is_file()
    assert (save_path / "manifest.json").is_file()
    clear_files(save_path)


def test_empty_datatable():
    dt = DataTable(name="test")
    assert dt.name == "test"
    assert dt.data is None
    assert not dt.is_loaded


def test_empty_metainfo():
    mi = MetaInfo()
    assert mi.study_ids is None
    assert mi.cell_ids is None


def test_save_load_metainfo():
    mi = MetaInfo(save_path)
    mi.save()
    assert (save_path / "studytable.json").is_file()
    assert (save_path / "manifest.json").is_file()
    assert MetaInfo.from_path(save_path) == mi

    clear_files(save_path)


def test_datatable_save():
    dt = DataTable(name="test")

    file_path = save_path / "test.json"
    # make sure the path does not exist originally
    assert not file_path.exists()
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # catch the exception if the data is not loaded
    with pytest.raises(ValueError):
        dt.save(file_path)

    # actually load some example data and test saving
    dt.data = Dataset.from_dict({"a": [1]})
    dt.save(file_path)
    assert file_path.is_file()

    # delete the file
    file_path.unlink()
    assert not file_path.exists()


def test_meta_info_on_disk_path():
    mi = MetaInfo(on_disk_path=tmp_dir)
    assert mi.on_disk_path == Path(tmp_dir)
    assert mi.on_disk_format == "json"


def test_add_gene_vocab():
    db = DataBank()
    db.gene_vocab = GeneVocab.from_dict({"a": 0, "b": 1, "c": 2})
    assert len(db.gene_vocab) == 3
    assert db.gene_vocab["a"] == 0
    assert "c" in db.gene_vocab

    with pytest.raises(ValueError):
        db.gene_vocab = ["a", "b", "c"]


def test_databank_tokenize():
    indptr = np.array([0, 2, 3, 6])
    indices = np.array([0, 2, 2, 0, 1, 2])
    data = np.array([1, 2, 3, 4, 5, 6])
    data = csr_matrix((data, indices, indptr), shape=(3, 3))
    # data is like:
    # [[1, 0, 2],
    #  [0, 0, 3],
    #  [4, 5, 6]]

    ind2ind = {0: 4, 2: 6}

    tokenized = DataBank()._tokenize(data, ind2ind)
    # tokenized is like:
    # {'id': [0, 1, 2],
    # 'genes': [[4, 6], [6], [4, 6]],
    # 'expressions': [[1, 2], [3], [4, 6]]}

    assert tokenized["id"] == [0, 1, 2]
    assert [d.tolist() for d in tokenized["genes"]] == [[4, 6], [6], [4, 6]]
    assert [d.tolist() for d in tokenized["expressions"]] == [[1, 2], [3], [4, 6]]

    # tokenize numpy array
    data = data.toarray()
    tokenized = DataBank()._tokenize(data, ind2ind)
    assert tokenized["id"] == [0, 1, 2]
    assert [d.tolist() for d in tokenized["genes"]] == [[4, 6], [6], [4, 6]]
    assert [d.tolist() for d in tokenized["expressions"]] == [[1, 2], [3], [4, 6]]

    # test array with rows and cols of only zeros
    data[:, 2] = 0
    tokenized = DataBank()._tokenize(data, ind2ind)
    assert tokenized["id"] == [0, 1]
    assert [d.tolist() for d in tokenized["genes"]] == [[4], [4]]
    assert [d.tolist() for d in tokenized["expressions"]] == [[1], [4]]

    # test array w/ rare non-zero values (_tokenize will auto convert it to sparse)
    data = np.zeros((3, 3))
    data[0, 0] = 1.0
    tokenized = DataBank()._tokenize(data, ind2ind)
    assert tokenized["id"] == [0]
    assert [d.tolist() for d in tokenized["genes"]] == [[4]]
    assert [d.tolist() for d in tokenized["expressions"]] == [[1.0]]

    # add the test for recursive batch calls


def test_databank_load_anndata():
    adata = AnnData(
        X=np.array([[1.0, 2.0, 3.0], [4.0, 0.0, 6.0]]),
        obs={"cell": ["cell1", "cell2"], "study": ["study1", "study2"]},
        var={"gene": ["gene_a", "gene_b", "gene_c"]},
    )
    gene_vocab = {"gene_a": 1, "gene_b": 0, "gene_c": 2}

    # test factory initialization from data
    db = DataBank.from_anndata(
        adata,
        gene_vocab,
        to=save_path,
        main_table_key="X",
        token_col="gene",
    )
    assert db.main_table_key == "X"

    converted_dataset = db.data_tables["X"].data
    assert converted_dataset["id"] == [0, 1]
    assert converted_dataset["genes"] == [[1, 0, 2], [1, 2]]
    assert converted_dataset["expressions"] == [[1.0, 2.0, 3.0], [4.0, 6.0]]

    assert (save_path / "X.datatable.json").is_file()
    assert (save_path / "studytable.json").is_file()
    assert (save_path / "manifest.json").is_file()
    assert (save_path / "gene_vocab.json").is_file()

    # test factory initialization from path
    db = DataBank.from_path(save_path)
    assert db.main_table_key == "X"

    main_dataset = db.data_tables["X"].data
    assert main_dataset["id"] == [0, 1]
    assert main_dataset["genes"] == [[1, 0, 2], [1, 2]]
    assert main_dataset["expressions"] == [[1.0, 2.0, 3.0], [4.0, 6.0]]

    clear_files(save_path)

    # test loading from anndata
    db = DataBank(
        meta_info=MetaInfo(on_disk_format=save_path),
        gene_vocab=GeneVocab.from_dict(gene_vocab),
    )
    data_tables = db.load_anndata(adata, data_keys=["X"], token_col="gene")
    assert len(data_tables) == 1

    converted_dataset = data_tables[0].data
    assert converted_dataset["id"] == [0, 1]
    assert converted_dataset["genes"] == [[1, 0, 2], [1, 2]]
    assert converted_dataset["expressions"] == [[1.0, 2.0, 3.0], [4.0, 6.0]]


def test_databank_load_multiple_anndata_layers():
    adata = AnnData(
        X=np.array([[1.0, 2.0, 3.0], [4.0, 0.0, 6.0]]),
        obs={"cell": ["cell1", "cell2"], "study": ["study1", "study2"]},
        var={"gene": ["gene_a", "gene_b", "gene_c"]},
        layers={"layer1": np.array([[1.0, 2.0, 3.0], [4.0, 0.0, 6.0]])},
    )
    gene_vocab = {"gene_a": 1, "gene_b": 0, "gene_c": 2}

    db = DataBank(meta_info=MetaInfo(), gene_vocab=GeneVocab.from_dict(gene_vocab))
    data_tables = db.load_anndata(adata, token_col="gene")
    assert len(data_tables) == 2
    assert data_tables[0].name == "X"
    assert data_tables[1].name == "layer1"

    converted_dataset = data_tables[0].data
    assert converted_dataset["id"] == [0, 1]
    assert converted_dataset["genes"] == [[1, 0, 2], [1, 2]]
    assert converted_dataset["expressions"] == [[1.0, 2.0, 3.0], [4.0, 6.0]]

    converted_dataset = data_tables[1].data
    assert converted_dataset["id"] == [0, 1]
    assert converted_dataset["genes"] == [[1, 0, 2], [1, 2]]
    assert converted_dataset["expressions"] == [[1.0, 2.0, 3.0], [4.0, 6.0]]
