import pickle
import tempfile
from pathlib import Path

import pytest
from scgpt.tokenizer import GeneVocab, get_default_gene_vocab
from scgpt.tokenizer.vocab_compat import Vocab, convert_legacy_vocab


def test_gene_vocab():
    gene_vocab = GeneVocab([])
    assert len(gene_vocab) == 0

    gene_vocab.append_token("abc")
    assert len(gene_vocab) == 1

    gene_vocab = GeneVocab(gene_vocab)
    assert gene_vocab["abc"] == 0

    gene_vocab = GeneVocab(["def", "g", "h"], specials=["a", "b", "c"])
    assert gene_vocab["a"] == 0
    assert gene_vocab["def"] == 3
    assert len(gene_vocab) == 6

    gene_vocab = GeneVocab(["a"], specials=["<pad>"], special_first=False)
    assert gene_vocab["<pad>"] == 1
    assert gene_vocab.get_default_index() == 1


def test_gene_vocab_from_dict():
    gene_vocab = GeneVocab.from_dict({"a": 0, "b": 1, "c": 2})
    assert len(gene_vocab) == 3
    assert gene_vocab["a"] == 0
    assert gene_vocab["c"] == 2


def test_gene_vocab_from_file():
    test_file = Path(__file__).parent / "vocab.json"
    gene_vocab = GeneVocab.from_file(test_file)
    assert len(gene_vocab) == 3


def test_gene_vocab_pad_token():
    gene_vocab = GeneVocab(["a", "b", "c"], specials=["<pad>"])
    assert gene_vocab.pad_token is None
    gene_vocab.pad_token = "<pad>"
    assert gene_vocab.pad_token == "<pad>"


def test_get_default_gene_vocab():
    gene_vocab = get_default_gene_vocab()
    assert gene_vocab["A12M1"] == 0


# ---------------------------------------------------------------------------
# Vocab tests
# ---------------------------------------------------------------------------


def test_vocab_basic():
    v = Vocab(["a", "b", "c"])
    assert len(v) == 3
    assert v["a"] == 0
    assert v["c"] == 2
    assert v[0] == "a"
    assert "b" in v
    assert "z" not in v
    assert v(["a", "c"]) == [0, 2]


def test_vocab_default_index():
    v = Vocab(["a", "b"], default_index=0)
    assert v["unknown"] == 0
    v.set_default_index(None)
    with pytest.raises(KeyError):
        _ = v["unknown"]


def test_vocab_append_insert():
    v = Vocab(["a", "b"])
    v.append_token("c")
    assert v["c"] == 2
    assert v.append_token("a") == 0

    v.insert_token("z", 0)
    assert v["z"] == 0
    assert v["a"] == 1
    assert len(v) == 4


def test_vocab_stoi_itos_roundtrip():
    tokens = ["<pad>", "<cls>", "BRCA1", "TP53"]
    v = Vocab(tokens)
    assert v.get_itos() == tokens
    stoi = v.get_stoi()
    for tok, idx in stoi.items():
        assert v[tok] == idx


# ---------------------------------------------------------------------------
# Legacy pickle compatibility
# ---------------------------------------------------------------------------


class _MockLegacyVocab:
    """Simulates a torchtext Vocab object for pickle testing."""

    def __init__(self, stoi):
        self._stoi = stoi
        self._itos = [t for t, _ in sorted(stoi.items(), key=lambda x: x[1])]
        self._default_index = -1

    def get_stoi(self):
        return dict(self._stoi)

    def get_itos(self):
        return list(self._itos)

    def get_default_index(self):
        return self._default_index


class _MockWrappedVocab:
    """Simulates a torchtext Vocab with .vocab attribute."""

    def __init__(self, inner):
        self.vocab = inner


def test_convert_legacy_vocab_dict():
    v = convert_legacy_vocab({"a": 0, "b": 1, "c": 2})
    assert isinstance(v, Vocab)
    assert v["a"] == 0
    assert len(v) == 3


def test_convert_legacy_vocab_with_get_stoi():
    mock = _MockLegacyVocab({"BRCA1": 0, "TP53": 1})
    v = convert_legacy_vocab(mock)
    assert v["BRCA1"] == 0
    assert v["TP53"] == 1
    assert v.get_default_index() is None  # -1 sentinel should become None


def test_convert_legacy_vocab_wrapped():
    inner = _MockLegacyVocab({"G1": 0, "G2": 1})
    wrapped = _MockWrappedVocab(inner)
    v = convert_legacy_vocab(wrapped)
    assert v["G1"] == 0
    assert len(v) == 2


def test_from_file_pickle_legacy():
    """GeneVocab.from_file handles pickled legacy vocab objects."""
    mock = _MockLegacyVocab({"<pad>": 0, "GENE1": 1, "GENE2": 2})
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        pickle.dump(mock, f)
        pkl_path = Path(f.name)
    try:
        gv = GeneVocab.from_file(pkl_path)
        assert len(gv) == 3
        assert gv["<pad>"] == 0
        assert gv["GENE2"] == 2
    finally:
        pkl_path.unlink()


def test_from_file_pickle_gene_vocab():
    """GeneVocab.from_file handles pickled GeneVocab objects directly."""
    original = GeneVocab(["BRCA1", "TP53"], specials=["<pad>"])
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        pickle.dump(original, f)
        pkl_path = Path(f.name)
    try:
        loaded = GeneVocab.from_file(pkl_path)
        assert len(loaded) == len(original)
        assert loaded["<pad>"] == original["<pad>"]
        assert loaded["BRCA1"] == original["BRCA1"]
    finally:
        pkl_path.unlink()
