import pytest
from pathlib import Path
from scgpt.tokenizer import GeneVocab, get_default_gene_vocab
from scgpt.tokenizer.vocab_compat import (
    BuiltinVocab,
    gene_vocab_base_backend,
    torchtext_import_succeeded,
    get_vocab_info,
    is_torchtext_vocab,
    from_torchtext_vocab,
)


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
    # current file path
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
# vocab_compat tests
# ---------------------------------------------------------------------------


def test_gene_vocab_base_backend_is_set():
    """gene_vocab_base_backend should be 'torchtext' or 'builtin'."""
    assert gene_vocab_base_backend in ("torchtext", "builtin")
    assert isinstance(get_vocab_info(), str) and len(get_vocab_info()) > 0


def test_builtin_vocab_basic():
    """Core Vocab API works independently of torchtext."""
    v = BuiltinVocab(["a", "b", "c"])
    assert len(v) == 3
    assert v["a"] == 0
    assert v["c"] == 2
    assert v[0] == "a"
    assert "b" in v
    assert "z" not in v
    assert v(["a", "c"]) == [0, 2]


def test_builtin_vocab_default_index():
    v = BuiltinVocab(["a", "b"], default_index=0)
    # OOV token should return default index, not raise
    assert v["unknown"] == 0
    v.set_default_index(None)
    with pytest.raises(KeyError):
        _ = v["unknown"]


def test_builtin_vocab_append_insert():
    v = BuiltinVocab(["a", "b"])
    v.append_token("c")
    assert v["c"] == 2
    # append existing is a no-op
    assert v.append_token("a") == 0

    v.insert_token("z", 0)
    assert v["z"] == 0
    assert v["a"] == 1   # shifted
    assert len(v) == 4


@pytest.mark.skipif(not torchtext_import_succeeded, reason="torchtext not installed")
def test_is_torchtext_vocab():
    """is_torchtext_vocab correctly identifies torchtext Vocab objects."""
    from torchtext.vocab import vocab as build_tt_vocab
    from collections import OrderedDict

    tt_vocab = build_tt_vocab(OrderedDict([("G1", 1), ("G2", 1)]))
    assert is_torchtext_vocab(tt_vocab)
    assert not is_torchtext_vocab(BuiltinVocab(["G1", "G2"]))


@pytest.mark.skipif(not torchtext_import_succeeded, reason="torchtext not installed")
def test_from_torchtext_vocab_conversion():
    """from_torchtext_vocab converts token/index mapping and default index."""
    from torchtext.vocab import vocab as build_tt_vocab
    from collections import OrderedDict

    tokens = OrderedDict([("<pad>", 2), ("<cls>", 2), ("BRCA1", 2), ("TP53", 2)])
    tt_vocab = build_tt_vocab(tokens, specials=list(tokens.keys()))
    tt_vocab.set_default_index(tt_vocab["<pad>"])

    converted = from_torchtext_vocab(tt_vocab)
    assert isinstance(converted, BuiltinVocab)
    for tok in tt_vocab.get_itos():
        assert converted[tok] == tt_vocab[tok]
    # default index must be propagated (torchtext -1 sentinel handled)
    assert converted.get_default_index() == tt_vocab.get_default_index()


@pytest.mark.skipif(not torchtext_import_succeeded, reason="torchtext not installed")
def test_gene_vocab_init_from_torchtext():
    """GeneVocab() accepts a torchtext Vocab object directly."""
    from torchtext.vocab import vocab as build_tt_vocab
    from collections import OrderedDict

    tokens = OrderedDict([("BRCA1", 1), ("TP53", 1), ("EGFR", 1)])
    tt_vocab = build_tt_vocab(tokens)

    gv = GeneVocab(tt_vocab)
    assert len(gv) == 3
    for tok in tt_vocab.get_itos():
        assert gv[tok] == tt_vocab[tok]

